#!/usr/bin/env python3
"""Intercept every matmul-like aten op during the Moonshine-tiny encoder forward pass
and compare the total MAC count against STATS (4,336,487,440).

usage: python3 python/probe_moonshine_macs.py [--samples N]   (default: 160000 = 10 s)
Run from the repo root after `source env.sh`.

Ops captured: aten.mm, aten.bmm, aten.addmm (bias ignored), aten.convolution.
All shapes and per-op MAC counts are printed.  The total is compared against
our STATS counter and the teammate's 4.4026 GMAC.

If any op appears here but not in our matmul_calls=136, its MACs are the miss.
"""
import argparse, collections, sys, torch
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import MoonshineConfig, MoonshineModel

OUR_STATS_MACS   = 4_336_487_440   # from STATS matmul_calls=136
TEAMMATE_MACS    = 4_402_600_000   # from the teammate's workload study (4.4026 G)


class MacMode(TorchDispatchMode):
    def __init__(self):
        self.rows = []           # (op_short, shapes, macs)
        self.by_op = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = str(func)

        macs = 0
        shapes = []

        if "convolution" in name:
            # args: (input, weight, bias, stride, padding, dilation, transposed, out_padding, groups)
            inp, wt = args[0], args[1]
            out_spatial = 1
            for d in out.shape[2:]: out_spatial *= d
            kernel_elems = 1
            for k in wt.shape[2:]: kernel_elems *= k
            macs = int(wt.shape[0]) * int(wt.shape[1]) * kernel_elems * out_spatial
            shapes = [list(inp.shape), list(wt.shape)]

        elif "addmm" in name:
            # addmm(bias, A, B): A=[M,K], B=[K,N]
            A, B = args[1], args[2]
            macs = int(A.shape[0]) * int(A.shape[1]) * int(B.shape[1])
            shapes = [list(A.shape), list(B.shape)]

        elif name == "aten.mm.default":
            A, B = args[0], args[1]
            macs = int(A.shape[0]) * int(A.shape[1]) * int(B.shape[1])
            shapes = [list(A.shape), list(B.shape)]

        elif "bmm" in name:
            A, B = args[0], args[1]
            macs = int(A.shape[0]) * int(A.shape[1]) * int(A.shape[2]) * int(B.shape[2])
            shapes = [list(A.shape), list(B.shape)]

        elif name == "aten.matmul.default":
            # Used by nn.Linear on ≥3D inputs (no bias → not addmm).
            # e.g. attention projections: [1, 415, 288] @ [288, 288] → [1, 415, 288]
            # General broadcast matmul: MACs = product(batch dims) * M * K * N
            A, B = args[0], args[1]
            if A.dim() >= 2 and B.dim() == 2:
                # batched: [..., M, K] @ [K, N]
                M = A.shape[-2] if A.dim() >= 2 else 1
                K = A.shape[-1]
                N = B.shape[-1]
                batch = 1
                for d in A.shape[:-2]: batch *= d
                macs = batch * M * K * N
            elif A.dim() == 2 and B.dim() >= 2:
                M, K = A.shape
                N = B.shape[-1]
                batch = 1
                for d in B.shape[:-2]: batch *= d
                macs = batch * M * K * N
            elif A.dim() >= 3 and B.dim() >= 3:
                # batched matmul: [..., M, K] @ [..., K, N]
                M, K, N = A.shape[-2], A.shape[-1], B.shape[-1]
                batch = 1
                for d in A.shape[:-2]: batch *= d
                macs = batch * M * K * N
            else:
                macs = 0
            shapes = [list(A.shape), list(B.shape)]

        elif "linear" in name:
            # fallback: aten.linear (if not already decomposed to addmm/matmul)
            inp, wt = args[0], args[1]
            M = 1
            for d in inp.shape[:-1]: M *= d
            macs = M * int(inp.shape[-1]) * int(wt.shape[0])
            shapes = [list(inp.shape), list(wt.shape)]

        if macs > 0:
            short = name.split(".")[-2] if "." in name else name
            self.rows.append((short, shapes, macs))
            self.by_op[short] += macs

        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=160_000)
    a = ap.parse_args()

    torch.manual_seed(0)
    cfg = MoonshineConfig()
    cfg._attn_implementation = "eager"
    enc = MoonshineModel(cfg).encoder.eval()
    for m_ in enc.modules():
        m_._non_persistent_buffers_set.clear()

    class Wrap(torch.nn.Module):
        def __init__(s, e): super().__init__(); s.e = e
        def forward(s, x): return s.e(x).last_hidden_state

    model = Wrap(enc)
    x = torch.randn(1, a.samples)

    mode = MacMode()
    with torch.no_grad(), mode:
        _ = model(x)

    print(f"\n{'op':14s}  {'shapes':60s}  {'MACs':>15s}")
    print("-" * 97)
    total = 0
    for short, shapes, macs in mode.rows:
        sh_str = " @ ".join(str(s) for s in shapes)
        print(f"{short:14s}  {sh_str:60s}  {macs:>15,}")
        total += macs

    print("-" * 97)
    print(f"\nTotal intercepted:  {total:>20,}")
    print(f"Our STATS (136 calls): {OUR_STATS_MACS:>16,}   diff = {total - OUR_STATS_MACS:+,}")
    print(f"Teammate (4.4026G):    {TEAMMATE_MACS:>16,}   diff = {total - TEAMMATE_MACS:+,}")

    print(f"\nBy op type:")
    for op, m in sorted(mode.by_op.items(), key=lambda x: -x[1]):
        print(f"  {op:14s}  {m:>18,}  ({100*m/total:.1f}%)")

    # Highlight any op that might not be rewritten to linalg.matmul
    print(f"\nTotal unique shapes: {len(mode.rows)}")
    print("(compare against 136 matmul_calls from STATS; any op here not in gen_shims.py is a miss)")


if __name__ == "__main__":
    main()
