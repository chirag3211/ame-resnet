#!/usr/bin/env python3
"""Export ResNet18 -> MLIR, plus a fixed input and PyTorch's golden logits.

Outputs (in --out, default ./build):
  input.bin           float32 NCHW input, seeded
  golden_logits.bin   float32 [1,1000], computed by PyTorch (CPU, fp32)
  meta.json           shapes, seed, top-1 index
  resnet18.mlir       MLIR from torch-mlir (skipped with --no-mlir)

Weights are RANDOM (seeded) by default so nothing needs downloading and the
comparison is still meaningful. Use --pretrained for ImageNet weights.
Use a small --size (e.g. 64) while bringing things up: Spike is slow.
"""
import argparse
import json
import os

import numpy as np
import torch
import torchvision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=224, help="input H=W (224 for the real thing)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--out", default="build")
    ap.add_argument("--output-type", default="linalg-on-tensors",
                    help="torch-mlir output type (linalg-on-tensors | tosa | torch ...)")
    ap.add_argument("--no-mlir", action="store_true", help="only write input + golden")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    weights = "DEFAULT" if args.pretrained else None
    model = torchvision.models.resnet18(weights=weights).eval()
    x = torch.randn(1, 3, args.size, args.size)

    with torch.no_grad():
        y = model(x)
    x.numpy().astype(np.float32).tofile(os.path.join(args.out, "input.bin"))
    y.numpy().astype(np.float32).tofile(os.path.join(args.out, "golden_logits.bin"))
    meta = {
        "size": args.size, "seed": args.seed, "pretrained": args.pretrained,
        "input_shape": list(x.shape), "output_shape": list(y.shape),
        "top1": int(y.argmax().item()),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("golden written:", meta)

    if args.no_mlir:
        return
    try:
        from torch_mlir import fx
    except ImportError:
        raise SystemExit(
            "torch_mlir is not installed. Install it (or Buddy's frontend) and rerun,\n"
            "or use --no-mlir to produce only the golden files."
        )
    with torch.no_grad():
        module = fx.export_and_import(
            model, x, output_type=args.output_type, func_name="forward"
        )
    path = os.path.join(args.out, "resnet18.mlir")
    with open(path, "w") as f:
        f.write(str(module))
    print("MLIR written:", path, f"({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
