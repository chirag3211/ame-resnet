#!/usr/bin/env python3
"""Compare the SAMP/SUMSQ lines of a Spike run (see gen_driver.py) with golden float32 output.
usage: compare_tensor.py spike_out.txt golden.bin [--tol 1e-3]   (same CLI as compare.py)"""
import argparse, re, struct, sys
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("spike_out"); ap.add_argument("golden")
ap.add_argument("--tol", type=float, default=1e-3); a = ap.parse_args()
g = np.fromfile(a.golden, dtype=np.float32)
n = g.size; ns = min(n, 2048); step = n // ns
samp, ss, outn, pj = {}, None, None, None
for line in open(a.spike_out, errors="replace"):
    m = re.match(r"\s*SAMP\s+(\d+)\s+0x([0-9a-fA-F]{8})", line)
    if m: samp[int(m.group(1))] = struct.unpack("<f", struct.pack("<I", int(m.group(2), 16)))[0]
    m = re.match(r"\s*SUMSQ\s+(\S+)", line)
    if m: ss = float(m.group(1))
    m = re.match(r"\s*PROJ\s+(\S+)", line)
    if m: pj = float(m.group(1))
    m = re.match(r"\s*OUTN\s+(\d+)", line)
    if m: outn = int(m.group(1))
idx = [s * step for s in range(ns)]
if outn != n or any(i not in samp for i in idx) or ss is None or pj is None:
    sys.exit(f"FAIL: output incomplete (OUTN={outn}, expected {n}; samples {len(samp)}/{ns})")
got = np.array([samp[i] for i in idx], dtype=np.float64); ref = g[idx].astype(np.float64)
err = float(np.max(np.abs(got - ref)))
rs = float(np.sum(g.astype(np.float64) ** 2)); rel = abs(ss - rs) / max(rs, 1e-30)
w = (((np.arange(n, dtype=np.uint64) * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)) >> np.uint64(22)).astype(np.float64)
w = (w - 512.0) / 512.0
gw = g.astype(np.float64) * w
prel = abs(pj - float(gw.sum())) / max(float(np.sqrt((gw ** 2).sum())), 1e-30)
print(f"max abs err (samples) = {err:.3e}  |gold|max = {float(np.max(np.abs(g))):.3g}  sumsq rel err = {rel:.2e}  projection err = {prel:.2e}")
ok = np.isfinite(got).all() and err <= a.tol and rel <= 1e-4 and prel <= 1e-4
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
