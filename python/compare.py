#!/usr/bin/env python3
"""Compare logits printed by the Spike run against PyTorch's golden logits.

The driver prints lines like:   LOGIT 17 0x3f8a1b2c   (raw float32 bits)
Usage: compare.py spike_output.txt build/golden_logits.bin [--tol 1e-3]
"""
import argparse
import re
import struct
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("spike_out")
ap.add_argument("golden")
ap.add_argument("--tol", type=float, default=1e-3, help="max abs error allowed")
args = ap.parse_args()

gold = np.fromfile(args.golden, dtype=np.float32)
got = np.full_like(gold, np.nan)
for line in open(args.spike_out, errors="replace"):
    m = re.match(r"\s*LOGIT\s+(\d+)\s+0x([0-9a-fA-F]{8})", line)
    if m:
        got[int(m.group(1))] = struct.unpack("<f", struct.pack("<I", int(m.group(2), 16)))[0]

missing = int(np.isnan(got).sum())
if missing:
    sys.exit(f"FAIL: {missing}/{len(gold)} logits missing from the Spike output")
err = float(np.max(np.abs(got - gold)))
t1_got, t1_gold = int(got.argmax()), int(gold.argmax())
print(f"max abs err = {err:.3e}   top1 spike={t1_got} pytorch={t1_gold}")
ok = (t1_got == t1_gold) and err <= args.tol
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
