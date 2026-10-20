#!/usr/bin/env python3
"""numpy-only check of a run's `result N ...` line against the stored TF fp16 softmax (reference/ref_softmax_fp16.bin).
   usage: check_result.py <run_output.txt> <ref_softmax_fp16.bin>      exit 0 = PASS"""
import sys, numpy as np
line = [l for l in open(sys.argv[1]) if l.startswith("result")]
if not line: sys.exit("no `result N ...` line in " + sys.argv[1] + " (did the run finish?)")
t = line[0].split(); N = int(t[1])
mine = np.array([int(x) for x in t[2].strip(",").split(",")], dtype=np.uint16).view(np.float16).astype(np.float32)
ref = np.fromfile(sys.argv[2], dtype=np.float16).reshape(-1, 1001)[N].astype(np.float32)
d = np.abs(mine - ref)
viol = int((d > 1e-2 + 1e-2 * np.abs(ref)).sum())            # per element, like numpy.allclose(rtol=1e-2, atol=1e-2)
ok = mine.argmax() == ref.argmax() and viol == 0
print(f"image {N}: top1 mine={mine.argmax()} ref={ref.argmax()} | max|dp|={d.max():.3e} L1={d.sum():.3e} sum={mine.sum():.4f} | top5 mine={np.argsort(-mine)[:5].tolist()} ref={np.argsort(-ref)[:5].tolist()}")
print(f"allclose(rtol=1e-2, atol=1e-2) violations: {viol} of 1001")
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
