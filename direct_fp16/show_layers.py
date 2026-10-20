#!/usr/bin/env python3
"""Print a per-layer / per-kind instruction breakdown from results_direct.jsonl.
   usage: show_layers.py [results_direct.jsonl] [row_index=-1]"""
import sys, json
def cd(a, b): return -(-a // b)
path = sys.argv[1] if len(sys.argv) > 1 else "results_direct.jsonl"
rows = [json.loads(l) for l in open(path)]
r = rows[int(sys.argv[2]) if len(sys.argv) > 2 else -1]; S = r["summary"]; L = r["layers"]
print(f"image {r['image']}  KC={r['kc']}  {r['result']}  git {r['git']}{'+dirty' if r['dirty'] else ''}  {r['ts']}  {r['note']}")
tot = S["total"]
print(f"\n{'category':<10}{'instret':>14}{'%':>7}")
for k in ("im2col", "matmul", "fc_matmul", "bn", "relu", "add", "maxpool", "avgpool", "tail"):
    print(f"{k:<10}{S[k]:>14,}{100 * S[k] / tot:>7.1f}")
print(f"{'total':<10}{tot:>14,}")
print(f"\n{'#':>2} {'layer':<20}{'k':>2}{'s':>2}{'M':>6}{'N':>5}{'K':>5} {'mfma*':>8}{'im2col':>11}{'matmul':>11}{'mm/mfma':>8}{'bn+relu+add':>13}")
kinds = {}
for d in L:
    mf = cd(d["M"], 8) * cd(d["N"], 8) * cd(d["K"], 8)
    ew = d["bn"] + d["relu"] + d["add"]
    print(f"{d['idx']:>2} {d['name']:<20}{d['k']:>2}{d['s']:>2}{d['M']:>6}{d['N']:>5}{d['K']:>5} {mf:>8,}{d['im2col']:>11,}{d['matmul']:>11,}{d['matmul'] / mf:>8.1f}{ew:>13,}")
    key = f"{d['k']}x{d['k']} s{d['s']}" + (" shortcut" if "shortcut" in d["name"] else "")
    a = kinds.setdefault(key, [0, 0, 0, 0]); a[0] += 1; a[1] += mf; a[2] += d["matmul"]; a[3] += d["im2col"]
print("\nby conv kind:  count  mfma*  matmul-instr  mm/mfma   im2col-instr")
for k, (n, mf, mm, im) in sorted(kinds.items()): print(f"  {k:<16}{n:>5}{mf:>11,}{mm:>14,}{mm / mf:>9.1f}{im:>14,}")
print("\n* mfma = tile-ideal ceil(M/8)*ceil(N/8)*ceil(K/8) (8x8x8 fp16 tile); the host emulation confirmed this equals the executed count.")
