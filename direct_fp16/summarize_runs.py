#!/usr/bin/env python3
"""Table of all Spike runs found in build/kc*/net_img*.txt: instruction counts + accuracy vs the TF reference.
   usage: summarize_runs.py [ref_softmax_fp16.bin]      (run from direct_fp16/)"""
import sys, re, glob, numpy as np
ref_all = np.fromfile(sys.argv[1] if len(sys.argv) > 1 else "reference/ref_softmax_fp16.bin", dtype=np.float16).reshape(-1, 1001).astype(np.float32)
rows = []
for path in glob.glob("build/kc*/net_img*.txt"):
    t = open(path).read()
    h = re.search(r"backend=(\S+) KC=(\d+) image=(\d+)", t); s = re.search(r"^SUMMARY,(.*)$", t, re.M); r = re.search(r"^result (\d+) (\S+)", t, re.M)
    if not (h and s and r) or h[1] != "spike-hw": continue
    S = {k: int(v) for k, v in (kv.split("=") for kv in s[1].split(","))}
    mine = np.array([int(x) for x in r[2].strip(",").split(",")], dtype=np.uint16).view(np.float16).astype(np.float32)
    ref = ref_all[int(r[1])]; d = np.abs(mine - ref)
    rows.append(dict(kc=int(h[2]), img=int(h[3]), total=S["total"], matmul=S["matmul"] + S["fc_matmul"], top1=bool(mine.argmax() == ref.argmax()),
                     maxdp=float(d.max()), l1=float(d.sum()), viol=int((d > 1e-2 + 1e-2 * np.abs(ref)).sum())))
if not rows: sys.exit("no Spike runs found in build/kc*/net_img*.txt")
rows.sort(key=lambda x: (x["kc"], x["img"]))
kcn = lambda k: "none" if k > 10**6 else str(k)
print(f"{'KC':>6}{'img':>4}{'total instret':>16}{'matmul+fc':>14}{'top1':>6}{'max|dp|':>10}{'L1':>10}{'viol':>6}")
for x in rows: print(f"{kcn(x['kc']):>6}{x['img']:>4}{x['total']:>16,}{x['matmul']:>14,}{'ok' if x['top1'] else 'NO':>6}{x['maxdp']:>10.2e}{x['l1']:>10.2e}{x['viol']:>6}")
print(f"\n{'KC':>6}{'#img':>5}{'mean total':>14}{'mean max|dp|':>14}{'mean L1':>10}{'top1 ok':>9}{'strict pass':>13}")
for kc in sorted({x["kc"] for x in rows}):
    g = [x for x in rows if x["kc"] == kc]
    print(f"{kcn(kc):>6}{len(g):>5}{np.mean([x['total'] for x in g]):>14,.0f}{np.mean([x['maxdp'] for x in g]):>14.2e}{np.mean([x['l1'] for x in g]):>10.2e}{sum(x['top1'] for x in g):>6}/{len(g)}{sum(x['viol'] == 0 for x in g):>9}/{len(g)}")
