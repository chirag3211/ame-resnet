#!/usr/bin/env python3
"""Append one row to results_direct.jsonl from a Spike run's output (LAYER/SUMMARY lines) + the check result.
   usage: record_direct.py <run_output.txt> <PASS|FAIL> [note]       (kept apart from ../results.jsonl: different schema)"""
import sys, json, time, subprocess, re
txt = open(sys.argv[1]).read()
hdr = re.search(r"backend=(\S+) KC=(\d+) image=(\d+)", txt)
if not hdr or hdr[1] != "spike-hw": sys.exit("not a Spike run (backend=%s): instret would be 0, not recording" % (hdr[1] if hdr else None))
s = re.search(r"^SUMMARY,(.*)$", txt, re.M)
if not s: sys.exit("no SUMMARY line: the run did not finish, not recording")
summ = {k: int(v) for k, v in (kv.split("=") for kv in s[1].split(","))}
layers = []
for m in re.finditer(r"^LAYER,(\d+),([^,]+),(.*)$", txt, re.M):
    d = {k: int(v) for k, v in (kv.split("=") for kv in m[3].split(","))}; d.update(idx=int(m[1]), name=m[2]); layers.append(d)
def sh(*a): return subprocess.run(a, capture_output=True, text=True).stdout.strip()
row = dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), model="r50v2_fp16_direct", backend="ame", kc=int(hdr[2]), image=int(hdr[3]),
           result=sys.argv[2], git=sh("git", "rev-parse", "--short", "HEAD"), dirty=bool(sh("git", "status", "--porcelain")),
           note=sys.argv[3] if len(sys.argv) > 3 else "", summary=summ, layers=layers)
open("results_direct.jsonl", "a").write(json.dumps(row) + "\n")
print(f"recorded: total={summ['total']:,} matmul={summ['matmul']:,} im2col={summ['im2col']:,} (image {row['image']}, KC {row['kc']}, {row['result']}) -> results_direct.jsonl")
