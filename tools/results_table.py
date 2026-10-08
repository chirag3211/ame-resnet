#!/usr/bin/env python3
"""Summarise results.jsonl: one row per model dir with the latest PASSing AME and scalar record.
usage: results_table.py [results.jsonl] [--json] [--all]
Prints a markdown table (warm RUN1 unless a dir has only RUN0) and sanity warnings:
  - 'everything else' (forward - in_matmul - in_copy) must be IDENTICAL in the AME and scalar run of one dir
    (it is in every pair checked so far); a mismatch means the two runs used different pipelines/drivers.
  - different git commits, dirty trees, or different LLVM_OPT between the two runs.
Columns: whole = scalar/AME forward; matmul = scalar/AME in_matmul; n = non-matmul share of the AME forward
(copy included); check = n + R(1-n) from the identity in STATE.md (should equal whole)."""
import json, sys

def load(path):
    recs = []
    for ln in open(path):
        ln = ln.strip()
        if ln: recs.append(json.loads(ln))
    return recs

def pick(recs):
    best = {}
    for r in recs:                      # later lines win
        if r.get("pass") and r.get("runs"): best[(r["dir"], r["backend"])] = r
    return best

def run(r, n):
    return r["runs"].get(n)

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else "results.jsonl"
    best = pick(load(path))
    dirs = sorted({d for d, _ in best})
    rows, warns = [], []
    for d in dirs:
        a, s = best.get((d, "ame")), best.get((d, "scalar"))
        pr = lambda r: (run(r, "1"), "1") if run(r, "1") else (run(r, "0"), "0")
        row = dict(dir=d, model=(a or s)["model"])
        if a:
            ra, na = pr(a); row.update(run=na, ame_forward=ra["forward"], ame_matmul=ra["in_matmul"], ame_copy=ra["in_copy"],
                                       macs=a.get("stats", {}).get("macs"))
            row["n"] = (ra["forward"] - ra["in_matmul"]) / ra["forward"]
            if a.get("stats", {}).get("macs"): row["ame_instr_per_mac"] = ra["in_matmul"] / a["stats"]["macs"]
        if s:
            rs, ns = pr(s); row.update(scalar_forward=rs["forward"], scalar_matmul=rs["in_matmul"])
            if s.get("stats", {}).get("macs"): row["scalar_instr_per_mac"] = rs["in_matmul"] / s["stats"]["macs"]
        if a and s:
            if ra["everything_else"] != rs["everything_else"] or ra["in_copy"] != rs["in_copy"]:
                warns.append(f"{d}: 'everything else' differs AME {ra['everything_else']} vs scalar {rs['everything_else']} "
                             f"(copy {ra['in_copy']} vs {rs['in_copy']}): different pipeline/driver?")
            if a.get("git_commit") != s.get("git_commit"): warns.append(f"{d}: AME commit {a.get('git_commit')} != scalar {s.get('git_commit')}")
            if a.get("flags", {}).get("llvm_opt") != s.get("flags", {}).get("llvm_opt"): warns.append(f"{d}: LLVM_OPT differs between the AME and scalar runs")
            if ns != na: warns.append(f"{d}: AME used RUN{na}, scalar RUN{ns}")
            row["whole"] = rs["forward"] / ra["forward"]; row["matmul"] = rs["in_matmul"] / ra["in_matmul"]
            row["check"] = row["n"] + row["matmul"] * (1 - row["n"])
            r0a, r0s = run(a, "0"), run(s, "0")
            if r0a and r0s: row["cold"] = r0s["forward"] / r0a["forward"]
        for r in (a, s):
            if r and r.get("git_dirty"): warns.append(f"{d}: {r['backend']} record made on a dirty git tree")
        rows.append(row)
    if "--json" in sys.argv:
        print(json.dumps(rows, indent=1)); return
    f = lambda v, p: "-" if v is None else format(v, p)
    print("| model (dir) | AME fwd | scalar fwd | whole | matmul | non-matmul n | check | cold/cold | scalar instr/MAC |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {f(r.get('ame_forward'), ',')} | {f(r.get('scalar_forward'), ',')} | {f(r.get('whole'), '.2f')}x "
              f"| {f(r.get('matmul'), '.2f')}x | {f(r.get('n') and r['n'] * 100, '.1f')}% | {f(r.get('check'), '.2f')}x "
              f"| {f(r.get('cold'), '.2f')}x | {f(r.get('scalar_instr_per_mac'), '.3f')} |")
    for w in warns: print("WARNING:", w, file=sys.stderr)

if __name__ == "__main__": main()
