#!/usr/bin/env python3
"""Parse one resnet-spike log and append ONE JSON line to results.jsonl (no hand copying).

usage: record.py LOG --dir DIR --backend ame|scalar [--set key=value ...] [--wall SECONDS] [--rc N]
                     [--note TEXT] [--backfill] [--git-commit SHA] [--out results.jsonl]
--backfill: the log comes from an earlier run, so the current HEAD / dirty flag / time say nothing about it.
            git_commit, git_dirty and ts are written as null (ingestion time goes to ingested_ts) and the note
            gets 'backfilled'. Give --git-commit SHA only if you KNOW the commit the run was built from.
The log is the output of `make ... resnet-spike` (stdout+stderr, LOGIT lines optional): RUNn INSTRET lines,
TOP1, STATS, the compare script's 'max abs err' line and PASS/FAIL. --set records flags (llvm_opt=O3 isa=... ...).
Derived per run (same definition as STATE.md): everything_else = forward - in_matmul - in_copy."""
import argparse, json, os, re, subprocess, sys, time

def git(*a):
    try: return subprocess.check_output(["git", *a], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception: return None

def parse(text):
    r = {"runs": {}, "errors": []}
    for ln in text.splitlines():
        m = re.match(r"RUN(\d+) INSTRET forward=(\d+) in_matmul=(\d+) in_copy=(\d+) copy_calls=(\d+) ksteps=(\d+) otiles=(\d+)", ln)
        if m:
            n, f, mm, cp, cc, ks, ot = m.groups()
            f, mm, cp = int(f), int(mm), int(cp)
            r["runs"][n] = dict(forward=f, in_matmul=mm, in_copy=cp, copy_calls=int(cc), ksteps=int(ks),
                                otiles=int(ot), everything_else=f - mm - cp)
            continue
        m = re.match(r"STATS backend=(\S+) matmul_calls=(\d+) macs=(\d+)", ln)
        if m: r["stats"] = dict(backend=m[1], matmul_calls=int(m[2]), macs=int(m[3])); continue
        m = re.match(r"TOP1 (\d+)", ln)
        if m: r["top1"] = int(m[1]); continue
        if "max abs err" in ln:
            r["compare_line"] = ln.strip()
            m = re.search(r"max abs err[^=]*=\s*([0-9.eE+-]+)", ln)
            if m: r["max_abs_err"] = float(m[1])
            continue
        if ln.strip() in ("PASS", "FAIL"): r["verdict"] = ln.strip(); continue
        if "rror" in ln and len(r["errors"]) < 5: r["errors"].append(ln.strip()[:200])
    r["pass"] = r.get("verdict") == "PASS"
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log"); ap.add_argument("--dir", required=True)
    ap.add_argument("--backend", required=True, choices=["ame", "scalar"])
    ap.add_argument("--set", action="append", default=[]); ap.add_argument("--wall", type=float)
    ap.add_argument("--rc", type=int); ap.add_argument("--note", default="")
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--git-commit", default=None)
    a = ap.parse_args()
    if a.git_commit and not a.backfill: ap.error("--git-commit only makes sense with --backfill")
    rec = parse(open(a.log, errors="replace").read())
    st = rec.get("stats", {}).get("backend")
    if st and (st == "AME") != (a.backend == "ame"):
        sys.exit(f"record.py: log says STATS backend={st} but --backend {a.backend}; not recording")
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if a.backfill:      # provenance of the ORIGINAL run is unknown: never stamp HEAD / dirty / now onto it
        commit, dirty, ts = a.git_commit, None, None
        if "backfilled" not in a.note: a.note = (a.note + " backfilled").strip()
    else:
        commit, dirty, ts = git("rev-parse", "--short", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no")), now
    out = dict(ts=ts, ingested_ts=now,
               model=re.sub(r"^build_", "", os.path.basename(a.dir.rstrip("/"))), dir=a.dir, backend=a.backend,
               flags=dict(kv.split("=", 1) for kv in a.set), git_commit=commit,
               git_dirty=dirty, wall_s=a.wall, make_rc=a.rc, log=a.log, note=a.note, **rec)
    with open(a.out, "a") as f: f.write(json.dumps(out) + "\n")
    r1 = out["runs"].get("1") or out["runs"].get("0") or {}
    print(f"recorded {out['model']} {a.backend}: forward={r1.get('forward')} matmul={r1.get('in_matmul')} "
          f"{'PASS' if out['pass'] else 'FAIL/none'} -> {a.out}")
    return 0 if out["pass"] else 1

if __name__ == "__main__": sys.exit(main())
