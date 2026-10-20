#!/usr/bin/env python3
"""Count matrix-extension instructions in a `spike -l` trace read from stdin.

  spike -l --isa=<isa> [pk] prog.elf 2>&1 >/dev/null | python3 mtrace_count.py [options]

A matrix instruction is any 32-bit instruction whose low 7 bits are 0x77, so the tool
does not depend on mnemonic names and works for any version of the matrix spec.

A load that page-faults is printed twice by Spike: the faulted attempt, then the retry
after the kernel maps the page. The faulted attempt is recognised from the following
'exception ... 0x<epc>' line (epc == pc of the previous instruction) and removed, so
the counts are retired instructions.

options:
  --label NAME        title for the report
  --runs N            the program runs the model N times (ame-resnet runs forward twice);
                      adds a per-run column
  --macs-per-run M    MACs in one run (e.g. the macs= value in spike_out.txt, per forward)
  --mul REGEX         which mnemonics are the multiply-accumulate
                      (default matches mfma/mfwma/mqma/mfwopa ...)
  --seq N             show the first N matrix instructions, loop-compressed (default 120)
  --top N             also list the N most frequent NON-matrix mnemonics (default 10)
"""
import argparse
import re
import sys
from collections import Counter

INSN = re.compile(r'^core\s+\d+:\s+0x([0-9a-f]+)\s+\(0x([0-9a-f]+)\)\s+(\S+)')
EXC = re.compile(r'exception\s+\S+.*?0x([0-9a-f]+)\s*$')


def compress(toks, maxp=8):
    """Collapse repeated groups: ['a','b','a','b'] -> ['2 x (a b)']."""
    out, i = [], 0
    while i < len(toks):
        for p in range(1, maxp + 1):
            if toks[i:i + p] == toks[i + p:i + 2 * p]:
                n = 1
                while toks[i:i + p] == toks[i + n * p:i + (n + 1) * p]:
                    n += 1
                out.append(f"{n} x ({' '.join(toks[i:i + p])})")
                i += n * p
                break
        else:
            out.append(toks[i])
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='trace')
    ap.add_argument('--runs', type=int, default=1)
    ap.add_argument('--macs-per-run', type=float)
    ap.add_argument('--mul', default=r'^m[a-z]*(ma|opa)\b')
    ap.add_argument('--seq', type=int, default=120)
    ap.add_argument('--top', type=int, default=10)
    a = ap.parse_args()
    mul_re = re.compile(a.mul)

    traced, faulted = Counter(), Counter()
    other, other_faulted = Counter(), Counter()   # non-matrix instructions
    seq = []
    last = None  # (pc, mnemonic, kept_in_seq, is_matrix) of the previous instruction
    for line in sys.stdin:
        m = INSN.match(line)
        if m:
            pc, enc, name = int(m.group(1), 16), int(m.group(2), 16), m.group(3)
            if enc & 0x7f == 0x77:
                traced[name] += 1
                kept = len(seq) < a.seq
                if kept:
                    seq.append(name)
                last = (pc, name, kept, True)
            else:
                other[name] += 1
                last = (pc, name, False, False)
            continue
        e = EXC.search(line)
        if e and last and int(e.group(1), 16) == last[0]:
            if last[3]:
                faulted[last[1]] += 1
                if last[2] and seq and seq[-1] == last[1]:
                    seq.pop()
            else:
                other_faulted[last[1]] += 1
            last = None

    retired = Counter({k: traced[k] - faulted[k] for k in traced})
    other_ret = Counter({k: other[k] - other_faulted[k] for k in other})
    total = sum(retired.values())
    all_total = total + sum(other_ret.values())
    mults = sum(v for k, v in retired.items() if mul_re.search(k))

    print(f"== {a.label} ==")
    hdr = f"{'mnemonic':<16}{'traced':>12}{'fault-retry':>13}{'retired':>12}"
    if a.runs > 1:
        hdr += f"{'per run':>12}"
    print(hdr)
    for k, v in retired.most_common():
        row = f"{k:<16}{traced[k]:>12,}{faulted[k]:>13,}{v:>12,}"
        if a.runs > 1:
            row += f"{v / a.runs:>12,.0f}"
        print(row)
    print(f"{'TOTAL':<16}{sum(traced.values()):>12,}{sum(faulted.values()):>13,}{total:>12,}")
    print()
    if mults == 0:
        print(f"no multiply-accumulate matched --mul '{a.mul}'")
    else:
        print(f"multiply-accumulate instructions : {mults:,}  (regex {a.mul})")
        print(f"matrix instructions per multiply : {total / mults:.3f}")
        if a.macs_per_run:
            mpr = mults / a.runs
            print(f"MACs per multiply-accumulate     : {a.macs_per_run / mpr:.1f}  (upper bound 'tile size')")
            print(f"matrix instructions per 1000 MACs: {1000 * (total / a.runs) / a.macs_per_run:.3f}")
    print()
    print(f"ALL retired instructions         : {all_total:,}  (matrix {total:,} = {100 * total / max(all_total, 1):.1f}%)")
    if a.macs_per_run:
        print(f"ALL instructions per MAC         : {(all_total / a.runs) / a.macs_per_run:.4f}")
    print("note: the trace also contains startup code and, under pk, pk's own instructions;")
    print("      for a clean per-forward figure use the program's own counters when it has them.")
    if a.top and other_ret:
        print(f"top {a.top} non-matrix mnemonics:")
        for k, v in other_ret.most_common(a.top):
            print(f"  {k:<14}{v:>14,}  {100 * v / all_total:5.1f}%")
    print()
    print(f"first {len(seq)} matrix instructions (retired), loop-compressed:")
    for item in compress(seq):
        print("  " + item)


if __name__ == '__main__':
    main()
