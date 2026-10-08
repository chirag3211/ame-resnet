#!/usr/bin/env python3
# usage: pchist.py ELF HISTFILE [FUNC] [NBLOCKS]  -> top functions, and hottest 128-byte blocks in FUNC (default: top function, 12 blocks)
import sys, re, bisect, subprocess, collections
elf, hist = sys.argv[1], sys.argv[2]
syms = []
out = subprocess.run(["riscv64-unknown-elf-nm", "-n", "--defined-only", elf], capture_output=True, text=True).stdout
for l in out.splitlines():
    p = l.split()
    if len(p) == 3 and p[1] in "tTwW": syms.append((int(p[0], 16), p[2]))
addrs = [a for a, _ in syms]
raw = []
for l in open(hist):
    m = re.fullmatch(r"([0-9a-f]+) (\d+)\s*", l)
    if m: raw.append((int(m.group(1), 16), int(m.group(2))))
if not raw: sys.exit("no histogram lines found in " + hist)
lo, hi = addrs[0], addrs[-1] + 0x10000
def inrange(shift): return sum(c for a, c in raw if lo <= (a << shift) <= hi)
shift = max((0, 1, 2), key=inrange)
total = sum(c for _, c in raw)
byfn, byblk, unm = collections.Counter(), collections.defaultdict(collections.Counter), 0
for a, c in raw:
    a <<= shift
    if not (lo <= a <= hi): unm += c; continue
    i = bisect.bisect_right(addrs, a) - 1
    name = syms[i][1]
    byfn[name] += c; byblk[name][a // 128 * 128] += c
print(f"shift={shift} total={total} unmapped(pk/kernel)={unm} ({100*unm/total:.1f}%)")
for n, c in byfn.most_common(12): print(f"{c:>12}  {100*c/total:5.1f}%  {n}")
top = sys.argv[3] if len(sys.argv) > 3 else byfn.most_common(1)[0][0]
nb = int(sys.argv[4]) if len(sys.argv) > 4 else 12
if top not in byblk: sys.exit(f"function {top} not in histogram")
print("hottest 128-byte blocks in", top)
for a, c in byblk[top].most_common(nb): print(f"  {a:#x}  {c:>12}  {100*c/total:5.1f}%")
