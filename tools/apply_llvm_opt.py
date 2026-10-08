#!/usr/bin/env python3
"""Idempotently add an optional LLVM_OPT=O2|O3 step (opt before llc) to the Makefile resnet18.o rule.
usage: python3 apply_llvm_opt.py [Makefile]"""
import re, sys
p = sys.argv[1] if len(sys.argv) > 1 else "Makefile"
s = open(p).read()
if "LLVM_OPT_BIN" in s:
    sys.exit("already applied")
pat = re.compile(r"^\$\(B\)/resnet18\.o: \$\(B\)/resnet18\.ll\n(?:\t.*\n?)+", re.M)
m = pat.search(s)
if not m:
    sys.exit("could not find the $(B)/resnet18.o rule; paste: grep -n -A4 'resnet18.o:' Makefile")
new = """# LLVM_OPT=O2|O3 runs LLVM's IR optimizer (opt) before llc; default: none.
# Not tracked by make: rm -f $(B)/*.o when changing it.
LLVM_OPT     ?=
LLVM_OPT_BIN ?= opt
$(B)/resnet18.o: $(B)/resnet18.ll
\t@if [ -n "$(LLVM_OPT)" ]; then \\
\t  $(LLVM_OPT_BIN) -passes='default<$(LLVM_OPT)>' -mtriple=riscv64-unknown-elf -mattr=+m,+a,+f,+d,+c $< -S -o $(B)/resnet18_opt.ll && \\
\t  cp $(B)/resnet18_opt.ll $(B)/resnet18_cg.ll; \\
\t else cp $< $(B)/resnet18_cg.ll; fi
\t$(LLC) $(B)/resnet18_cg.ll -mtriple=riscv64-unknown-elf -mattr=+m,+a,+f,+d,+c \\
\t  -target-abi=$(ABI) -filetype=obj -o $@
"""
open(p, "w").write(s[:m.start()] + new + s[m.end():])
print("patched", p)
