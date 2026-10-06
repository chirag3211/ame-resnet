#!/usr/bin/env python3
"""Add llvm.emit_c_interface to @forward only (not to the ame_mm_* declarations)."""
import re, sys
p = sys.argv[1]
t = open(p).read()
t, n = re.subn(r"^(\s*func\.func @forward\(.*\)(?: -> [^\n{]*?)?)\s*\{$",
               r"\1 attributes {llvm.emit_c_interface} {", t, count=1, flags=re.M)
if n != 1:
    sys.exit("add_ciface: could not find a plain 'func.func @forward(...) {' line; "
             "paste the forward signature to Claude")
open(p, "w").write(t)
print("tagged @forward with llvm.emit_c_interface")
