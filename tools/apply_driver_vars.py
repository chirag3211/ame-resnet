#!/usr/bin/env python3
"""Idempotently let the Makefile choose the driver and the compare script: make DRIVER=<dir>/main_gen.c COMPARE=python/compare_tensor.py ..."""
import sys
p = sys.argv[1] if len(sys.argv) > 1 else "Makefile"
s = open(p).read()
if "DRIVER" in s:
    sys.exit("already applied")
a = "$(B)/resnet18.elf: src/main_resnet.c src/memref_copy.c"
b = "\tpython3 python/compare.py $(B)/spike_out.txt $(B)/golden_logits.bin"
assert a in s and b in s, "unexpected Makefile layout"
s = s.replace(a, "DRIVER  ?= src/main_resnet.c\nCOMPARE ?= python/compare.py\n$(B)/resnet18.elf: $(DRIVER) src/memref_copy.c")
s = s.replace(b, "\tpython3 $(COMPARE) $(B)/spike_out.txt $(B)/golden_logits.bin")
open(p, "w").write(s); print("patched", p)
