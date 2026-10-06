#!/usr/bin/env python3
"""Decode every OP-M32 (0x77) word in an ELF with the matrix-capable spike-dasm."""
import os, re, subprocess, sys
elf = sys.argv[1]
dasm = os.path.expanduser("~/riscv-stc/bin/spike-dasm")
dump = subprocess.run(["riscv64-unknown-elf-objdump", "-d", elf], capture_output=True, text=True).stdout
for line in dump.splitlines():
    m = re.match(r"\s*[0-9a-f]+:\s+([0-9a-f]{8})\s", line)
    if m and (int(m.group(1), 16) & 0x7f) == 0x77:
        w = m.group(1)
        out = subprocess.run([dasm], input=f"DASM({w})\n", capture_output=True, text=True).stdout.strip()
        print(w, out)
