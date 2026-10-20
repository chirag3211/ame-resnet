#!/usr/bin/env bash
# ResNet-50 end to end: export -> (AME + scalar) build + Spike run -> results.jsonl -> table.
#
#   usage:  tools/run_resnet50.sh [SIZE ...]        default sizes: 64 224
#   env:    SKIP_SCALAR=1   run AME only
#           SPIKE_MEM=4096  Spike memory in MB (record.sh default)
#           SPIKE=...       matrix Spike (default ~/riscv-stc/bin/spike); used for AME runs only
#           FORCE=1         re-run and re-record a dir/backend that already has a row in results.jsonl
#
# Directories follow the existing convention (build_r18_64, build_r18_224): build_r50_<SIZE>.
# Each run appends one JSON line per backend to results.jsonl via tools/record.sh.
# Safe to re-run: the export is skipped if the directory already has the MLIR + golden files,
# and the Makefile only redoes the steps whose flags/inputs changed.

cd "$(dirname "$0")/.." || exit 1          # run from the repo root (pk file I/O and relative paths need it)
source env.sh                              # before `set -u`: the venv activate script is not nounset-clean
set -euo pipefail

SIZES=("$@"); [ ${#SIZES[@]} -gt 0 ] || SIZES=(64 224)
SPIKE=${SPIKE:-$HOME/riscv-stc/bin/spike}
SPIKE_MEM=${SPIKE_MEM:-4096}
RISCV=${RISCV:-$HOME/riscv}
# NOTE: SPIKE is deliberately NOT exported. The Makefile reads $(SPIKE) from the environment, so exporting it
# made the scalar run use the matrix Spike with a plain rv64gc ISA (it crashed with SIGFPE). It is passed per call below.

# ---------------------------------------------------------------- 0. environment check
miss=0
need() { command -v "$1" >/dev/null 2>&1 || { echo "MISSING tool: $1"; miss=1; }; }
for t in riscv64-unknown-elf-gcc riscv64-unknown-elf-objcopy riscv64-unknown-elf-nm mlir-opt mlir-translate llc opt python3; do need "$t"; done
[ -x "$SPIKE" ] || { echo "MISSING matrix Spike: $SPIKE"; miss=1; }
if ! command -v spike >/dev/null 2>&1; then echo "MISSING stock spike on PATH (scalar baseline uses it; expected in $RISCV/bin)"; miss=1
elif [ "$(command -v spike)" -ef "$SPIKE" ]; then echo "PATH 'spike' is the matrix Spike ($SPIKE); the scalar baseline needs the stock one in $RISCV/bin (check env.sh / PATH order)"; miss=1; fi
[ -f "$RISCV/riscv64-unknown-elf/bin/pk" ] || { echo "MISSING pk: $RISCV/riscv64-unknown-elf/bin/pk"; miss=1; }
python3 -c "import torch, torchvision, torch_mlir, numpy" 2>/dev/null || { echo "MISSING python modules (torch, torchvision, torch_mlir, numpy): is ~/venv active?"; miss=1; }
[ $miss -eq 0 ] || { echo "environment incomplete, aborting"; exit 1; }

# ---------------------------------------------------------------- 1. exporter (derived from export_resnet18.py)
if [ ! -f python/export_resnet50.py ]; then
  sed 's/resnet18/resnet50/g; s/ResNet18/ResNet50/g' python/export_resnet18.py > python/export_resnet50.py
  echo "created python/export_resnet50.py (from export_resnet18.py)"
fi
python3 -m py_compile python/export_resnet50.py

# results.jsonl rows record git commit + dirty flag, and results_table.py warns on a dirty tree
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
  echo "NOTE: git tree is dirty (new exporter/script?). Commit first if you want clean provenance in results.jsonl:"
  echo "      git add python/export_resnet50.py tools/run_resnet50.sh && git commit -m 'ResNet-50 exporter + run script'"
fi

# already_recorded DIR BACKEND: true if results.jsonl already has a row for it (avoids duplicate rows on re-runs)
already_recorded() {
  [ "${FORCE:-0}" = 1 ] && return 1
  [ -f results.jsonl ] || return 1
  python3 - "$1" "$2" <<'PY'
import json, sys
d, b = sys.argv[1:3]
for l in open("results.jsonl"):
    try: r = json.loads(l)
    except ValueError: continue
    if r.get("dir") == d and r.get("backend") == b: sys.exit(0)
sys.exit(1)
PY
}

# ---------------------------------------------------------------- 2. per size: export, AME run, scalar run
for S in "${SIZES[@]}"; do
  D=build_r50_$S
  echo; echo "================ ResNet-50, input ${S}x${S}  ->  $D ================"

  if [ ! -s "$D/resnet50.mlir" ] || [ ! -s "$D/input.bin" ] || [ ! -s "$D/golden_logits.bin" ]; then
    python3 python/export_resnet50.py --size "$S" --out "$D"
  else
    echo "export: $D already has resnet50.mlir / input.bin / golden_logits.bin, skipping"
  fi
  ln -sfn resnet50.mlir "$D/resnet18.mlir"           # the Makefile still names everything resnet18.*

  if already_recorded "$D" ame; then
    echo "ame: $D already in results.jsonl, skipping (FORCE=1 to redo)"
  else
    SPIKE="$SPIKE" tools/record.sh "$D" ame SIZE="$S" SPIKE_MEM="$SPIKE_MEM"
    grep -q 'PASS' "$D/log_ame.txt" || echo "WARNING: no PASS in $D/log_ame.txt (see the log; the row was still recorded)"
  fi

  if [ "${SKIP_SCALAR:-0}" != 1 ]; then
    if already_recorded "$D" scalar; then
      echo "scalar: $D already in results.jsonl, skipping (FORCE=1 to redo)"
    else
      env -u SPIKE tools/record.sh "$D" scalar SIZE="$S" SPIKE_MEM="$SPIKE_MEM"   # stock spike from PATH
      grep -q 'PASS' "$D/log_scalar.txt" || echo "WARNING: no PASS in $D/log_scalar.txt (see the log; the row was still recorded)"
    fi
  fi

  # MAC sanity check against the 4.0892 GMAC reference quoted in STATE.md (valid at 224 only)
  if [ "$S" = 224 ]; then
    python3 - "$D/log_ame.txt" <<'PY'
import re, sys
m = re.search(r"STATS backend=\S+ matmul_calls=(\d+) macs=(\d+)", open(sys.argv[1]).read())
if m:
    macs = int(m[2]); ref = 4.0892e9
    print(f"MAC check: macs={macs:,} vs reference {ref:,.0f} ({(macs-ref)/ref*100:+.2f}%), matmul_calls={m[1]}")
    if abs(macs-ref)/ref > 0.01: print("WARNING: more than 1% off the reference -> check conv_rewrite on the bottleneck / stride-2 1x1 convs")
PY
  fi
done

# ---------------------------------------------------------------- 3. summary
echo; echo "================ results (r50 rows + header) ================"
python3 tools/results_table.py | grep -E '^\| model|^\|---|r50_|^WARNING.*r50|^\\\*' || true
echo; echo "rows appended to results.jsonl; logs: build_r50_<SIZE>/log_{ame,scalar}.txt"
