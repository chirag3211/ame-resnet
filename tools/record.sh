#!/usr/bin/env bash
# Build + run one model on one backend and append the result to results.jsonl.
#   usage: tools/record.sh DIR ame|scalar [KEY=VALUE ...]        (run `source env.sh` first)
#   KEY=VALUE are passed to make and recorded as flags, e.g. LLVM_OPT=O2 SPIKE_MEM=8192 SIZE=224 AME_COUNT_STEPS=1
#   Defaults: LLVM_OPT=O3, SPIKE_MEM=4096; ame -> USE_AME=1 + matrix Spike/ISA, scalar -> stock spike, rv64gc_zicntr.
#   Driver: DIR/main_gen.c + python/compare_tensor.py if DIR/main_gen.c exists, else src/main_resnet.c + python/compare.py.
#   Log: DIR/log_<backend>.txt (LOGIT lines dropped). The Makefile stamps flags, so no `rm -f DIR/*.o` is needed.
#   Re-ingest an old log without running: python3 tools/record.py LOG --dir DIR --backend ame --set llvm_opt=O3
set -uo pipefail
[ $# -ge 2 ] || { sed -n 2,9p "$0" | sed 's/^# \{0,1\}//'; exit 2; }
D=${1%/}; BE=$2; shift 2
[ -d "$D" ] || { echo "no such dir: $D" >&2; exit 2; }
case $BE in ame|scalar) ;; *) echo "backend must be ame or scalar" >&2; exit 2;; esac
declare -A V=([LLVM_OPT]=O3 [SPIKE_MEM]=4096 [SIZE]=224)
if [ -f "$D/main_gen.c" ]; then V[DRIVER]=$D/main_gen.c; V[COMPARE]=python/compare_tensor.py; fi
if [ $BE = ame ]; then
  V[USE_AME]=1; V[SPIKE]=${SPIKE:-$HOME/riscv-stc/bin/spike}; V[SPIKE_ISA]=rv64imafdcv_zicntr_matrix
else
  V[SPIKE_ISA]=rv64gc_zicntr
fi
for kv in "$@"; do V[${kv%%=*}]=${kv#*=}; done
ARGS=(); SETS=()
for k in "${!V[@]}"; do ARGS+=("$k=${V[$k]}"); SETS+=(--set "$(echo "$k" | tr A-Z a-z)=${V[$k]}"); done
LOG=$D/log_$BE.txt
echo "make B=$D ${ARGS[*]} resnet-spike  ->  $LOG"
T0=$(date +%s)
make B="$D" "${ARGS[@]}" resnet-spike 2>&1 | grep -v LOGIT | tee "$LOG" | grep -E 'rror|RUN|STATS|max abs|PASS|FAIL'
RC=${PIPESTATUS[0]}
python3 tools/record.py "$LOG" --dir "$D" --backend "$BE" "${SETS[@]}" --wall $(( $(date +%s) - T0 )) --rc "$RC"
