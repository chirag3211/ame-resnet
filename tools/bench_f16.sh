#!/bin/sh
# Build and run the one-matmul benchmark for three kernels on Spike under pk, and print for each:
#   - the program's own result line (PASS/FAIL against a double-precision reference, warm-run instret)
#   - the matrix-instruction table from the trace (tools/mtrace_count.py, same tool used for riscv-dnn)
#
#   kernel 0: fp32 in/out                      (src/ame_hw.c, your existing kernel)
#   kernel 1: fp16 in, fp32 acc, fp16 out; compiler-style type switching  (src/ame_hw_f16.c, faithful)
#   kernel 2: same arithmetic with redundant mtype writes removed          (src/ame_hw_f16.c, hoisted)
#
# usage: tools/bench_f16.sh [M N K [PAD]]      defaults: 64 64 64 0
# env overrides: RISCV, CROSS, SPIKE, PK, ISA, HFLAGS (e.g. HFLAGS="-DH_KLOOP=1" for the hoisted kernel)
set -e
M=${1:-64}; N=${2:-64}; K=${3:-64}; PAD=${4:-0}
RISCV=${RISCV:-$HOME/riscv}
CROSS=${CROSS:-riscv64-unknown-elf-}
SPIKE=${SPIKE:-$HOME/riscv-stc/bin/spike}
PK=${PK:-$RISCV/riscv64-unknown-elf/bin/pk}
ISA=${ISA:-rv64imafdcv_zfh_zicntr_matrix}
B=build/bench_f16
mkdir -p "$B"
for k in 0 1 2; do
  ${CROSS}gcc -march=rv64gc -mabi=lp64d -O2 -Wall -Wextra -Iinclude -DUSE_AME $HFLAGS \
    -DBENCH_KERNEL=$k -DBENCH_M=$M -DBENCH_N=$N -DBENCH_K=$K -DBENCH_PAD=$PAD \
    tests/bench_matmul.c src/ame_hw_f16.c src/ame_hw.c src/ref_matmul.c -o "$B/bench_k$k.elf"
  echo "================ kernel $k  (M=$M N=$N K=$K pad=$PAD)"
  "$SPIKE" -l --isa="$ISA" -m2048 "$PK" "$B/bench_k$k.elf" 2>&1 >"$B/out_k$k.txt" \
    | python3 tools/mtrace_count.py --label "kernel $k  ${M}x${N}x${K}" --runs 2 --macs-per-run $((M*N*K)) --top 0 --seq 40
  grep BENCH "$B/out_k$k.txt" || { echo "no BENCH line: program output was:"; cat "$B/out_k$k.txt"; }
done
