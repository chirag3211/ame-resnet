#!/usr/bin/env bash
# ResNet18 at 64 and 224, scalar and AME, warm RUN1, one line each.  usage: tools/bench.sh [O2|O3|none]  (default O3)
# Needs: source env.sh. Exports go to build_r18_<size>/ so build/ is untouched. ~10 min (scalar 224 dominates).
O=${1:-O3}; [ "$O" = none ] && O=""
SP=${SPIKE:-$HOME/riscv-stc/bin/spike}
for S in 64 224; do
  B=build_r18_$S
  [ -f $B/resnet18.mlir ] || python3 python/export_resnet18.py --size $S --out $B >/dev/null 2>&1
  for BE in scalar ame; do
    rm -f $B/resnet18.elf $B/*.o
    if [ $BE = ame ]; then A="USE_AME=1 SPIKE=$SP SPIKE_ISA=rv64imafdcv_zicntr_matrix"; else A="SPIKE_ISA=rv64gc_zicntr"; fi
    out=$(make B=$B LLVM_OPT=$O $A resnet-spike SIZE=$S 2>&1 | grep -v LOGIT)
    echo "$out" | grep RUN1 | awk -v s=$S -v b=$BE '{split($3,f,"=");split($4,m,"=");split($5,c,"=");printf "%s %s forward=%.2fM matmul=%.2fM copy=%.2fM else=%.2fM | ",s,b,f[2]/1e6,m[2]/1e6,c[2]/1e6,(f[2]-m[2]-c[2])/1e6}'
    echo "$out" | grep -E '^TOP1|max abs|rror' | tr '\n' ' '; echo
  done
done
