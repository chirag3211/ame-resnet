# STATE (Oct 6 2026)  -- ResNet18 on riscv-stc AME via torch-mlir -> Spike

## Environment (WSL Ubuntu, user chirag)
- Repo ~/ame-resnet. Always `source env.sh` first (activates ~/venv, THEN sets PATH).
- ~/riscv: GCC 16.1 rv64gc/lp64d, stock Spike (no matrix), pk. Do not overwrite.
- ~/riscv-stc/bin/spike: riscv-stc fork, branch matrix, commit 6add0673 (spec v0.5 in ~/work/riscv-matrix-spec; where spec and Spike differ, Spike wins).
- ~/buddy-mlir (mlir-opt, mlir-translate, llc), ~/venv (torch 2.10, torch-mlir 20261001).

## Flow
python3 python/export_resnet18.py --size N
AME:    make USE_AME=1 SPIKE=$HOME/riscv-stc/bin/spike SPIKE_ISA=rv64imafdcv_zicntr_matrix resnet-spike SIZE=N
Scalar: make SPIKE_ISA=rv64gc_zicntr resnet-spike SIZE=N
- Switching backend: rm -f build/resnet18.elf build/*.o first (Makefile does not track USE_AME).
- rdinstret traps without zicntr in the ISA string. ISA string needs a z-extension between v and matrix.
- `make clean` keeps resnet18.mlir, input.bin, golden_logits.bin, meta.json.
- main_resnet.c runs forward twice: RUN0 = cold, RUN1 = warm. Report RUN1.

## Design
linalg.conv -> img2col + contract -> fold unit dims -> 21 rank-2 linalg.matmul -> bufferize -> matmul_to_call.py -> external ame_mm_f32_N
-> LLVM -> RV64 object; gen_shims.py shims -> ame_matmul_f32 (src/ame_dispatch.c) -> ame_hw.c (real matrix insns) or ref_matmul.c.
AME fp32: tile m=8,k=4,n=4 granted; C in acc0 across K loop (zeroed by stride-0 load); loops advance by granted size.
Wrappers: include/ame_insn.h (.insn r, opcode 0x77). int8 still uses the scalar reference.

## Verified results (retired instructions on Spike; NOT cycles, no hardware speedup claim)
Each matrix instruction retires as exactly 1 instruction (tests/insn_cost.c).
First-touch cost (RUN0-RUN1, ~65M at 64, ~122M at 224) is identical in both backends; in the histogram it appears in pk/kernel code.

| warm RUN1          | 64 scalar | 64 AME  | 224 scalar | 224 AME  |
|--------------------|-----------|---------|------------|----------|
| forward            | 1231.94M  | 134.11M | 14138.0M   | 1607.2M  |
| in_matmul          | 1118.15M  | 20.31M  | 12776.8M   | 245.96M  |
| in_copy            | 18.48M    | 18.48M  | 224.5M     | 224.5M   |
| everything else    | 95.3M     | 95.3M   | 1136.7M    | 1136.7M  |
Whole-forward ratio 9.19x (64), 8.80x (224). Matmul-only ratio 55.0x (64), 51.9x (224).
Correctness: TOP1 238 and max abs err 3.8e-6 at 224; TOP1 381 at 64. K-step = 17 instr (incl. 3 counter instr), ~126 of 128 MACs.
AME forward at 224: matmul 15%, memrefCopy 14%, generated loops 71%.

## Do not use
Earlier figures 6.6x, 16.2x, 0.489 instr/MAC: cold-start inflated.

## Open items
1. Identify the hot generated loops (top blocks at 64x64: 0x10f00 0x11480 0x10700 0x11a80 0x12080 0x17600; ~54% of forward in 12 blocks). tools/pchist.py gives the histogram.
2. memrefCopy: ~14% of forward; half is memmove. Row-wise fast path is a small win.
3. Fusing elementwise ops before bufferization: untested on this LLVM.
4. Matmul tuning (hoist msettilek, multi-accumulator): at most ~3% of forward now, low priority.
5. int8 (mqma.b.mm) needs a quantized model; separate project.
6. Target is the riscv-stc v0.5 matrix extension, not the task-group Zvame draft.
