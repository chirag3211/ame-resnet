# STATE (Oct 7 2026)  -- PyTorch -> torch-mlir -> RV64 -> riscv-stc Spike (AME matrix extension)
Models so far: ResNet18 (224 and 64), ViT-B/16 (2-layer and full 12-layer). Label convention below:
**measured** = printed by the program, **derived** = arithmetic on measured numbers, **hypothesis** = not verified.

## Environment (WSL Ubuntu, user chirag; WSL RAM budget 15 GB, can go to 20 GB)
- Repo ~/ame-resnet. Always `source env.sh` first (activates ~/venv, THEN sets PATH).
- ~/riscv: GCC 16.1 rv64gc/lp64d, stock Spike (no matrix), pk. Do not overwrite.
- ~/riscv-stc/bin/spike: riscv-stc fork, branch matrix, commit 6add0673 (spec v0.5 in ~/work/riscv-matrix-spec; where spec and Spike differ, Spike wins).
- ~/buddy-mlir (mlir-opt, mlir-translate, llc), ~/venv (torch 2.10, torch-mlir 20261001).

## Flow
ResNet18: `python3 python/export_resnet18.py --size N`
ViT:      `python3 python/export_vit.py L` (L layers, writes build_vitL/vit.mlir + input.bin + golden_logits.bin; head is randomized, otherwise torchvision zero-inits it and all logits are 0)
          then `ln -sf vit.mlir build_vitL/resnet18.mlir` (the Makefile still uses the name resnet18.*)
AME:    make [B=<dir>] USE_AME=1 SPIKE=$HOME/riscv-stc/bin/spike SPIKE_ISA=rv64imafdcv_zicntr_matrix resnet-spike SIZE=N
Scalar: make [B=<dir>] SPIKE_ISA=rv64gc_zicntr resnet-spike SIZE=N
- B=<dir> keeps models apart (lower_resnet.sh writes its intermediates next to the output file). SIZE must match the exported input (224 for ViT). A SIZE/model mismatch reads past the input buffer and FAILs (seen once).
- Switching backend: rm -f <B>/resnet18.elf <B>/*.o first (Makefile does not track USE_AME, nor mlir/conv_rewrite.py).
- rdinstret traps without zicntr in the ISA string. ISA string needs a z-extension between v and matrix.
- `make clean` keeps resnet18.mlir, input.bin, golden_logits.bin, meta.json.
- main_resnet.c runs forward twice: RUN0 = cold, RUN1 = warm. Report RUN1. It is reused for every model with signature (1x3x224x224 f32) -> (1x1000 f32).
- input.o: objcopy runs inside $(B), symbol is _binary_input_bin_start (path-independent).

## Design
Per model: torch-mlir linalg-on-tensors -> mlir/conv_rewrite.py (conv_2d_nchw_fchw -> 5D gather generic + collapse + linalg.matmul + expand; no div/mod; N=1, f32, dilation 1, no groups) -> fold unit dims
-> rank-2 linalg.matmul and rank-3 linalg.batch_matmul -> bufferize -> matmul_to_call.py -> external ame_mm_f32_N / ame_bmm_f32_N
-> LLVM (math.erf via --convert-math-to-libm -> erff) -> RV64 object; gen_shims.py shims -> ame_matmul_f32 (src/ame_dispatch.c) -> ame_hw.c (real matrix insns) or ref_matmul.c.
ame_bmm_f32 (src/memref_shims.c) loops over the batch and calls ame_matmul_f32 once per batch element (so STATS counts one call per head).
AME fp32: tile m=8,k=4,n=4 granted; C in acc0 across K loop (zeroed by stride-0 load); loops advance by granted size.
Wrappers: include/ame_insn.h (.insn r, opcode 0x77). int8 still uses the scalar reference.

## Verified results (retired instructions on Spike; NOT cycles, no hardware speedup claim)
Each matrix instruction retires as exactly 1 instruction (tests/insn_cost.c). First-touch cost (RUN0-RUN1) is identical in both backends: ~65M at ResNet 64, ~122M at ResNet 224.
"everything else" = forward - in_matmul - in_copy (derived).

### ResNet18, warm RUN1 (after conv_rewrite.py replaced the built-in img2col)
| | 64 scalar | 64 AME | 224 scalar | 224 AME |
|---|---|---|---|---|
| forward | 1169.25M | 71.41M | 13324.80M | 794.00M |
| in_matmul | 1118.15M | 20.31M | 12776.75M | 245.96M |
| in_copy (memrefCopy) | 18.48M | 18.48M | 224.53M | 224.53M |
| everything else | 32.62M | 32.62M | 323.52M | 323.52M |
Whole-forward ratio (derived) 16.37x (64), 16.78x (224). Matmul-only ratio 55.0x (64), 51.9x (224).
Correctness: 64: TOP1 381, max abs err 2.98e-6. 224: TOP1 238, max abs err 3.8e-6 (both backends PASS).
Scalar 224 run took 6m24s wall (~70M instr/s on Spike).
AME 224 forward shares (derived): matmul 31%, memrefCopy 28%, everything else 41%.
Before the rewrite (built-in img2col, same method): forward 134.11M (64) / 1607.2M (224), everything else 95.3M / 1136.7M, ratio 9.19x / 8.80x.
The rewrite removed 62.7M (64) / 813.2M (224) instructions in both backends. Cause (hypothesis): the built-in img2col's floordiv/mod index math (~40 integer instr per gathered element after --lower-affine); not isolated with a histogram.

### ViT-B/16, AME, warm RUN1, seeded random weights (head randomized), 224x224
| | 2-layer | full 12-layer |
|---|---|---|
| forward | 858.33M | 4953.64M |
| in_matmul | 410.35M | 2380.90M |
| in_copy | 0 | 0 |
| everything else | 447.98M | 2572.74M |
| macs | 3.024G | 17.564G |
| matmul_calls (STATS) | 58 | 338 |
| TOP1 / max abs err | 829 / 3.5e-6 PASS | 272 / 4.6e-6 PASS |
Matmul efficiency 0.1356 instr/MAC (same as ResNet18). Non-matmul share 52% (derived).
338 calls = 50 matmuls (48 Linear + patch conv + head) + 24 batch_matmul ops x 12 heads (derived; matches).
Full-model run: 661 MB vit.mlir (~7.6 bytes/param), `make` wall time 44m06s including export-independent lowering and both Spike runs, max RSS 6.15 GB (of the largest process in the tree; which one is not identified).
No scalar ViT run yet. Hypothesis from ResNet's scalar cost (7.0-7.5 instr/MAC): scalar full ViT ~125-135B instructions, ~25-27x whole-forward ratio. Not measured.
No tensor.pad in ViT, so memrefCopy is 0 (consistent with the 18 ResNet copies being the pads).

## Do not use
Earlier figures 6.6x, 16.2x, 0.489 instr/MAC for ResNet18 AME (cold-start inflated). The warm 16.37x/16.78x above is a different, valid measurement.
9.19x / 8.80x are valid but pre-rewrite.

## Open items
1. Profile the ViT non-matmul 52% (candidates: softmax, LayerNorm, GELU/erff, residual adds, activation transposes/permutes, weight transposes). Command in "Profiling" below.
2. Weight transposes: in the 2-layer ViT IR, 9 of the 26 linalg.transpose ops take a constant (the Linear weights, 8 + head) and are not folded, so they likely run as scalar copies every forward. ~85.7M floats for 12 layers (derived). Cost not measured. Fix options: pre-transpose in the exporter (nn.MultiheadAttention needs a custom attention module) or constant-fold in MLIR.
3. ResNet memrefCopy: 28% of AME forward at 224 (224.5M). memrefCopy_impl does one memmove per element; row-wise fast path should cut most of it (hypothesis).
4. Fusing elementwise ops before bufferization: untested on this LLVM.
5. Matmul tuning (hoist msettilek, multi-accumulator): matmul is 31-48% of forward; K-step = 17 instr (incl. 3 counter instr), ~126 of 128 MACs.
6. Decode steps are matrix-vector (M=1): an 8-row tile would use ~1/8 of the unit (hypothesis); may need a different mapping.
7. int8 (mqma.b.mm) needs a quantized model; needed to fit the large LLMs in RAM.
8. conv_rewrite.py gaps: groups/depthwise, conv1d, transposed conv, dilation, N>1 (needed for Whisper/Moonshine/Canary/Kokoro/Mamba).
9. Target is the riscv-stc v0.5 matrix extension, not the task-group Zvame draft.

## Profiling (Spike PC histogram; command from memory, check `spike --help 2>&1 | grep -i hist`)
spike -g prints "PC Histogram" lines (pc count) to stderr at exit (both RUN0 and RUN1 and pk code are included):
  $HOME/riscv-stc/bin/spike -g --isa=rv64imafdcv_zicntr_matrix -m2048 $HOME/riscv/riscv64-unknown-elf/bin/pk build_vit2/resnet18.elf 2> build_vit2/hist.txt > /dev/null
  python3 tools/pchist.py build_vit2/resnet18.elf build_vit2/hist.txt
Generated code is one function, so look at the hottest 128-byte blocks and `riscv64-unknown-elf-objdump -d`; libm callees (erff, expf) show up by name in the function list.

## Workload plan (assignment: ResNet-50, ViT-B/16, Whisper small / large-v3-turbo, Moonshine tiny/base, Canary-Qwen 2.5B, Kokoro 82M, Phi-4-mini, Phi-4, Ministral 3 3B/8B, Gemma 3n E2B/E4B)
- ResNet18 is the baseline; ResNet-50 skipped (coverage argument). ViT-B/16 done at full size.
- fp32 fits 15 GB: Moonshine, Kokoro, Whisper small and large-v3-turbo; Canary-Qwen borderline at 20 GB. The 3B+ LLMs do not fit in fp32; use reduced-layer random-weight configs (2 layers, same widths) and label them as such; extrapolating by layer count is a hypothesis.
- Mamba (e.g. mamba-130m) planned for coverage: projections are matmuls, the selective scan and depthwise causal conv are not covered yet.
- Next: profile ViT, then Whisper/Moonshine (conv1d, cross-attention), then reduced-config LLMs (prefill 128/512/2048, decode 512/2048).