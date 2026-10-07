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
ViT:      `python3 python/export_vit.py L [OUTDIR] [--orig]` -> build_vitLp/ (default: pre-transposed weights + custom attention, checked against torchvision; max diff 1.7e-6) or build_vitL/ with --orig (plain torchvision; the 858.33M / 4953.64M numbers). Head is randomized (torchvision zero-inits it -> all logits 0).
          then `ln -sf vit.mlir <OUTDIR>/resnet18.mlir` (the Makefile still uses the name resnet18.*)
Generic model (any single f32 input/output): exporter writes <dir>/{model.mlir, resnet18.mlir(symlink), input.bin, golden_logits.bin, model.json, main_gen.c}; build with
  make B=<dir> DRIVER=<dir>/main_gen.c COMPARE=python/compare_tensor.py LLVM_OPT=O3 USE_AME=1 SPIKE=... SPIKE_ISA=... resnet-spike
  main_gen.c runs forward RUNS times (default 2; `python3 python/gen_driver.py <dir>/model.json <dir>/main_gen.c 1` for one cold run). compare_tensor.py checks 2048 sampled outputs + sum of squares + a position-weighted checksum (catches single-element errors >= ~0.5; not exact).
AME:    make [B=<dir>] USE_AME=1 SPIKE=$HOME/riscv-stc/bin/spike SPIKE_ISA=rv64imafdcv_zicntr_matrix resnet-spike SIZE=N
Scalar: make [B=<dir>] SPIKE_ISA=rv64gc_zicntr resnet-spike SIZE=N
- B=<dir> keeps models apart (lower_resnet.sh writes its intermediates next to the output file). SIZE must match the exported input (224 for ViT). A SIZE/model mismatch reads past the input buffer and FAILs (seen once).
- Switching backend: rm -f <B>/resnet18.elf <B>/*.o first (Makefile does not track USE_AME, nor mlir/conv_rewrite.py).
- Optional switches: `LLVM_OPT=O3` (not tracked by make: rm -f <B>/*.o when changing), `AME_COUNT_STEPS=1` (re-enables ksteps/otiles, costs 3 instr/K-step).
- rdinstret traps without zicntr in the ISA string. ISA string needs a z-extension between v and matrix.
- `make clean` keeps resnet18.mlir, input.bin, golden_logits.bin, meta.json.
- main_resnet.c runs forward twice: RUN0 = cold, RUN1 = warm. Report RUN1. It is reused for every model with signature (1x3x224x224 f32) -> (1x1000 f32).
- input.o: objcopy runs inside $(B), symbol is _binary_input_bin_start (path-independent).

## Design
Per model: torch-mlir linalg-on-tensors -> mlir/conv_rewrite.py (conv_2d_nchw_fchw and conv_1d_ncw_fcw -> gather generic + collapse + linalg.matmul + expand; no div/mod; N=1, f32, dilation 1, no groups) -> fold unit dims
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
### ViT-B/16 2-layer, pre-transposed weights (default exporter), AME, warm RUN1 (measured)
forward 699.64M (was 858.33M, -18.5%); in_matmul 410.35M (unchanged); everything else 289.29M (derived, was 447.98M); matmul share 59%. TOP1 829, max abs err 3.6e-6, PASS; 58 calls, 3.024G MACs unchanged.
Constant transposes in the IR 9 -> 0; total linalg.transpose 26 -> 9. Saving exceeds the weight transposes alone (see Profile findings); the rest is attributed (hypothesis) to the simpler attention graph. Full 12-layer pre-transposed run not done (hypothesis: ~4.0G forward if savings scale per layer).

Matmul efficiency 0.1356 instr/MAC (same as ResNet18). Non-matmul share 52% (derived).
338 calls = 50 matmuls (48 Linear + patch conv + head) + 24 batch_matmul ops x 12 heads (derived; matches).
Full-model run: 661 MB vit.mlir (~7.6 bytes/param), `make` wall time 44m06s including export-independent lowering and both Spike runs, max RSS 6.15 GB (of the largest process in the tree; which one is not identified).
No scalar ViT run yet. Hypothesis from ResNet's scalar cost (7.0-7.5 instr/MAC): scalar full ViT ~125-135B instructions, ~25-27x whole-forward ratio. Not measured.
No tensor.pad in ViT, so memrefCopy is 0 (consistent with the 18 ResNet copies being the pads).

### ViT-B/16 2-layer pre-transposed: matmul counters off, then LLVM O3 (AME, warm RUN1, measured)
| | counters on | counters off | + LLVM_OPT=O3 |
|---|---|---|---|
| forward | 699.64M | 627.16M | **562.19M** |
| in_matmul | 410.35M | 337.87M | 337.87M |
| everything else (derived) | 289.29M | 289.29M | **224.33M** |
TOP1 829, max abs err 3.636e-6 (identical bits of error across the last two columns), PASS in all three. RUN0-RUN1 gap 109.5M.
- Counters off: -72.48M = 3.02 instr per K-step x 24.03M K-steps; matmul cost 0.1117 instr/MAC (was 0.1357). ksteps/otiles print 0 unless built with AME_COUNT_STEPS=1.
- LLVM O3 (opt before llc; the pipeline previously ran no LLVM IR optimization): -64.97M (-22.5% of non-matmul, -10.4% of forward). Matmul unchanged (it is C compiled by gcc).
- Cumulative on the 2-layer ViT AME forward: 858.33M (original export, counters on, no opt) -> 562.19M (-34.5%). Matmul share is now 60%.
- Not yet profiled after O3. Hypothesis: erff+expf (~87M/run, if unchanged) are now ~39% of the remaining 224M non-matmul.
- ResNet18 table above was measured with counters ON and no opt. Re-baseline pending: `tools/bench.sh` (derived prediction at 224 AME: in_matmul ~202.8M from removing 3 x 14.4M counter instr).

### ResNet18 CURRENT BASELINE: counters off + LLVM_OPT=O3 + run-based memrefCopy (warm RUN1, measured via tools/bench.sh; ratios derived)
| | 64 scalar | 64 AME | 224 scalar | 224 AME |
|---|---|---|---|---|
| forward | 1133.41M | 31.99M | 12948.00M | 373.75M |
| matmul | 1118.14M | 16.73M | 12776.75M | 202.51M |
| memrefCopy | 4.13M | 4.13M | 39.31M | 39.31M |
| everything else | 11.14M | 11.14M | 131.94M | 131.94M |
Whole-forward ratio 35.43x (64), 34.64x (224); matmul-only 66.83x, 63.09x. TOP1 381 / 238, max abs err 2.98e-6 / 3.8e-6, PASS. (Retired instructions, not cycles.)
AME 224 shares (derived): matmul 54.2%, everything else 35.3%, memrefCopy 10.5%. AME 224 forward vs original 1607.2M: 4.30x lower.
memrefCopy 224.53M -> 39.31M (-82.5%): ~123 -> ~21 instr per element (derived, ~1.83M floats in the 18 pads).
Progress of the 224 whole-forward ratio (all warm RUN1): 8.80x (built-in img2col) -> 16.78x (conv_rewrite.py) -> 23.49x (counters off + O3) -> 34.64x (run-based memrefCopy).

### (intermediate step) ResNet18 re-baseline: counters off + LLVM_OPT=O3 (warm RUN1, measured via tools/bench.sh; ratios derived)
| | 64 scalar | 64 AME | 224 scalar | 224 AME |
|---|---|---|---|---|
| forward | 1147.76M | 46.34M | 13133.22M | 558.98M |
| matmul | 1118.14M | 16.73M | 12776.75M | 202.51M |
| memrefCopy | 18.48M | 18.48M | 224.53M | 224.53M |
| everything else | 11.14M | 11.14M | 131.94M | 131.94M |
Whole-forward ratio 24.77x (64), 23.49x (224); matmul-only 66.83x (64), 63.09x (224). TOP1 381 / 238, max abs err 2.98e-6 / 3.8e-6, PASS. (Retired instructions, not cycles.)
AME 224 shares (derived): memrefCopy 40.2%, matmul 36.2%, everything else 23.6%. This table supersedes the earlier ResNet table, which had counters on and no opt.
AME 224 forward vs the original built-in-img2col run: 1607.2M -> 558.98M (2.88x lower).
memrefCopy is compiled by gcc, so LLVM O3 does not touch it; ~123 instr per element (derived: 224.5M / ~1.83M floats in the 18 pads).

## Profile after O3 + counters off + run-based copy (histogram halved for per-run; all derived from measured counts)
ResNet18 224 AME: matmul 202.51M (= in_matmul), forward loops ~129.5M, memcpy ~37.4M (the copy cost), memset ~2.5M. Top 8 loop blocks = ~39.6M/run (30.6% of loops); largest block ~7.3M/run (<2% of forward).
ViT 2-layer AME: matmul 337.87M, forward loops ~134.1M, erff ~50.6M, expf ~36.8M, memset+memcpy ~2.7M (sum of non-matmul 224.3M = measured). erff+expf = 39% of non-matmul, 15.6% of forward. Top 8 loop blocks ~40.7M/run (30% of loops); largest ~7.2M/run.
Conclusion: flat tail, no dominant loop left. Options (none measured): V-extension vectorization of elementwise loops (needs the scalar baseline compiled identically to stay fair), elementwise fusion, custom erf/exp kernels (changes numerics slightly; ~40 instr/call now).
Next phase = coverage. Blocker for models above ~100M params: weights inlined as MLIR text (7.6 bytes/param; ViT-B 661 MB, 6.15 GB RSS). Plan: export weights as function arguments, load from a binary file at run time (pk file I/O unverified), generate the driver.

### Whisper-tiny ENCODER, 30 s (1x80x3000 -> 1x1500x384), AME, LLVM_OPT=O3, counters off, random weights, pre-transposed Linears (warm RUN1, measured)
forward 7848.19M; in_matmul 2052.39M (26.2%); in_copy 18.13M; everything else 5777.67M (73.6%, derived). 74 matmul calls, 18,468,864,000 MACs (matches hand count: 2 convs + 24 Linear + 48 attention [12/layer]). 2 conv_1d rewritten; max abs err (sampled) 1.8e-6, projection err 1.1e-7, PASS. Matmul 0.1111 instr/MAC.
Hypothesis (not profiled): softmax expf ~2.2G (54M scores x ~40), GELU erff ~0.4G, remaining loops ~3.2G.
Spike wall time: >=19 min for 2 runs (~16G instr, RUN0 not recorded) -> <=~14M instr/s for the matrix-enabled Spike vs ~70M instr/s measured for scalar stock Spike (hypothesis: matrix Spike is several times slower). pk reads host files (verified, both Spikes).
Export: `python3 python/export_whisper.py tiny [--frames N] [--orig]` -> build_whisper_tiny_enc[N]p/ (random weights; config sizes for base/small/large-v3-turbo are from memory, check vs model cards). Pre-transposition verified in IR: 17 linalg.transpose left, 0 of constants.
Estimated (hand count, not run): Whisper-small encoder ~172G MACs (~9x tiny), i.e. hours per run at the observed Spike speed -> use RUNS=1 and/or reduced layers.

### Whisper-tiny DECODER STEP (1 new token, 32 cached tokens, 1500 encoder frames; logits 1x1x51865), AME, O3, counters off (warm RUN1, measured)
forward 34.09M; in_matmul 29.09M (85.3%); in_copy 0.80M (16 cache concats); everything else 4.20M (12.3%, derived). 129 matmul calls, 32,883,072 MACs (hand count exact: 33 matmuls + 96 attention [24/layer]). Step vs HF full-sequence decoder 1.5e-6; output err 1.3e-6; PASS.
Matmul 0.885 instr/MAC = 7.96x the encoder's 0.111 (M=1 uses 1 of 8 tile rows; hypothesis from tile geometry, consistent with the ~8x). Vocabulary projection = 19.9M of 32.9M MACs (~60%).
RUN0 176.0M vs RUN1 34.1M: 142M first-touch of the 385 MB of constants (mostly inside in_matmul). Scalar (measured, same config): forward 235.59M, in_matmul 230.59M (7.01 instr/MAC), in_copy 0.80M, everything else 4.20M (identical to AME). Ratios (derived): whole-forward 6.9x, matmul-only 7.9x (vs 34.6x / 63x for ResNet18 224).
Model: python/export_whisper_dec.py (custom step module; K caches stored transposed [H,hd,T]; checked vs HF decoder before export). HF's own decoder export produced tensor.scan/tensor.concat (not handled), so the custom module is used.
Idea (not done): matrix-vector role swap (weights on the 8-row side) would raise best case from 16 to 32 MACs/K-step (~2x), needs a suitable transposed tile load (spec not checked).

### Weights as function arguments (implemented; verified on the Whisper-tiny decoder step)
`--weights-as-args` on export_whisper.py / export_whisper_dec.py: parameters+buffers become trailing function args (torch.func.functional_call), written to <dir>/weights.bin (64B-aligned), IR has no weight constants. gen_driver.py loads weights.bin at run time with fopen/fread (pk reads host files: verified with a probe on both Spikes); run Spike from the repo root; raise SPIKE_MEM for big weights (make ... SPIKE_MEM=4096). Verified on the decoder step: model.mlir 385 MB -> 117 KB, weights.bin 189 MB (102 weight args), PASS with identical errors; AME RUN1 forward 34,143,861 (inline: 34,085,757, +0.17%, all in non-matmul, cause not isolated; in_matmul and in_copy identical). RUN0-RUN1 gap fell from 142M to 0.80M (first-touch moves into fread before timing), so RUNS=1 is accurate to ~2% for big models.
Size estimates (derived from 8 bytes/param of inline MLIR): Whisper-small decoder ~1.4 GB, large-v3-turbo decoder ~1.9 GB, large-v3-turbo encoder ~3.2 GB of MLIR text -> need this path.

## Do not use
Earlier figures 6.6x, 16.2x, 0.489 instr/MAC for ResNet18 AME (cold-start inflated). The warm 16.37x/16.78x above is a different, valid measurement.
9.19x / 8.80x are valid but pre-rewrite.

## Open items
1. Profile what is left of the ViT non-matmul (289M/run, 2-layer pre-transposed): ~200M of generated loops unidentified (derived, assumes erff/expf unchanged). Candidates: LayerNorm, softmax, residual adds, remaining 9 activation transposes, GELU arithmetic. Re-run the histogram on build_vit2p.
2. DONE (ViT): weight transposes were confirmed as the hot loops (see Profile findings) and removed by pre-transposing in the exporter. Not applied to other models yet.
2b. DONE for ViT: matmul step counters are now compile-time optional (AME_COUNT_STEPS=1); K-step is 14 instr. ResNet re-baseline pending.
2c. DONE for ViT: LLVM_OPT=O3 (opt -> llc). `opt` was not in the buddy-mlir LLVM build and had to be built (`cmake --build ~/buddy-mlir/llvm/build --target opt`). Re-baseline ResNet (scalar and AME alike) with tools/bench.sh.
3. ResNet memrefCopy: was 40% of AME forward at 224 (224.5M), now 10.5%. DONE: run-based fast path (src/memref_copy.c: merge contiguous inner dims, one memmove per run), checked natively on 20000 random strided cases and on Spike (see CURRENT BASELINE).
4. Fusing elementwise ops before bufferization: untested on this LLVM.
5. Matmul tuning (hoist msettilek, multi-accumulator): matmul is 31-48% of forward; K-step = 17 instr (incl. 3 counter instr), ~126 of 128 MACs.
6. Decode steps are matrix-vector (M=1): an 8-row tile would use ~1/8 of the unit (hypothesis); may need a different mapping.
7. int8 (mqma.b.mm) needs a quantized model; needed to fit the large LLMs in RAM.
8. conv_rewrite.py gaps: groups/depthwise, transposed conv, dilation, N>1 (conv1d done) (needed for Whisper/Moonshine/Canary/Kokoro/Mamba).
9. Target is the riscv-stc v0.5 matrix extension, not the task-group Zvame draft.

## Profile findings (ViT 2-layer, original export, hist covers RUN0+RUN1+pk, halved for per-run)
Per run: ame_hw K-loop ~410M (2 blocks, = 24.03M ksteps x 17); `forward` loops ~357.6M; erff ~50.6M (~42 instr/call if 1 call/GELU element); expf ~36.8M (~40/call); memcpy ~2.9M. Sum of non-matmul = 447.9M, matches the counter-derived 447.98M.
Unmapped (pk/kernel) 142.4M ~ RUN0-RUN1 gap 133.9M + startup.
Hot blocks in forward were weight-transpose loops (7-8 instr/element, strided reads): predicted vs histogram/2: QKV 14.16M/14.16M, fc1 18.87M/18.88M, fc2 16.52M/16.54M (within 0.1%). Six blocks = 99.5M per run (22% of non-matmul, 11.6% of forward).
tools/pchist.py usage: `pchist.py ELF HIST [FUNC] [NBLOCKS]`.

## Profiling (Spike PC histogram; command from memory, check `spike --help 2>&1 | grep -i hist`)
spike -g prints "PC Histogram" lines (pc count) to stderr at exit (both RUN0 and RUN1 and pk code are included):
  $HOME/riscv-stc/bin/spike -g --isa=rv64imafdcv_zicntr_matrix -m2048 $HOME/riscv/riscv64-unknown-elf/bin/pk build_vit2/resnet18.elf 2> build_vit2/hist.txt > /dev/null
  python3 tools/pchist.py build_vit2/resnet18.elf build_vit2/hist.txt
Generated code is one function, so look at the hottest 128-byte blocks and `riscv64-unknown-elf-objdump -d`; libm callees (erff, expf) show up by name in the function list.

## Workload plan (assignment: ResNet-50, ViT-B/16, Whisper small / large-v3-turbo, Moonshine tiny/base, Canary-Qwen 2.5B, Kokoro 82M, Phi-4-mini, Phi-4, Ministral 3 3B/8B, Gemma 3n E2B/E4B)
- ResNet18 is the baseline; ResNet-50 skipped (coverage argument). ViT-B/16 done at full size.
- fp32 fits 15 GB: Moonshine, Kokoro, Whisper small and large-v3-turbo; Canary-Qwen borderline at 20 GB. The 3B+ LLMs do not fit in fp32; use reduced-layer random-weight configs (2 layers, same widths) and label them as such; extrapolating by layer count is a hypothesis.
- Mamba (e.g. mamba-130m) planned for coverage: projections are matmuls, the selective scan and depthwise causal conv are not covered yet.
- Done: ResNet18, ViT-B/16, Whisper-tiny encoder. Next: Whisper decoder step (KV cache + cross-attention), weights-as-function-arguments loaded from a file (needed for Whisper-small and up), Moonshine, then reduced-config LLMs (prefill 128/512/2048, decode 512/2048). Consider a fused softmax for long-sequence attention.
