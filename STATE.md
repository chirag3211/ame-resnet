# STATE (Oct 8 2026)  -- PyTorch -> torch-mlir -> RV64 -> riscv-stc Spike (AME matrix extension)
Models so far: ResNet18 (224 and 64), ViT-B/16 (2-layer and full 12-layer), Whisper-tiny encoder + decoder step, Moonshine-tiny encoder, Whisper-small encoder (30 s and 3 s) + decoder step. Scalar baselines are done for every workload except the Whisper-small 30 s encoder (about 5 h per run, extrapolated; see the Whisper-small section). Next steps and open work: TODO.md. Label convention below:
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
(SUPERSEDED by 'ResNet18 CURRENT BASELINE' below: this table was measured with matmul counters ON and no LLVM opt. Kept for the 16.78x step in the progress line.)
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
(SUPERSEDED: original export, counters on, no O3. 2-layer: see the pre-transposed sections; 12-layer: see the build_vit12p section, AME 3262.79M and scalar 124,395M on the current pipeline. The 4953.64M number below is retired.)
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
Constant transposes in the IR 9 -> 0; total linalg.transpose 26 -> 9. Saving exceeds the weight transposes alone (see Profile findings); the rest is attributed (hypothesis) to the simpler attention graph. The full 12-layer pre-transposed run is done: 3262.79M with O3 + counters off (see the build_vit12p section); the earlier ~4.0G hypothesis (pre-transposing only) was not tested separately.

Matmul efficiency 0.1356 instr/MAC (same as ResNet18) and non-matmul share 52% (derived), both for the original export with counters on; on the current pipeline they are 0.1116 instr/MAC and 39.9%.
338 calls = 50 matmuls (48 Linear + patch conv + head) + 24 batch_matmul ops x 12 heads (derived; matches).
Full-model run: 661 MB vit.mlir (~7.6 bytes/param), `make` wall time 44m06s including export-independent lowering and both Spike runs, max RSS 6.15 GB (of the largest process in the tree; which one is not identified).
Scalar ViT 12-layer is now measured (build_vit12p section): 124,395M, just below (0.5%) the low end of the 125-135B hypothesis, but the whole-forward ratio is 38.13x, not the hypothesised ~25-27x, because the AME forward fell to 3262.79M (pre-transposed weights, counters off, O3). The old hypothesis assumed the 4953.64M AME forward.
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
- DONE: ResNet18 re-baselined (see CURRENT BASELINE); the first ResNet18 table was measured with counters ON and no opt. The derived prediction at 224 AME, in_matmul ~202.8M from removing 3 x 14.4M counter instr, was confirmed: 202.51M measured.

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
Next phase = coverage. Blocker for models above ~100M params: weights inlined as MLIR text (7.6 bytes/param; ViT-B 661 MB, 6.15 GB RSS). Plan: export weights as function arguments, load from a binary file at run time (pk file I/O: verified, see 'Weights as function arguments'; implemented), generate the driver.

### Whisper-tiny ENCODER, 30 s (1x80x3000 -> 1x1500x384), AME, LLVM_OPT=O3, counters off, random weights, pre-transposed Linears (warm RUN1, measured)
forward 7848.19M; in_matmul 2052.39M (26.2%); in_copy 18.13M; everything else 5777.67M (73.6%, derived). 74 matmul calls, 18,468,864,000 MACs (matches hand count: 2 convs + 24 Linear + 48 attention [12/layer]). 2 conv_1d rewritten; max abs err (sampled) 1.8e-6, projection err 1.1e-7, PASS. Matmul 0.1111 instr/MAC.
SCALAR (same dir build_whisper_tiny_encp, `LLVM_OPT=O3 SPIKE_ISA=rv64gc_zicntr`, measured; ratios derived): RUN0 136,265,375,079, RUN1 **135,622,397,346**; in_matmul 129,826,596,148 (RUN1); in_copy 18,129,618 (2 calls); everything else 5,777,671,580 (identical to the AME value above, 5777.67M). STATS backend=reference-scalar matmul_calls=74 macs=18,468,864,000 -> 7.029 instr/MAC. PASS, errors identical to AME (max abs err sampled 1.758e-6, |gold|max 3.48, sumsq 3.98e-9, projection 1.09e-7). AME vs scalar: whole-forward **17.28x**, matmul-only 63.26x; cold/warm gap 642,977,733 (scalar).
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

### Moonshine-tiny ENCODER, 10 s (1x160000 raw audio samples -> 1x415x288), AME, LLVM_OPT=O3, counters off, random weights inlined, pre-transposed Linears (measured; ratios derived)
Export: `python3 python/export_moonshine.py tiny [--samples N] [--orig] [--weights-as-args]` -> build_moonshine_tiny_enc10sp/ (the script must live in python/ next to ame_export.py). Pre-transposed vs original max abs diff 0.0. conv_rewrite: 3 convs rewritten, 0 left; 9 distinct matmul signatures; model.mlir 61 MB (0 weight args).
| | RUN0 (cold) | RUN1 (warm) |
|---|---|---|
| forward | 2,074,692,566 | **1,856,301,995** |
| in_matmul | 519,775,177 | 481,374,354 (25.9%) |
| in_copy (50 calls) | 93,306,715 | 88,404,666 (4.8%) |
| everything else (derived) | 1,461,610,674 | 1,286,522,975 (69.3%) |
RUN0-RUN1 gap 218.39M (11.8% of RUN1; weights are inlined, so first-touch is inside the forward). Report RUN1.
STATS: matmul_calls=136, macs=4,336,487,440 -> matmul 0.1110 instr/MAC (same as Whisper/ResNet). Correctness: max abs err (sampled) 6.437e-6, |gold|max 5.15, sumsq rel err 2.40e-9, projection err 1.32e-6, PASS.
Call/MAC reconciliation (derived from r18_3_calls.mlir signatures + hand count; matches STATS exactly): 3 conv matmuls (288x127@127x2499, 576x2016@2016x831, 288x1728@1728x415; 1,262,904,480 MACs) + 6 layers x (4 projections 415x288@288x288, fc1 415x288@288x1152, fc2 415x1152@1152x288, QK^T and AV as 8x415x36 / 8x415x415 batch_matmul = 16 per-head calls) = 132 calls, 3,073,576,320 MACs + 1 rotary matmul (bmm 1x16x1 @ 1x1x415, 6,640 MACs) = 136 calls. Config verified from transformers MoonshineConfig() defaults (printed in the venv): hidden 288, 6 layers, 8 heads, FFN 1152, head_dim 36, partial_rotary_factor 0.9 -> rotary dim int(36*0.9)=32 -> 16 frequencies (matches the 1x16x1 rotary bmm in the IR); 415 positions.
Open: the 69% non-matmul share and the 88M in_copy (50 copies, vs 33M for the much larger Whisper-small encoder) are not profiled; cause of the copy cost not isolated (hypothesis: conv rewrite/padding).

### Weights as function arguments (implemented; verified on the Whisper-tiny decoder step)
`--weights-as-args` on export_whisper.py / export_whisper_dec.py: parameters+buffers become trailing function args (torch.func.functional_call), written to <dir>/weights.bin (64B-aligned), IR has no weight constants. gen_driver.py loads weights.bin at run time with fopen/fread (pk reads host files: verified with a probe on both Spikes); run Spike from the repo root; raise SPIKE_MEM for big weights (make ... SPIKE_MEM=4096). Verified on the decoder step: model.mlir 385 MB -> 117 KB, weights.bin 189 MB (102 weight args), PASS with identical errors; AME RUN1 forward 34,143,861 (inline: 34,085,757, +0.17%, all in non-matmul, cause not isolated; in_matmul and in_copy identical). RUN0-RUN1 gap fell from 142M to 0.80M (first-touch moves into fread before timing), so RUNS=1 is accurate to ~2% for big models.
Size estimates (derived from 8 bytes/param of inline MLIR): Whisper-small decoder ~1.4 GB, large-v3-turbo decoder ~1.9 GB, large-v3-turbo encoder ~3.2 GB of MLIR text -> need this path.

### Moonshine-tiny ENCODER, 10 s: SCALAR build and AME-vs-scalar (warm RUN1, measured; ratios derived)
Command: same as the AME build but `rm -f $D/resnet18.elf $D/*.o` first and `LLVM_OPT=O3 SPIKE_ISA=rv64gc_zicntr` (no USE_AME, stock spike). Same dir build_moonshine_tiny_enc10sp, same model.mlir, same driver.
| | RUN0 (cold) | RUN1 (warm) |
|---|---|---|
| forward | 32,069,448,808 | **31,851,064,097** |
| in_matmul | 30,514,537,279 | 30,476,136,456 (95.7%) |
| in_copy (50 calls) | 93,306,715 | 88,404,666 (0.28%) |
| everything else (derived) | | 1,286,522,975 (4.0%) |
STATS backend=reference-scalar matmul_calls=136 macs=4,336,487,440 -> scalar matmul 7.028 instr/MAC. PASS, errors identical to AME (max abs err sampled 6.437e-6, sumsq rel err 2.40e-9, projection err 1.32e-6).
AME vs scalar (derived): whole-forward 17.16x (31851.06M / 1856.30M); matmul-only 63.31x (30476.14M / 481.37M). Everything else identical in both builds (1,286,522,975, exact). RUN0-RUN1 gap 218.38M scalar vs 218.39M AME (first-touch cost the same in both).

### ViT-B/16 2-layer, pre-transposed, O3, counters off: SCALAR build and AME-vs-scalar (warm RUN1, measured; ratios derived)
Dir build_vit2p, `make B=build_vit2p LLVM_OPT=O3 SPIKE_ISA=rv64gc_zicntr resnet-spike SIZE=224` after `rm -f build_vit2p/resnet18.elf build_vit2p/*.o`.
| | AME (O3, counters off) | scalar (O3) |
|---|---|---|
| RUN0 forward | | 21,531,591,653 |
| RUN1 forward | 562.19M | **21,422,072,116** |
| in_matmul | 337.87M | 21,197,745,690 |
| in_copy | 0 | 0 |
| everything else (derived) | 224.32M | 224.33M (224,326,426) |
STATS backend=reference-scalar matmul_calls=58 macs=3,024,282,624 -> scalar matmul 7.009 instr/MAC. TOP1 829 (matches PyTorch), max abs err 3.636e-6 (identical to AME). RUN0-RUN1 gap 109.52M (same as the AME gap, 109.5M).
AME vs scalar (derived): whole-forward 38.10x; matmul-only 62.74x. The AME column uses the 2-decimal-M figures recorded above, so ratios are good to ~4 digits.
Full 12-layer ViT on the current pipeline: see the next section.

### ViT-B/16 12-layer, pre-transposed (build_vit12p), O3, counters off: AME and SCALAR (measured; ratios derived). This replaces the old 4953.64M number.
Export `python3 python/export_vit.py 12` (-> build_vit12p), `ln -sf vit.mlir build_vit12p/resnet18.mlir`, both backends with LLVM_OPT=O3 (logs build_vit12p/log_ame.txt, log_scalar.txt).
| | AME RUN0 | AME RUN1 | scalar RUN0 | scalar RUN1 |
|---|---|---|---|---|
| forward | 3,840,857,492 | **3,262,794,077** | 124,973,187,275 | **124,395,123,392** |
| in_matmul | 2,392,567,965 | 1,960,347,135 | 123,524,897,280 | 123,092,676,450 |
| in_copy | 0 | 0 | 0 | 0 |
| everything else (derived) | | 1,302,446,942 | | 1,302,446,942 |
STATS: matmul_calls=338, macs=17,563,828,224 (both backends). TOP1 272 = PyTorch in both; max abs err 4.888e-6 in both (identical). Derived: AME matmul 0.1116 instr/MAC, scalar matmul 7.008 instr/MAC; whole-forward 38.13x; matmul-only 62.79x; non-matmul = 39.9% of AME forward. "Everything else" is IDENTICAL in AME and scalar to the instruction (1,302,446,942). The earlier prediction for the scalar run (~124G per run, from 7.009 instr/MAC x 17.564G + ~1.3G) came within 0.1%.
Cold/warm: gap 578.06M in both backends (AME 578,063,415; scalar 578,063,883), of which 432,220,830 sits in in_matmul in both (identical). Hypothesis: first touch of the inline weights (page faults) is counted inside the matmul calls and is backend independent. So RUN0/RUN0 ratios are comparable.
Old build (original export, counters on, no O3) was 4953.64M AME; the current pipeline is 1.52x lower (3262.79M). The old number is retired; do not compare it with any scalar number.
ViT layer scaling (derived): per-layer MACs (17,563.83M - 3,024.28M)/10 = 1,453.96M, equal to the hand count (197 x (4x768^2 + 2x768x3072) + 2x197^2x768 = 1,453.9M); fixed part 116.4M (patch embed 115.6M + head). Whole-forward ratio 38.10x at 2 layers and 38.13x at 12 layers. Two points cannot prove linearity; a 4-layer point is still open (TODO.md).

### Whisper-small (AME for every run; scalar for the 3 s encoder and the decoder step; warm RUN1 unless noted; measured, shares derived)
Encoder 30 s (dir build_whisper_small_encpw, weights-as-args, SPIKE_MEM=8192, log run.log): RUN0 57,083,722,312, RUN1 **53,611,568,800**; in_matmul 19,031,629,026 (35.5%) in both runs; in_copy 33,133,266 (2 calls, 0.06%); everything else (derived) 34,546,806,508 (64.4%). STATS macs=172,081,152,000, matmul_calls=362 (12 layers x 30 + 2 convs). Matmul 0.1106 instr/MAC. PASS: max abs err (samples) 5.484e-6, |gold|max 4.56, sumsq rel err 6.89e-9, projection err 7.14e-7. Cold/warm gap 3,472,153,512 (6.5% of RUN1), entirely in the non-matmul part (matmul and copy are identical in RUN0 and RUN1). Cause not identified. The first guess (first touch of the attention-score buffers, 12 x 1500^2 x 4 B = 108 MB) is doubtful: at 3 s the same buffers are 12 x 150^2 x 4 B = 1.08 MB (100x smaller) but the gap is 129.65M, only 26.8x smaller than 3.47G (derived), and the 3 s gap is the same in AME and scalar. So a RUN0-only result can overstate by several percent here.
Encoder 3 s (build_whisper_small_enc300pw, 300 mel frames = 150 positions, driver regenerated with 2 runs; AME and scalar, measured; shares/ratios derived): AME RUN0 2,577,479,808, RUN1 **2,447,829,072**; scalar RUN0 95,540,606,504, RUN1 **95,410,969,978**; in_matmul AME 1,500,225,642 / scalar 94,463,366,548 (identical in RUN0 and RUN1 within each backend); in_copy 3,368,466 (2 calls) in all four. Everything else (derived) 944,234,964 in RUN1 for BOTH backends (identical to the instruction; 38.6% of the AME forward; non-matmul incl. copy 38.7%). STATS macs=13,475,635,200, matmul_calls=362; AME matmul 0.1113 instr/MAC, scalar matmul 7.010 instr/MAC. AME vs scalar: warm whole-forward **38.98x**, matmul-only 62.97x; cold (RUN0/RUN0) 37.07x. PASS in both, identical errors: max abs err (samples) 4.768e-6, |gold|max 4.26, sumsq 1.42e-8, projection 1.19e-6. Cold/warm gap 129,650,736 (AME) and 129,636,526 (scalar), 5.3% of AME RUN1, all in non-matmul (matmul and copy identical in RUN0 and RUN1); the gap is the same in both backends. (An earlier single-run driver gave RUN0 2,577,476,237 AME / 95,540,608,256 scalar; the 3.6k / 1.8k differences come from re-generating the driver and are 0.0001%.) Do NOT extrapolate this to 30 s: attention is 3.1% of MACs at 150 positions and 24.1% at 1500 (derived), and softmax work grows with length^2 ("everything else" share, copies excluded: 38.6% at 3 s vs 64.4% at 30 s; with copies 38.7% vs 64.5%).
Decoder step (build_whisper_small_dec32pw, 32 cached tokens, 1500 encoder frames, weights-as-args): RUN0 181,302,052, RUN1 **176,884,705**; in_matmul 147,447,393 (83.4%, identical in both runs); in_copy RUN1 4,811,304 (48 calls, 2.7%); everything else (derived) 24,626,008 (13.9%). STATS macs=167,179,008, matmul_calls=673 = 12 x 8 linears + head + 12 layers x 12 heads x 4 attention matmuls. Hand count matches exactly: linears 138,922,752 (head 39,832,320 = 23.8%) + attention 28,256,256 (33 positions self, 1500 cross). Matmul 0.882 instr/MAC (M=1 tile waste; Whisper-tiny step 0.885). Compare (re-run on spike_out.txt, python/compare_tensor.py): PASS, max abs err (samples) 3.666e-6, |gold|max 2.35, sumsq 2.28e-8, projection 4.56e-7.
Decoder step SCALAR (same dir, `LLVM_OPT=O3 SPIKE_ISA=rv64gc_zicntr SPIKE_MEM=4096`, measured; ratios derived): RUN0 1,206,530,812, RUN1 **1,202,122,736**; in_matmul 1,172,685,424; in_copy RUN1 4,811,304 (48 calls); everything else 24,626,008 (identical to AME to the instruction). STATS matmul_calls=673 macs=167,179,008 -> 7.015 instr/MAC. PASS, errors identical to the AME compare (3.666e-6, 2.28e-8, 4.56e-7). AME vs scalar: whole-forward **6.80x**, matmul-only 7.95x. The earlier prediction (~1.20G scalar forward, ~6.8x) held (1.202G, 6.80x).
MAC cross-check with the teammate: Whisper-small encoder 30 s hand count and STATS both give 172,081,152,000 = his 172.0812 GMAC (exact).
Scalar baselines for Whisper-small: encoder 3 s and decoder step measured (above). NOT run: encoder 30 s (172G MACs x 7.0 = ~1.2T instr, about 5 hours per run at 70M instr/s). Extrapolated, never measured: ~23.0x whole-forward, from the identity in the summary below with n = 0.645 (non-matmul share of the AME forward, copy included) and R = 62.97 (matmul-only ratio measured on the 3 s encoder). The result moves with R: using the 30 s encoder's own AME matmul cost (0.1106 instr/MAC) with the scalar 7.008-7.029 instr/MAC seen on other models gives R = 63.4-63.6 and 23.1-23.2x, so quote ~23x (23.0-23.2 depending on R). The same prediction method was right for the 3 s encoder (37x predicted, 37.07x cold / 38.98x warm measured) and the decoder step (6.8x predicted, 6.80x measured).

### Summary of AME vs scalar ratios so far (warm RUN1, derived)
| Workload | whole-forward | matmul-only | scalar matmul instr/MAC | non-matmul share of AME forward |
|---|---|---|---|---|
| ResNet18 224 | 34.64x | 63.09x | ~7.0 | 45.8% (copy 10.5% + other 35.3%) |
| ResNet18 64 | 35.43x | 66.83x | ~7.0 | 47.7% |
| ViT-B/16 2-layer pre-transposed | 38.10x | 62.74x | 7.009 | 39.9% |
| Moonshine-tiny enc 10 s | 17.16x | 63.31x | 7.028 | 74.1% (copy 4.8% + other 69.3%) |
| ViT-B/16 12-layer pre-transposed | 38.13x | 62.79x | 7.008 | 39.9% |
| Whisper-tiny dec step | 6.9x | 7.9x | 7.01 | 14.7% (matmul 0.885 instr/MAC, M=1) |
| Whisper-tiny enc 30 s | 17.28x | 63.26x | 7.029 | 73.9% |
| Whisper-small enc 30 s | not run (extrapolated ~23x, 23.0-23.2 depending on R, see below) | | | 64.5% (copy 0.06% + other 64.4%) |
| Whisper-small enc 3 s | 38.98x (cold/cold 37.07x) | 62.97x | 7.010 | 38.7% |
| Whisper-small dec step | 6.80x | 7.95x | 7.015 | 16.6% (copy 2.7% + other 13.9%) |
Reading (derived/hypothesis): the matmul ratio is ~63x for every large-M workload because it is set by the kernel (one mfma = 128 MACs per ~14-instr K-step vs ~7 instr/MAC scalar), not by the model. The whole-forward ratio is set by the non-matmul share, which stays on the scalar core in both builds (identical "everything else" in every pair that was checked). The non-matmul share depends on sequence length (Whisper-small encoder: 38.7% at 3 s, 64.5% at 30 s, copies included) and falls as d grows (Whisper-tiny 30 s 73.9% vs Whisper-small 30 s 64.5%; hypothesis: matmul work scales ~d^2, elementwise ~d). Decode steps (M=1) use 1 of 8 tile rows, so even the matmul ratio is only 7.9x. Exact identity (derived, valid because the non-matmul work is identical in both builds): whole-forward ratio = n + R x (1 - n), with n = non-matmul share of the AME forward and R = matmul-only ratio. It reproduces every measured pair: ResNet18 224 34.64x, ViT 12-layer 38.12x (measured 38.13x), Moonshine 17.16x, Whisper-tiny encoder 17.28x, Whisper-small 3 s 38.98x, decoder steps 6.9x and 6.80x. So given R (set by the kernels: ~63 for large M, ~7.9 for M=1) the model-level ratio depends only on n. This is an accounting identity, not an independent model or result: its use is that n comes cheaply from one AME run and R from the kernel, so the scalar ratio can be estimated without a scalar run (as done for the Whisper-small 30 s encoder, extrapolated).

## fp16 on AME: findings (Oct 8 2026; investigated, NOT ported; the whole pipeline is still fp32)
Question asked: can the pipeline be run in fp16 (the teammate's workload study used fp16 shapes)? Answer so far: the instruction exists, but the accumulator is fp16, so a straight port is not numerically usable for long K.
- **Spec** (~/work/riscv-matrix-spec): e16 loads/stores/moves exist (mlae16.m, mlbe16.m, mlce16.m, msce16.m ...); ext-type.adoc:548 says e16 load/store/move are used for FP16 data, :600 for BF16, :181 for int16. The spec's e16 matmul examples store the C tile with 16-bit element stores (stride n*2).
- **Spike** (~/work/riscv-stc-sim, built into ~/riscv-stc/bin/spike): has mfma_hf_mm (fp16), next to mfma_f_mm (fp32), mfma_d_mm (fp64), generic mfma_mm, and sparse mfma_spa_* / mfma_spb_* forms (not used here). `mfmax_*_mm` is an ELEMENTWISE MAX of two tiles (f16_max/f32_max/...), i.e. for max-pooling; it is NOT a widening multiply-add. int8 is mqma_b_mm / mqmau_b_mm. e32 also has a tf32 mode (f32_to_tf32) selected by the mfp32 field.
- **Encodings** (include/riscv/encoding.h): MATCH_MFMA_HF_MM 0x22001877 (funct3=1, funct7=0x11), MATCH_MFMA_F_MM 0x22002877 (funct3=2). msetfp16 = 0x203e077 (field id in the rs1 slot = x7, value in the rs2 slot), msetfp32 = 0x2046077 (field id x8). mtype bits: mfp16 = bits 10-11, mfp32 = bits 12-13 (set_fp(rd, simm5, 10) for fp16, 12 for fp32). MTYPE_FP16 = 0x1, MTYPE_BF16 = 0x2 (m_ext_macros.h:12-13). After `msetsew e16` + `msetfp16 1`, mtype reads 0x401 (measured).
- **Accumulator is fp16 (read from source and measured).** MXU_VFP_VV_LOOP (m_ext_macros.h:1207) case e16 reads/writes the accumulator as float16_t (bfloat16_t for BF16) and does `td = f16_mulAdd(ts1, ts2, td)`, so every partial sum is rounded to fp16. The e32 case uses float32_t. Probe tests/ame_fp16_probe.c (all-ones inputs, accumulate in acc0 across K in one tile): K=8 -> 8, K=1024 -> 1024, K=2048 -> 2048, K=2560 -> 2048, K=4096 -> 2048 (measured). The sum stalls at 2048 (fp16 spacing 2 there). All-ones is the worst case; real signed data stalls less, but error size on real shapes is NOT measured yet.
- **Tile (e16) granted m=8 k=8 n=8** (measured), vs 8/4/4 for fp32: one fp16 mfma = 512 MACs vs 128 for fp32 (derived). Hypothesis (not measured): matmul cost could drop to ~0.03 instr/MAC if the K-step stays ~14 instr.
- **ISA string needs zfh.** M_FLOAT_TYPE_CHECK requires EXT_ZFH for e16 (EXT_MATRIX_ZMF8E* for e8 float, 'F' for e32, 'D' for e64). Without it the first mfma_hf traps ("An illegal instruction was executed", va/inst 0x22101877, seen with rv64imafdcv_zicntr_matrix). Working string: `rv64imafdcv_zfh_zicntr_matrix` (measured; the probe ran). Order: z-extensions between v and matrix.
- Probe build/run: `riscv64-unknown-elf-gcc -O2 -march=rv64gc -mabi=lp64d -Iinclude tests/ame_fp16_probe.c -o build/ame_fp16_probe.elf` then `$HOME/riscv-stc/bin/spike --isa=rv64imafdcv_zfh_zicntr_matrix $HOME/riscv/riscv64-unknown-elf/bin/pk build/ame_fp16_probe.elf`. FP16VAL defaults to 1 (correct).
- **Consequences if fp16 is pursued (all hypotheses/derived, none measured):** (1) K=768 (ViT) and K=1152 (Moonshine FFN) accumulated in fp16 will have errors far above the current ~1e-6; the PASS criterion would need a tolerance or K-chunking (accumulate e.g. 64-256 K in fp16, then convert and add into fp32 in software), and the flush cost changes every instr/MAC number. (2) A scalar fp16 baseline needs zfh in the compiler (-march=...zfh) and in Spike, otherwise _Float16 is emulated through fp32 and the baseline is unfairly slow. (3) The MLIR side assumes f32: conv_rewrite.py (f32 only), matmul_to_call.py (@ame_mm_f32_<n>), gen_shims.py, memref_shims.c, drivers and compare scripts; erff/expf have no fp16 libm (upcast adds instructions). (4) fp16 is a separate experiment, not a re-run; keep the fp32 results as the main data.
- Decision (Oct 8): keep fp32 as the main pipeline; next fp16 step is a small error-vs-chunk-size test on real shapes (TODO.md), before touching the MLIR pipeline.

## Relation to the teammate's workload study (Workload_Characterization.html; Oct 8)
- His page is STATIC: it records every PyTorch op with exact shapes on the meta device (no weights), fp16, batch 1, TorchDispatchMode, and reports MACs, bytes (inputs + outputs per op, no cache reuse, upper bound), intensity. Exception: Kokoro runs with real weights in fp32. Nothing else on the page is stated to differ.
- This repo is MEASURED: retired instructions on Spike, fp32, seeded random weights, reduced-layer or custom export graphs in places (pre-transposed ViT, hand-written Whisper decoder step). So his bytes and intensity are NOT comparable with anything computed from fp32 tensors here (2x bytes); his MACs ARE comparable (precision independent).
- MAC cross-check: ViT-B/16 17.5638 GMAC (his) vs 17.564G (STATS, build_vit12p): match. Whisper-small encoder 30 s 172.0812 GMAC (his) vs 172,081,152,000 (STATS and hand count): exact. Moonshine tiny encoder 4.4026 GMAC (his, 10 s) vs 4.336G (STATS): ours is 1.5% lower; cause not investigated (hypothesis: small non-matmul-counted ops, e.g. the rotary, pooling or other ops his op-recorder counts). Whisper-tiny, Whisper-small decoder step and ResNet18 are not on his page as run here; ResNet-50 is (4.0892 GMAC) and is skipped here.
- Agreement: his decode steps have intensity <= 0.8 FLOP/B (memory-bound); here the Whisper-tiny decode step gets only 6.9x whole-forward and 0.885 instr/MAC for matmul (M=1). His non-MAC byte shares (softmax, norm, elementwise, data movement) correspond to the scalar-core work that is 40-74% of AME forward instructions here.
- Results page for this repo, in his style (same layout, tokens, filters): AME_Measured_Results.html (generated outside the repo; built from the numbers in this file; needs the new scalar rows added, see TODO.md).

## Do not use
Earlier figures 6.6x, 16.2x, 0.489 instr/MAC for ResNet18 AME (cold-start inflated). The warm 16.37x/16.78x above is a different, valid measurement.
9.19x / 8.80x are valid but pre-rewrite.

## Open items
(Full prioritized plan and what is done: TODO.md. Items below are the original list.)
0. Review issues found Oct 8 (details in TODO.md): matmul ratio ~63x is mostly tile size and the scalar baseline is a plain C i-k-j loop (unit stride, no register blocking, no RVV; the scalar Spike ISA is rv64gc_zicntr, so an RVV baseline needs a v ISA string); non-matmul share (39-74% of AME forward) is unexplained for Moonshine/Whisper (only ViT profiled); AME vs scalar rows should come from one pipeline version (ViT 12-layer is now re-run on the current pipeline; confirm the same for the ResNet18 and Whisper-tiny rows); compare_tensor.py is sampled, not exact; Makefile does not track USE_AME / LLVM_OPT / conv_rewrite.py; results are copied by hand.
1. PARTLY DONE (see 'Profile after O3 + counters off + run-based copy': flat tail, erff+expf 39% of non-matmul). Profile what is left of the ViT non-matmul (289M/run at the time, 224.3M now, 2-layer pre-transposed): ~200M of generated loops unidentified (derived, assumes erff/expf unchanged). Candidates: LayerNorm, softmax, residual adds, remaining 9 activation transposes, GELU arithmetic. Re-run the histogram on build_vit2p.
2. DONE (ViT): weight transposes were confirmed as the hot loops (see Profile findings) and removed by pre-transposing in the exporter. Also applied to the Whisper and Moonshine exports (pre-transposed Linears); not applied to ResNet18 (no Linear weights to transpose except the head).
2b. DONE for ViT: matmul step counters are now compile-time optional (AME_COUNT_STEPS=1); K-step is 14 instr. ResNet18 re-baselined (CURRENT BASELINE).
2c. DONE for ViT: LLVM_OPT=O3 (opt -> llc). `opt` was not in the buddy-mlir LLVM build and had to be built (`cmake --build ~/buddy-mlir/llvm/build --target opt`). ResNet18 re-baselined (scalar and AME alike) with tools/bench.sh: see CURRENT BASELINE.
3. ResNet memrefCopy: was 40% of AME forward at 224 (224.5M), now 10.5%. DONE: run-based fast path (src/memref_copy.c: merge contiguous inner dims, one memmove per run), checked natively on 20000 random strided cases and on Spike (see CURRENT BASELINE).
4. Fusing elementwise ops before bufferization: untested on this LLVM.
5. Matmul tuning (hoist msettilek, multi-accumulator): matmul is 26-61% of AME forward for encoders and full models (derived: Moonshine and Whisper-tiny encoder ~26%, ResNet18 224 54%, ViT and Whisper-small 3 s ~60%) and 83-85% for decode steps; K-step = 14 instr with counters off (17 with them), 128 MACs per mfma. The 31-48% / 17 figures were the counters-on numbers.
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
- Done (AME): ResNet18, ViT-B/16, Whisper-tiny encoder + decoder step, Moonshine-tiny encoder (10 s). Done (scalar too): ResNet18, ViT 2-layer and 12-layer, Whisper-tiny encoder and decoder step, Whisper-small encoder 3 s and decoder step, Moonshine-tiny. Whisper-small encoder 30 s is AME only (scalar would take about 5 h per run; extrapolated, see the Whisper-small section). Next: see TODO.md. Remaining on the original plan: Moonshine-base, then reduced-config LLMs (prefill 128/512/2048, decode 512/2048). Consider a fused softmax for long-sequence attention.
