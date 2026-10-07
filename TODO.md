# TODO (Oct 8 2026). Numbers and details live in STATE.md; this file is status + next steps.
Convention: [x] done, [~] in progress / running, [ ] not started. Tags: (measured) (derived) (hypothesis).

## Done
- [x] Pipeline: torch -> torch-mlir -> conv_rewrite -> bufferize -> external AME matmul calls -> RV64 -> Spike (AME and scalar backends)
- [x] ResNet18 (64, 224) AME + scalar, current baseline (counters off, O3, run-based memrefCopy): 34.64x whole forward at 224
- [x] ViT-B/16: 2-layer AME + scalar on the current pipeline (38.10x whole, 62.74x matmul); 12-layer AME on the OLD build only
- [x] Whisper-tiny encoder (AME only) and decoder step (AME + scalar, 6.9x)
- [x] Moonshine-tiny encoder 10 s: AME + scalar (17.16x whole, 63.31x matmul), PASS in both
- [x] Weights-as-function-arguments path (verified on the Whisper-tiny decoder step)
- [x] fp16 investigation: Spike has mfma_hf_mm but the accumulator is fp16 (stalls at 2048, measured); tile 8x8x8; needs zfh in ISA. Decision: stay on fp32 (STATE.md "fp16 on AME")
- [x] Compared with the teammate's study: precision (fp16 shapes vs fp32 here), MAC cross-check (ViT matches, Moonshine 1.5% lower)
- [x] Results page in the teammate's style (AME_Measured_Results.html)

## In progress
- [~] ViT-B/16 12-layer on the current pipeline, AME and scalar, dir build_vit12p (`python3 python/export_vit.py 12`, `ln -sf vit.mlir build_vit12p/resnet18.mlir`, then both backends with LLVM_OPT=O3, rm -f $D/resnet18.elf $D/*.o between). Main driver runs forward twice (RUN0 cold, RUN1 warm; report RUN1). Scalar expected 1 h+ (hypothesis, ~125-135G instr/run)
- [~] Whisper-small (running as of commit 6715a58)

## Next: easy (hours)
- [ ] Makefile: stamp USE_AME / LLVM_OPT / ISA into $(B)/.flags and depend on it; make resnet18_llvm.mlir depend on mlir/conv_rewrite.py (stale .o gave wrong-backend risk)
- [ ] tools/record.sh: run a build and append one JSON line (model, dir, backend, flags, git commit, RUN0/RUN1 counters, STATS, PASS) to results.jsonl; generate the results page DATA from it (no hand copying)
- [ ] Scalar runs still missing: Whisper-tiny encoder, ViT 12-layer (running), Whisper-small, Moonshine-base when exported
- [ ] Re-run all models on ONE pipeline version (same commit, O3, counters off) so rows are comparable; retire the old ViT 12-layer 4953.64M number
- [ ] Update the results page with the new scalar rows (Moonshine, ViT 2-layer, ViT 12-layer) and the 17.16x / 38.10x ratios
- [ ] Investigate Moonshine MACs: ours 4.336G vs the teammate's 4.4026G (1.5%); find which ops the op-recorder counts that the matmul counter does not
- [ ] Label everything: "scalar = naive C triple loop", "PASS = sampled check", "fp32 random weights, reduced configs" on the page and in STATE.md

## Next: medium (a day each)
- [ ] Profile the non-matmul share (40-74% of AME forward): run tools/pchist.py on Moonshine-tiny and Whisper-tiny encoder; predict from shapes (expf/erff counts x ~40 instr, loop element counts) and compare to the histogram, as done for ViT
- [ ] Moonshine in_copy 88M (50 copies) vs 33M for the larger Whisper-small encoder: log shape and element count of every memrefCopy call
- [ ] Fairer scalar baseline: (a) register-blocked scalar matmul, (b) RVV matmul (vfmacc; the Spike ISA already has v), compile the non-matmul code identically for each; report ratios against all baselines
- [ ] Exact correctness mode in compare_tensor.py (all elements) for at least Moonshine-tiny and one ViT size
- [ ] Sensitivity analysis instead of a cycle claim: model cycles = non-matmul instr x CPI + K-steps x mfma latency, plot whole-forward ratio vs mfma latency (1..16)
- [ ] Check layer extrapolation: ViT with 1, 2, 4 layers, confirm forward grows linearly (supports the reduced-layer LLM plan, still a hypothesis)
- [ ] fp16 error test (small program in tests/, reuse tests/ame_fp16_probe.c wrappers): random K=768 and K=1152 matmuls through mfma_hf with K-chunking (accumulate 64/128/256 K in fp16, flush to fp32 in software) vs an fp32 reference; record max error and the instr/MAC including the flush cost. Run with ISA rv64imafdcv_zfh_zicntr_matrix
- [ ] Re-baseline ResNet18 and ViT with tools/bench.sh after the Makefile/record changes

## Next: hard (open-ended)
- [ ] Coverage: conv_rewrite.py for depthwise/grouped/dilated/transposed conv and N>1 (needed for Kokoro, Canary, Mamba); LSTM and FFT or exclude and say so; weights-as-args for everything >= Whisper-small; reduced-layer random-weight configs for the 3B+ LLMs (Phi-4-mini, Phi-4, Ministral 3, Gemma 3n); Canary-Qwen borderline at 20 GB
- [ ] Reduce the non-matmul share once profiled: fused softmax, elementwise fusion before bufferization, vectorized elementwise loops, custom erf/exp kernels (changes numerics slightly)
- [ ] Matmul tuning: hoist msettilek, multi-accumulator (matmul is 26-54% of AME forward)
- [ ] Decode steps (M=1): matrix-vector role swap (weights on the 8-row side) might double the best case; spec not checked
- [ ] int8 (mqma.b.mm) with a quantized model; needed to fit the large LLMs
- [ ] fp16 port ONLY if the error test passes: f16 export, f16 paths in conv_rewrite.py / matmul_to_call.py / gen_shims.py / memref_shims.c / drivers / compare, scalar fp16 baseline (zfh in gcc and Spike), upcasts for erff/expf. Treat as a separate labelled experiment on one model (Moonshine-tiny), not a re-run of the study
- [ ] Real timing: needs a cycle-accurate model or RTL for AME; until then use the sensitivity analysis above, never a hardware speedup claim
- [ ] Spike speed (matrix Spike ~14M instr/s vs ~70M scalar, hypothesis): for big models use RUNS=1 with weights-as-args and state the cold/warm error bound
