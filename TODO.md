# TODO (Oct 8 2026). Numbers and details live in STATE.md; this file is status + next steps.
Convention: [x] done, [~] in progress / running, [ ] not started. Tags: (measured) (derived) (hypothesis).

## Done
- [x] Pipeline: torch -> torch-mlir -> conv_rewrite -> bufferize -> external AME matmul calls -> RV64 -> Spike (AME and scalar backends)
- [x] ResNet18 (64, 224) AME + scalar, current baseline (counters off, O3, run-based memrefCopy): 34.64x whole forward at 224
- [x] ViT-B/16: 2-layer AND 12-layer, AME + scalar, current pipeline (38.10x / 38.13x whole forward, 62.74x / 62.79x matmul); old 12-layer number (4953.64M) retired
- [x] Whisper-tiny encoder (AME only) and decoder step (AME + scalar, 6.9x)
- [x] Moonshine-tiny encoder 10 s: AME + scalar (17.16x whole, 63.31x matmul), PASS in both
- [x] Weights-as-function-arguments path (verified on the Whisper-tiny decoder step)
- [x] fp16 investigation: Spike has mfma_hf_mm but the accumulator is fp16 (stalls at 2048, measured); tile 8x8x8; needs zfh in ISA. Decision: stay on fp32 (STATE.md "fp16 on AME")
- [x] Compared with the teammate's study: precision (fp16 shapes vs fp32 here), MAC cross-check (ViT matches, Moonshine 1.5% lower)
- [x] Results page in the teammate's style (AME_Measured_Results.html)

- [x] Whisper-small: encoder 30 s (AME, 53.61G, 64.4% non-matmul), encoder 3 s (AME + scalar, 2 runs, 38.98x warm / 37.07x cold), decoder step (AME, 176.88M, PASS); MACs match the teammate's 172.0812 G exactly

## In progress
- (nothing running as of the last results; check `pgrep -a spike`)

## Next: easy (hours)
- [ ] Makefile: stamp USE_AME / LLVM_OPT / ISA into $(B)/.flags and depend on it; make resnet18_llvm.mlir depend on mlir/conv_rewrite.py (stale .o gave wrong-backend risk)
- [ ] tools/record.sh: run a build and append one JSON line (model, dir, backend, flags, git commit, RUN0/RUN1 counters, STATS, PASS) to results.jsonl; generate the results page DATA from it (no hand copying)
- [ ] Scalar runs still missing: Whisper-tiny encoder; Whisper-small decoder step (cheap, ~1.2G instr; the 3 s encoder is done, 38.98x warm); Whisper-small encoder 30 s is NOT feasible (~1.2T instr, ~5 h), report as extrapolated from 7.0 instr/MAC; Moonshine-base when exported
- [ ] Explain the Whisper-small encoder cold/warm gap (3.47G = 6.5% at 30 s, 129.65M = 5.3% at 3 s; all in non-matmul and identical in AME and scalar). The attention-score first-touch guess is doubtful: the score buffer is 100x smaller at 3 s (1.08 MB vs 108 MB) but the gap is only 26.8x smaller; other first-touch buffers or one-time init are more likely. Separate it with a PC histogram of RUN0 vs RUN1 (tools/pchist.py) and decide whether results should always use RUN1 (RUNS=2); the 3 s encoder was re-run with 2 runs: cold gap 129.65M = 5.3%, same in AME and scalar
- [ ] Re-run all models on ONE pipeline version (same commit, O3, counters off): ViT 2- and 12-layer are done on the current pipeline; still to confirm for ResNet18, Whisper-tiny and Moonshine rows (record.sh will make this a one-liner); the old ViT 12-layer 4953.64M number is retired
- [ ] Update the results page with the new scalar rows (Moonshine, ViT 2-layer, ViT 12-layer) and the 17.16x / 38.10x ratios
- [ ] Investigate Moonshine MACs: ours 4.336G vs the teammate's 4.4026G (1.5%); find which ops the op-recorder counts that the matmul counter does not
- [ ] Label everything: "scalar = plain C i-k-j loop, no blocking", "PASS = sampled check", "fp32 random weights, reduced configs" on the page and in STATE.md

## Next: medium (a day each)
- [ ] Profile the non-matmul share (39-74% of AME forward): run tools/pchist.py on Moonshine-tiny and Whisper-tiny encoder; predict from shapes (expf/erff counts x ~40 instr, loop element counts) and compare to the histogram, as done for ViT
- [ ] Moonshine in_copy 88M (50 copies) vs 33M for the larger Whisper-small encoder: log shape and element count of every memrefCopy call
- [ ] Fairer scalar baseline: (a) register-blocked scalar matmul, (b) RVV matmul (vfmacc; the Spike ISA already has v), compile the non-matmul code identically for each; report ratios against all baselines
- [ ] Exact correctness mode in compare_tensor.py (all elements) for at least Moonshine-tiny and one ViT size
- [ ] Sensitivity analysis instead of a cycle claim: model cycles = non-matmul instr x CPI + K-steps x mfma latency, plot whole-forward ratio vs mfma latency (1..16)
- [ ] Check layer extrapolation: ViT 2 and 12 layers are consistent (38.10x vs 38.13x, per-layer MACs match the hand count); add a 4-layer point to test linearity (supports the reduced-layer LLM plan, still a hypothesis for LLMs)
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
