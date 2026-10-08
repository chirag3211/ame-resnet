# TODO (Oct 8 2026). Numbers and details live in STATE.md; this file is status + next steps.
Convention: [x] done, [~] in progress / running, [ ] not started. Tags: (measured) (derived) (hypothesis).

## Done
- [x] Pipeline: torch -> torch-mlir -> conv_rewrite -> bufferize -> external AME matmul calls -> RV64 -> Spike (AME and scalar backends)
- [x] ResNet18 (64, 224) AME + scalar, current baseline (counters off, O3, run-based memrefCopy): 34.64x whole forward at 224
- [x] ViT-B/16: 2-layer AND 12-layer, AME + scalar, current pipeline (38.10x / 38.13x whole forward, 62.74x / 62.79x matmul); old 12-layer number (4953.64M) retired
- [x] Whisper-tiny encoder (AME + scalar, 17.28x) and decoder step (AME + scalar, 6.9x)
- [x] Moonshine-tiny encoder 10 s: AME + scalar (17.16x whole, 63.31x matmul), PASS in both
- [x] Weights-as-function-arguments path (verified on the Whisper-tiny decoder step)
- [x] fp16 investigation: Spike has mfma_hf_mm but the accumulator is fp16 (stalls at 2048, measured); tile 8x8x8; needs zfh in ISA. Decision: stay on fp32 (STATE.md "fp16 on AME")
- [x] Compared with the teammate's study: precision (fp16 shapes vs fp32 here), MAC cross-check (ViT matches, Moonshine 1.5% lower)
- [x] Results page in the teammate's style (AME_Measured_Results.html)

- [x] Whisper-small: encoder 30 s (AME, 53.61G, 64.5% non-matmul incl. copies), encoder 3 s (AME + scalar, 2 runs, 38.98x warm / 37.07x cold), decoder step (AME + scalar, 6.80x, PASS); MACs match the teammate's 172.0812 G exactly

- [x] Scalar baselines: Whisper-tiny encoder (17.28x), Whisper-small decoder step (6.80x), both PASS with errors identical to AME; non-matmul work identical in both builds; whole-forward ratio = n + R(1-n) reproduces every pair (STATE.md summary)

## In progress
- (nothing running as of the last results; check `pgrep -a spike`)

## Next: easy (hours)
- [~] Makefile flag stamps (.flags_cg for opt/llc, .flags_link for the C build) + deps on mlir/*.py and include/*.h: written and tested with a stub toolchain (no-op rebuild does nothing; USE_AME toggle relinks only; LLVM_OPT change re-runs opt+llc). Still to do: one real build to confirm, e.g. `make B=build_vit2p ... USE_AME=1` after a scalar build and check that llc is not re-run and the instruction counts match the table (562.19M)
- [~] tools/record.sh + tools/record.py + tools/results_table.py: written and tested end to end with a stub spike (records one JSON line; the table reproduces the Whisper-small 3 s row). Still to do: run it for real on one small model (e.g. `tools/record.sh build_vit2p ame` and `scalar`), back-fill the history from the existing logs with `python3 tools/record.py <log> --dir <dir> --backend ame|scalar --set llvm_opt=O3` (note: old logs have no wall time / git commit), then feed `results_table.py --json` to the results page generator
- [ ] Scalar runs: all done except Whisper-small encoder 30 s (infeasible, ~5 h per run; stays extrapolated at ~23x, 23.0-23.2 depending on R). Moonshine-base when exported
- [ ] Explain the Whisper-small encoder cold/warm gap (3.47G = 6.5% at 30 s, 129.65M = 5.3% at 3 s; all in non-matmul and identical in AME and scalar). The attention-score first-touch guess is doubtful: the score buffer is 100x smaller at 3 s (1.08 MB vs 108 MB) but the gap is only 26.8x smaller; other first-touch buffers or one-time init are more likely. Separate it with a PC histogram of RUN0 vs RUN1 (tools/pchist.py) and decide whether results should always use RUN1 (RUNS=2); the 3 s encoder was re-run with 2 runs: cold gap 129.65M = 5.3%, same in AME and scalar
- [ ] Re-run all models on ONE pipeline version (same commit, O3, counters off): ViT 2- and 12-layer are done on the current pipeline; still to confirm for ResNet18, Whisper-tiny and Moonshine rows (record.sh will make this a one-liner); the old ViT 12-layer 4953.64M number is retired
- [ ] Regenerate the results page from results.jsonl once record.sh exists (the page was updated by hand on Oct 8 with every row measured so far: ViT 2- and 12-layer, Moonshine, Whisper-tiny encoder 17.28x, Whisper-small 3 s 38.98x and decoder step 6.80x, Whisper-small 30 s AME only)
- [ ] Investigate Moonshine MACs: ours 4.336G vs the teammate's 4.4026G (1.5%); find which ops the op-recorder counts that the matmul counter does not
- [ ] Label everything: "scalar = plain C i-k-j loop, no blocking", "PASS = sampled check", "fp32 random weights, reduced configs" on the page and in STATE.md

## Next: medium (a day each)
- [ ] Profile the non-matmul share (39-74% of AME forward): run tools/pchist.py on Moonshine-tiny and Whisper-tiny encoder; predict from shapes (expf/erff counts x ~40 instr, loop element counts) and compare to the histogram, as done for ViT
- [ ] Moonshine in_copy 88M (50 copies) vs 33M for the larger Whisper-small encoder: log shape and element count of every memrefCopy call
- [ ] Fairer scalar baseline: (a) register-blocked scalar matmul, (b) RVV matmul (vfmacc; the Spike ISA already has v), compile the non-matmul code identically for each; report ratios against all baselines
- [ ] Exact correctness mode in compare_tensor.py (all elements) for at least Moonshine-tiny and one ViT size
- [ ] Sensitivity analysis instead of a cycle claim: model cycles = non-matmul instr x CPI + K-steps x mfma latency, plot whole-forward ratio vs mfma latency (1..16)
- [ ] Check layer extrapolation: ViT 2 and 12 layers are consistent (38.10x vs 38.13x, per-layer MACs match the hand count); add a 4-layer point to test linearity (supports the reduced-layer LLM plan, still a hypothesis for LLMs)
- [x] fp16 error test run on Spike (Oct 8): errors identical to the host emulation; measured K-step 14, per-chunk flush F = 673 per 8x8 tile (+ ~1834 per tile harness overhead in the printed instr/MAC); kernel-only cost 14/512 + 673/(64 x chunk), break-even with fp32 (0.111) at chunk 128 (rel RMS ~1.7e-3 signed). Details in STATE.md fp16 section
- [ ] fp16 next: (a) RVV flush experiment (hypothesis: ~45 instr per tile, 0.038 instr/MAC at chunk 64); (b) a relative-tolerance PASS criterion (~1e-3) for any fp16 run; (c) real activations instead of random data (dump a Moonshine FFN input); (d) make the test report kernel-only instr/MAC (exclude the acc[] zero-init and copy-out); (e) check why the local toolchain's flush loop is ~19% heavier than GCC 13.2's
- [ ] Re-baseline ResNet18 and ViT with tools/bench.sh after the Makefile/record changes

## Next: hard (open-ended)
- [ ] Coverage: conv_rewrite.py for depthwise/grouped/dilated/transposed conv and N>1 (needed for Kokoro, Canary, Mamba); LSTM and FFT or exclude and say so; weights-as-args for everything >= Whisper-small; reduced-layer random-weight configs for the 3B+ LLMs (Phi-4-mini, Phi-4, Ministral 3, Gemma 3n); Canary-Qwen borderline at 20 GB
- [ ] Reduce the non-matmul share once profiled: fused softmax, elementwise fusion before bufferization, vectorized elementwise loops, custom erf/exp kernels (changes numerics slightly)
- [ ] Matmul tuning: hoist msettilek, multi-accumulator (matmul is 26-61% of AME forward for encoders and full models, 83-85% for decode steps)
- [ ] Decode steps (M=1): matrix-vector role swap (weights on the 8-row side) might double the best case; spec not checked
- [ ] int8 (mqma.b.mm) with a quantized model; needed to fit the large LLMs
- [ ] fp16 port ONLY if the error test passes: f16 export, f16 paths in conv_rewrite.py / matmul_to_call.py / gen_shims.py / memref_shims.c / drivers / compare, scalar fp16 baseline (zfh in gcc and Spike), upcasts for erff/expf. Treat as a separate labelled experiment on one model (Moonshine-tiny), not a re-run of the study
- [ ] Real timing: needs a cycle-accurate model or RTL for AME; until then use the sensitivity analysis above, never a hardware speedup claim
- [ ] Spike speed (matrix Spike ~14M instr/s vs ~70M scalar, hypothesis): for big models use RUNS=1 with weights-as-args and state the cold/warm error bound
