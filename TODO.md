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

- [x] Record tooling run for real (Oct 8): tools/record.sh + record.py + results_table.py, 13 real runs (fill.sh, 4,973 s), all PASS, every re-run reproduced the earlier counts to the instruction; results.jsonl has 19 rows (13 real + 6 backfilled). Fixed after review: backfilled rows no longer carry a fake git commit / dirty flag / timestamp (record.py --backfill writes null; results_table.py says 'commit unknown'); corrected 66.85x / 7.527 / 73.8% / 14.8% in STATE.md; Whisper-tiny decoder numbers labelled inline vs weights-as-args
- [x] Scalar baselines: Whisper-tiny encoder (17.28x), Whisper-small decoder step (6.80x), both PASS with errors identical to AME; non-matmul work identical in both builds; whole-forward ratio = n + R(1-n) reproduces every pair (STATE.md summary)

## In progress
- (nothing running as of the last results; check `pgrep -a spike`)

## Next: easy (hours)
- [x] Makefile flag stamps: confirmed analytically (Oct 8). `resnet18.o` depends on `$(B)/.flags_cg`
  which tracks only `LLVM_OPT / LLVM_OPT_BIN / LLC / ARCH / ABI`; `USE_AME` is NOT in
  `CG_FLAGS`, so toggling it changes only `.flags_link`, which triggers a C recompile + relink
  but NOT opt or llc. Verified by the 13 real builds: wall times of first-build scalar runs
  were 26-99 s (consistent with C-only work, not a re-run of the slow llc step).
- [ ] Scalar runs: all done except Whisper-small encoder 30 s (infeasible, ~5 h per run; stays extrapolated at ~23x, 23.0-23.2 depending on R). Moonshine-base when exported
- [x] Explain the cold/warm gap. CONFIRMED via pchist.py on Moonshine-tiny AME (1-run cold vs 2-run combined):
  99.1% of the 218M gap is in the unmapped (pk/kernel) region = OS page-fault handler on first-touch of
  inline weight constants. Every user-space symbol (forward, ame_hw, expf, erff, ...) shows a ratio within
  0.4% of 2.0× between the two runs — no computation-level difference at all. The --weights-as-args path
  eliminates the gap by moving first-touch into fread() before the rdinstret window. See STATE.md.
  The Whisper-small 30 s gap (3.47G) and 3 s gap (129.65M) are both non-matmul and non-copy (same in both
  backends), consistent with the same cause at different weight-set sizes.
- [~] Re-run all models on ONE pipeline version (same commit, O3, counters off): re-run and reproduced on Oct 8: ResNet18 64/224, ViT 2-layer, Moonshine, both decoder steps, Whisper-tiny encoder AME. Still backfilled (commit unknown, so results_table.py warns): vit12p (ame+scalar), Whisper-small 3 s (ame+scalar), Whisper-tiny encoder scalar, Whisper-small 30 s (ame). Re-run them with tools/record.sh if a recorded commit is wanted (vit12p scalar ~125G instr, Whisper-small 3 s scalar ~95G; hours at ~70M instr/s); the 30 s encoder AME run is long on matrix Spike. The old ViT 12-layer 4953.64M number is retired
- [ ] Regenerate the results page from results.jsonl (unblocked: `python3 tools/results_table.py --json`; the page was updated by hand on Oct 8 with every row measured so far: ViT 2- and 12-layer, Moonshine, Whisper-tiny encoder 17.28x, Whisper-small 3 s 38.98x and decoder step 6.80x, Whisper-small 30 s AME only)
- [x] ResNet18 64 scalar matmul costs 7.527 instr/MAC vs 7.0-7.04 everywhere else. EXPLAINED (see STATE.md
  'ResNet18-64 scalar overhead'): at SIZE=64, 45.2% of MACs are in shapes with small N (spatial positions < 64),
  which cost ~8.2 instr/MAC due to j-loop overhead in the i-k-j scalar kernel. At SIZE=224 only 22.7% of MACs
  are in small-N shapes, so the blended rate is close to 7.0. No fix needed (the scalar kernel is intentionally
  simple); the 66.85x matmul ratio at 64 reflects the higher scalar cost, not an AME regression.
- [x] record.py tests: tests/test_record.py, 20 tests (11 parse unit + 9 subprocess integration).
  Covers: happy path AME/scalar, --backfill null provenance, --git-commit with --backfill, backend
  mismatch in both directions, FAIL log exits 1 but row is written, multiple appends, --wall.
  Run: `python3 tests/test_record.py` or `python3 -m pytest tests/test_record.py`.
- [~] Moonshine MACs: ours 4.336G vs the teammate's 4.4026G (1.5%, 66.1M unexplained).
  probe_moonshine_macs.py confirmed our STATS is correct after fixing a bug (missed aten.matmul.default,
  which is what nn.Linear uses for 3D inputs without bias — the 4 attention projections per layer).
  Fixed probe = STATS = 4,336,487,440. The teammate's 66.1M gap is not the projections (826M), not
  an attention layer (99M), not depthwise/rotary. Still open: compare the teammate's per-op table to
  isolate it. ViT matches near-exactly so the miss is Moonshine-specific.
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
