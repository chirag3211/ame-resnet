# ame-resnet: PyTorch models -> torch-mlir -> MLIR -> RV64 -> Spike (RISC-V AME matrix extension)

Runs ResNet18 and ViT-B/16 (fp32, seeded random weights) end to end on the riscv-stc fork of Spike
(branch `matrix`, commit 6add0673, matrix spec v0.5), using real matrix instructions for
every convolution, Linear layer and attention matmul. Correctness is checked against PyTorch logits.
Metric is **retired instructions (rdinstret) on Spike, not cycles**: no hardware-speedup claim.
Current measurements, environment and open items live in `STATE.md`.

## Pipeline
```
python/export_resnet18.py        torch -> torch-mlir (linalg-on-tensors) -> build/resnet18.mlir
python/export_vit.py L           ViT with L layers -> build_vitLp/vit.mlir (pre-transposed weights; --orig = plain torchvision -> build_vitL/)
                                 + build/input.bin, build/golden_logits.bin (PyTorch reference)
mlir/lower_resnet.sh
  [1] mlir/conv_rewrite.py       conv_2d_nchw_fchw -> 5D gather generic (no div/mod) + collapse
                                 + linalg.matmul + expand   (CONV_REWRITE=0 selects the old
                                 transform-dialect img2col in mlir/conv_to_matmul.mlir)
                                 then fold-unit-extent-dims -> 21 rank-2 linalg.matmul
  [2] one-shot-bufferize         tensors -> memrefs
  [3] mlir/matmul_to_call.py     each linalg.matmul -> call @ame_mm_f32_<n>(A,B,C);
                                 each linalg.batch_matmul -> call @ame_bmm_f32_<n> (loops over the batch)
      mlir/add_ciface.py         emit_c_interface on @forward
  [4] lower to LLVM dialect      loops, affine, scf, math (erf -> libm erff), memref, func -> llvm
mlir-translate + llc             LLVM IR -> RV64 object (rv64gc, lp64d)
mlir/gen_shims.py                generates the C shims ame_mm_f32_<n> -> ame_matmul_f32
src/main_resnet.c, src/memref_copy.c, src/ame_dispatch.c, src/ame_hw.c, src/ref_matmul.c  -> ELF
spike --isa=... pk build/resnet18.elf   -> logits + instruction counters
python/compare.py                max abs err and TOP1 vs PyTorch
```

## Backends (selected at build time)
| | scalar reference | AME |
|---|---|---|
| make | `make SPIKE_ISA=rv64gc_zicntr resnet-spike SIZE=N` | `make USE_AME=1 SPIKE=$HOME/riscv-stc/bin/spike SPIKE_ISA=rv64imafdcv_zicntr_matrix resnet-spike SIZE=N` |
| matmul | `src/ref_matmul.c` | `src/ame_hw.c` via `include/ame_insn.h` (`.insn r`, opcode 0x77) |

AME fp32 kernel: msettile m/k/n (granted 8/4/4), C accumulator in acc0 across the K loop
(zeroed by a stride-0 load of a zero row), one `mfma.f.mm` per K step. int8 entry points exist
but still use the scalar reference. Only the matmuls change between backends; all other code
(gather, padding, ReLU, pooling, memrefCopy) is identical.

**Switching backend:** `rm -f build/resnet18.elf build/*.o` first (the Makefile does not track `USE_AME`;
it also does not list `mlir/conv_rewrite.py` as a dependency of `resnet18_llvm.mlir`).
`rdinstret` needs `zicntr` in the ISA string.

## Running another model (e.g. ViT)
```
python3 python/export_vit.py 12                     # -> build_vit12p; or 2 for a quick 2-layer check
ln -sf vit.mlir build_vit12p/resnet18.mlir          # the Makefile still names things resnet18.*
make B=build_vit12p USE_AME=1 SPIKE=$HOME/riscv-stc/bin/spike SPIKE_ISA=rv64imafdcv_zicntr_matrix resnet-spike SIZE=224
```
`B=<dir>` keeps each model's intermediates separate. `main_resnet.c` is reused for any model with a
(1x3x224x224 f32) -> (1x1000 f32) `forward`. SIZE must match the exported input.
Full ViT-B/16: ~661 MB MLIR (weights inlined), ~44 min for the whole `make` (hypothesis: mostly toolchain, not Spike).

## Generic models (Whisper etc.)
`python/export_whisper.py`, `python/ame_export.py` (shared: pre-transposed Linear, export), `python/gen_driver.py` (driver for a single-input/single-output model),
`python/compare_tensor.py` (sampled + checksum comparison). Build with `make B=<dir> DRIVER=<dir>/main_gen.c COMPARE=python/compare_tensor.py ...` (see STATE.md).
`conv_rewrite.py` handles conv_2d_nchw_fchw and conv_1d_ncw_fcw.

## Measuring
`main_resnet.c` runs `forward` twice. RUN0 is cold (includes ~65M instr of one-time pk first-touch
at 64x64) and RUN1 is warm; **report RUN1 only**. It prints forward / in_matmul / in_copy
(memrefCopy) counts; "everything else" = forward - in_matmul - in_copy.

## Layout
- `python/` exporters (ResNet18, ViT) and logit comparison
- `mlir/` lowering script and text-level rewrites (`conv_rewrite.py`, `matmul_to_call.py` incl. batch_matmul, `add_ciface.py`, `gen_shims.py`)
- `src/`, `include/` runtime: kernel API, dispatch + counters, AME backend, scalar reference, memrefCopy, memref shims
- `tests/` kernel API tests (native and Spike), single MLIR matmul test, and AME probes (`insn_cost.c`, `ame_tile.c`, `cold_warm.c`, `insn_probe.c`)
- `tools/pchist.py` PC histogram -> hot functions/blocks; `tools/dasm_check.py` encoding check
- `STATE.md` environment, verified numbers, open items (source of truth for results)

## Phase tests (older bring-up path)
`make test-native`, `make test-spike`, `make mlir-matmul-spike` (single matmul through MLIR), then ResNet18.

## Known limits
`conv_rewrite.py`: N=1, f32, no dilation, no groups, conv_2d_nchw_fchw only. int8 still scalar. Weight transposes are not folded by MLIR (weights are opaque resources), so ViT exports pre-transpose them in Python.
