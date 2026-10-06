# ame-resnet: ResNet18 -> MLIR -> RISC-V Spike (with an AME matmul hook)

All matmuls go through one C API (`include/ame_kernel.h`). Today it is backed by a
scalar reference; later `src/ame_hw.c` swaps in AME instructions (`make USE_AME=1`).
Datatype for now: **fp32** (int8 entry points exist too). The AME datatype choice comes later.

## Phases (do them in order; each must pass before the next)

| Phase | What | Command | Needs |
|---|---|---|---|
| A  | kernel API vs exact reference, on host | `make test-native` | gcc only |
| A  | same test on Spike | `make test-spike` | riscv gcc, spike, pk |
| B  | one `linalg.matmul` through MLIR -> call into the API -> Spike | `make mlir-matmul-spike` | + mlir-opt, mlir-translate, llc |
| C  | ResNet18 | see below | + torch, torchvision, torch-mlir |

## Phase C
```bash
source env.sh
python3 python/export_resnet18.py --size 64        # start small! also writes golden logits
make resnet-spike SIZE=64                          # lower, build, run on Spike, compare to PyTorch
# once that passes: --size 224 and make resnet-spike SIZE=224
```
`export_resnet18.py --no-mlir` writes only the input and golden logits.

## Layout
- `src/ref_matmul.c`   scalar GEMM (reference + fallback)
- `src/ame_dispatch.c` entry points, call/MAC counters, backend switch
- `src/ame_hw.c`       AME backend SKELETON (stops the build until encodings are filled in)
- `src/memref_shims.c` + `include/memref_shims.h`  MLIR memref-descriptor ABI -> kernel API
- `mlir/matmul_to_call.py`  rewrites bufferized `linalg.matmul` into calls to `ame_mm_*`
- `mlir/gen_shims.py`  generates the `ame_mm_*` C shims, and checks the 21-argument ABI
- `mlir/lower_resnet.sh`, `mlir/conv_to_matmul.mlir`  Phase C lowering (first draft)
- `src/main_resnet.c`  Phase C driver (template)
- `python/export_resnet18.py`, `python/compare.py`

## Status of this code
Verified on x86 (host gcc): the kernel API tests (including mutation checks that the
tests catch bugs), the memref shims on a strided sub-view, `matmul_to_call.py` and
`gen_shims.py` on hand-made inputs, `compare.py`, and syntax of all scripts.

NOT yet run: anything involving riscv gcc / Spike / pk, mlir-opt / llc, torch /
torch-mlir. The pass pipeline in `mlir/lower_resnet.sh`, the transform script and
`main_resnet.c` are first drafts and will need iteration against your LLVM version.
Known gap: `linalg.batch_matmul` is not rewritten yet (a warning is printed).
