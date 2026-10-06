#!/usr/bin/env bash
# FIRST DRAFT (untested) lowering pipeline for ResNet18:
#   linalg-on-tensors -> conv->img2col+matmul -> bufferize -> matmul->calls -> LLVM
# Expect to iterate: pass names/options differ across LLVM versions. Run it step
# by step and paste any error to Claude.
set -euo pipefail
IN=${1:-build/resnet18.mlir}
OUT=${2:-build/resnet18_llvm.mlir}
OPT=${MLIR_OPT:-mlir-opt}

echo "[1/4] conv -> gather + matmul"
if [ "${CONV_REWRITE:-1}" = 1 ]; then
  python3 mlir/conv_rewrite.py "$IN" build/resnet18_rw.mlir
  $OPT build/resnet18_rw.mlir --canonicalize \
    --linalg-fold-unit-extent-dims --canonicalize \
    --linalg-specialize-generic-ops -o build/r18_1_matmul.mlir
else
  $OPT "$IN" \
    --transform-preload-library="transform-library-paths=mlir/conv_to_matmul.mlir" \
    --transform-interpreter \
    --canonicalize \
    --linalg-fold-unit-extent-dims --canonicalize \
    --linalg-specialize-generic-ops \
    -o build/r18_1_matmul.mlir
fi

echo "[2/4] bufferize"
$OPT build/r18_1_matmul.mlir \
  --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
  --buffer-deallocation-pipeline \
  --convert-bufferization-to-memref \
  --canonicalize \
  -o build/r18_2_buf.mlir

echo "[3/4] linalg.matmul -> calls into ame_matmul"
python3 mlir/matmul_to_call.py build/r18_2_buf.mlir build/r18_3_calls.mlir
python3 mlir/add_ciface.py build/r18_3_calls.mlir

echo "[4/4] everything else -> LLVM dialect"
$OPT build/r18_3_calls.mlir \
  --convert-linalg-to-loops --expand-strided-metadata --lower-affine \
  --convert-scf-to-cf --convert-math-to-llvm --convert-arith-to-llvm --convert-index-to-llvm --convert-cf-to-llvm \
  --finalize-memref-to-llvm --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  -o "$OUT"
echo "wrote $OUT"
