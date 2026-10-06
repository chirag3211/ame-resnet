// Phase B input: a single matmul already on memrefs (no frontend needed).
// C[8x12] += A[8x16] * B[16x12]
func.func @matmul(%A: memref<8x16xf32>, %B: memref<16x12xf32>, %C: memref<8x12xf32>)
    attributes {llvm.emit_c_interface} {
  linalg.matmul ins(%A, %B : memref<8x16xf32>, memref<16x12xf32>)
                outs(%C : memref<8x12xf32>)
  return
}
