/* Phase A.5: check the memref-descriptor shims (the ABI MLIR will call) on the
 * host, including a strided sub-view, without needing MLIR at all. */
#include <stdio.h>
#include <stdint.h>
#include "ame_kernel.h"
#include "memref_shims.h"

AME_SHIM(ame_mm_f32_test, ame_matmul_f32_memref)
AME_SHIM(ame_mm_i8_test,  ame_matmul_i8_memref)

int main(void) {
  /* 6x10 parent buffers; use the 4x5 / 5x3 / 4x3 sub-views starting at (1,2). */
  static float A[6 * 10], B[6 * 10], C[6 * 10], R[4 * 3];
  for (int i = 0; i < 60; i++) { A[i] = (float)((i * 7) % 11) - 5; B[i] = (float)((i * 3) % 13) - 6; C[i] = 1.0f; }
  int64_t off = 1 * 10 + 2;
  /* reference: C_sub += A_sub * B_sub */
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 3; j++) {
      float s = C[off + i * 10 + j];
      for (int k = 0; k < 5; k++) s += A[off + i * 10 + k] * B[off + k * 10 + j];
      R[i * 3 + j] = s;
    }
  ame_mm_f32_test(A, A, off, 4, 5, 10, 1,
                  B, B, off, 5, 3, 10, 1,
                  C, C, off, 4, 3, 10, 1);
  int bad = 0;
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 3; j++)
      if (C[off + i * 10 + j] != R[i * 3 + j]) bad++;

  /* untouched elements outside the sub-view must be unchanged */
  if (C[0] != 1.0f || C[off + 3] != 1.0f) bad++;

  const ame_stats_t *s = ame_stats_get();
  printf("shim calls=%lu\n", (unsigned long)s->calls);
  printf(bad ? "FAIL shim\n" : "PASS shim\n");
  return bad ? 1 : 0;
}
