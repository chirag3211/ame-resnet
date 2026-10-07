/* linalg.matmul semantics: C[M,N] += A[M,K] * B[K,N]  (accumulate). */
#include <stdio.h>
#include <stdlib.h>
#include "ame_kernel.h"
#include "memref_shims.h"

static void check(const char *who, int64_t st1, int64_t size1) {
  /* kernel needs unit stride along the innermost dim (unless that dim has 1 element) */
  if (size1 > 1 && st1 != 1) {
    printf("%s: unsupported memref layout (inner stride %ld != 1)\n", who, (long)st1);
    exit(2);
  }
}

#define DEFINE_IMPL(FN, KERNEL, TA, TB, TC)                                      \
  void FN(MR2_PARAMS(a), MR2_PARAMS(b), MR2_PARAMS(c)) {                         \
    (void)a_alloc; (void)b_alloc; (void)c_alloc;                                 \
    const int64_t M = a_s0, K = a_s1, N = b_s1;                                  \
    if (b_s0 != K || c_s0 != M || c_s1 != N) {                                   \
      printf(#FN ": shape mismatch\n"); exit(2);                                 \
    }                                                                            \
    check(#FN " A", a_st1, a_s1); check(#FN " B", b_st1, b_s1);                  \
    check(#FN " C", c_st1, c_s1);                                                \
    KERNEL((const TA *)a_align + a_off, (const TB *)b_align + b_off,             \
           (TC *)c_align + c_off, M, N, K, a_st0, b_st0, c_st0, 1);              \
  }

DEFINE_IMPL(ame_matmul_f32_memref, ame_matmul_f32, float, float, float)
DEFINE_IMPL(ame_matmul_i8_memref, ame_matmul_i8, int8_t, int8_t, int32_t)

/* linalg.batch_matmul: C[b] += A[b] * B[b], batch loop over the leading dim. */
void ame_bmm_f32_memref(MR3_PARAMS(a), MR3_PARAMS(b), MR3_PARAMS(c)) {
  (void)a_alloc; (void)b_alloc; (void)c_alloc;
  const int64_t BT = a_s0, M = a_s1, K = a_s2, N = b_s2;
  if (b_s0 != BT || c_s0 != BT || b_s1 != K || c_s1 != M || c_s2 != N) {
    printf("ame_bmm_f32_memref: shape mismatch\n"); exit(2);
  }
  check("bmm A", a_st2, a_s2); check("bmm B", b_st2, b_s2); check("bmm C", c_st2, c_s2);
  for (int64_t i = 0; i < BT; i++)
    ame_matmul_f32((const float *)a_align + a_off + i * a_st0,
                   (const float *)b_align + b_off + i * b_st0,
                   (float *)c_align + c_off + i * c_st0, M, N, K, a_st1, b_st1, c_st1, 1);
}
