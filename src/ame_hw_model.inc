/* AME backend: tiled GEMM written against the tile primitives in ame_model.h.
 * Per output tile: zero/load acc, loop K-tiles (load A, load B, MAC), store acc. */
#include "ame_kernel.h"
#include "ame_model.h"

#define AME_MIN(a, b) ((a) < (b) ? (a) : (b))

#define AME_HW_MATMUL(SUF, TA, TB, TC, TK) \
  void ame_hw_matmul_##SUF(const TA *A, const TB *B, TC *C, int64_t M, int64_t N, \
                           int64_t K, int64_t lda, int64_t ldb, int64_t ldc, \
                           int accumulate) { \
    for (int64_t m0 = 0; m0 < M; m0 += AME_TM) { \
      int m = (int)AME_MIN((int64_t)AME_TM, M - m0); \
      for (int64_t n0 = 0; n0 < N; n0 += AME_TN) { \
        int n = (int)AME_MIN((int64_t)AME_TN, N - n0); \
        if (accumulate) ame_load_acc_##SUF(C + m0 * ldc + n0, ldc, m, n); \
        else            ame_zero_acc_##SUF(m, n); \
        for (int64_t k0 = 0; k0 < K; k0 += (TK)) { \
          int k = (int)AME_MIN((int64_t)(TK), K - k0); \
          ame_load_a_##SUF(A + m0 * lda + k0, lda, m, k); \
          ame_load_b_##SUF(B + k0 * ldb + n0, ldb, k, n); \
          ame_mac_##SUF(m, n, k); \
        } \
        ame_store_acc_##SUF(C + m0 * ldc + n0, ldc, m, n); \
      } \
    } \
  }

AME_HW_MATMUL(f32, float,  float,  float,   AME_TK_F32)
AME_HW_MATMUL(i8,  int8_t, int8_t, int32_t, AME_TK_I8)
