/* Scalar reference GEMM (i-k-j order: unit-stride inner loop). */
#include "ame_kernel.h"

void ref_matmul_f32(const float *A, const float *B, float *C,
                    int64_t M, int64_t N, int64_t K,
                    int64_t lda, int64_t ldb, int64_t ldc, int accumulate) {
  for (int64_t i = 0; i < M; i++) {
    float *c = C + i * ldc;
    if (!accumulate)
      for (int64_t j = 0; j < N; j++) c[j] = 0.0f;
    for (int64_t k = 0; k < K; k++) {
      const float a = A[i * lda + k];
      const float *b = B + k * ldb;
      for (int64_t j = 0; j < N; j++) c[j] += a * b[j];
    }
  }
}

void ref_matmul_i8(const int8_t *A, const int8_t *B, int32_t *C,
                   int64_t M, int64_t N, int64_t K,
                   int64_t lda, int64_t ldb, int64_t ldc, int accumulate) {
  for (int64_t i = 0; i < M; i++) {
    int32_t *c = C + i * ldc;
    if (!accumulate)
      for (int64_t j = 0; j < N; j++) c[j] = 0;
    for (int64_t k = 0; k < K; k++) {
      const int32_t a = A[i * lda + k];
      const int8_t *b = B + k * ldb;
      for (int64_t j = 0; j < N; j++) c[j] += a * (int32_t)b[j];
    }
  }
}
