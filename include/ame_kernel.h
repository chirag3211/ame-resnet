/* Tile-GEMM kernel API used by the MLIR-generated code.
 * Row-major:  C[MxN] (+)= A[MxK] * B[KxN]
 * lda/ldb/ldc are leading dimensions in ELEMENTS. accumulate=0 overwrites C. */
#ifndef AME_KERNEL_H
#define AME_KERNEL_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void ame_matmul_f32(const float *A, const float *B, float *C,
                    int64_t M, int64_t N, int64_t K,
                    int64_t lda, int64_t ldb, int64_t ldc, int accumulate);

/* int8 inputs, int32 accumulate (the natural fit for matrix extensions). */
void ame_matmul_i8(const int8_t *A, const int8_t *B, int32_t *C,
                   int64_t M, int64_t N, int64_t K,
                   int64_t lda, int64_t ldb, int64_t ldc, int accumulate);

/* Call counters: confirm the matmuls really go through this API. */
typedef struct { uint64_t calls; uint64_t macs; } ame_stats_t;
const ame_stats_t *ame_stats_get(void);
void ame_stats_reset(void);
const char *ame_backend_name(void);

/* Reference (scalar) implementations, always available. */
void ref_matmul_f32(const float *A, const float *B, float *C,
                    int64_t M, int64_t N, int64_t K,
                    int64_t lda, int64_t ldb, int64_t ldc, int accumulate);
void ref_matmul_i8(const int8_t *A, const int8_t *B, int32_t *C,
                   int64_t M, int64_t N, int64_t K,
                   int64_t lda, int64_t ldb, int64_t ldc, int accumulate);

#ifdef USE_AME
/* Implemented in src/ame_hw.c using AME instructions. */
void ame_hw_matmul_f32(const float *A, const float *B, float *C,
                       int64_t M, int64_t N, int64_t K,
                       int64_t lda, int64_t ldb, int64_t ldc, int accumulate);
void ame_hw_matmul_i8(const int8_t *A, const int8_t *B, int32_t *C,
                      int64_t M, int64_t N, int64_t K,
                      int64_t lda, int64_t ldb, int64_t ldc, int accumulate);
#endif

#ifdef __cplusplus
}
#endif
#endif
