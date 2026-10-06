/* Public entry points: count calls, then dispatch to AME or the reference. */
#include "ame_kernel.h"

#ifdef __riscv
static inline unsigned long rd_instret(void) { unsigned long v; __asm__ volatile("rdinstret %0" : "=r"(v)); return v; }
#else
static inline unsigned long rd_instret(void) { return 0; }
#endif
unsigned long ame_instret_in_matmul, ame_ksteps, ame_otiles;

static ame_stats_t g_stats;

const ame_stats_t *ame_stats_get(void) { return &g_stats; }
void ame_stats_reset(void) { g_stats.calls = 0; g_stats.macs = 0; }

const char *ame_backend_name(void) {
#ifdef USE_AME
  return "AME";
#else
  return "reference-scalar";
#endif
}

void ame_matmul_f32(const float *A, const float *B, float *C,
                    int64_t M, int64_t N, int64_t K,
                    int64_t lda, int64_t ldb, int64_t ldc, int accumulate) {
  g_stats.calls++;
  g_stats.macs += (uint64_t)M * (uint64_t)N * (uint64_t)K;
#ifdef USE_AME
  { unsigned long t0 = rd_instret();
    ame_hw_matmul_f32(A, B, C, M, N, K, lda, ldb, ldc, accumulate);
    ame_instret_in_matmul += rd_instret() - t0; }
#else
  { unsigned long t0 = rd_instret();
    ref_matmul_f32(A, B, C, M, N, K, lda, ldb, ldc, accumulate);
    ame_instret_in_matmul += rd_instret() - t0; }
#endif
}

void ame_matmul_i8(const int8_t *A, const int8_t *B, int32_t *C,
                   int64_t M, int64_t N, int64_t K,
                   int64_t lda, int64_t ldb, int64_t ldc, int accumulate) {
  g_stats.calls++;
  g_stats.macs += (uint64_t)M * (uint64_t)N * (uint64_t)K;
#ifdef USE_AME
  ame_hw_matmul_i8(A, B, C, M, N, K, lda, ldb, ldc, accumulate);
#else
  ref_matmul_i8(A, B, C, M, N, K, lda, ldb, ldc, accumulate);
#endif
}
