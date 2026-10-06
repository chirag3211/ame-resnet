/* AME backend. On RISC-V it uses the real matrix instructions (riscv-stc Spike fork,
 * spec v0.5) through include/ame_insn.h. On other hosts it falls back to the software
 * tile model, so `make USE_AME=1 test-native` still exercises the tiling logic.
 * fp32: real instructions. int8: scalar reference for now (not yet ported). */
#include "ame_kernel.h"

#ifdef __riscv
#include "ame_insn.h"
extern unsigned long ame_ksteps, ame_otiles;

/* a row of zeros; loaded with stride 0 to clear acc0 (no clear instruction exists) */
static const float ame_zero_row[256] __attribute__((aligned(64)));

void ame_hw_matmul_f32(const float *A, const float *B, float *C,
                       int64_t M, int64_t N, int64_t K,
                       int64_t lda, int64_t ldb, int64_t ldc, int accumulate) {
  if (M <= 0 || N <= 0 || K <= 0) return;
  ame_set_sew32();
  ame_set_fp32();
  for (int64_t m0 = 0; m0 < M; ) {
    int64_t tm = (int64_t)ame_settile_m((uint64_t)(M - m0));
    for (int64_t n0 = 0; n0 < N; ) {
      int64_t tn = (int64_t)ame_settile_n((uint64_t)(N - n0));
      if (accumulate) ame_ld_c32(C + m0 * ldc + n0, (uint64_t)(ldc * 4));
      else            ame_ld_c32(ame_zero_row, 0);
      for (int64_t k0 = 0; k0 < K; ) {
        int64_t tk = (int64_t)ame_settile_k((uint64_t)(K - k0));
        ame_ld_a32(A + m0 * lda + k0, (uint64_t)(lda * 4));
        ame_ld_b32(B + k0 * ldb + n0, (uint64_t)(ldb * 4));
        ame_mfma_f();
        ame_ksteps++;
        k0 += tk;
      }
      ame_st_c32(C + m0 * ldc + n0, (uint64_t)(ldc * 4));
      ame_otiles++;
      n0 += tn;
    }
    m0 += tm;
  }
}

void ame_hw_matmul_i8(const int8_t *A, const int8_t *B, int32_t *C,
                      int64_t M, int64_t N, int64_t K,
                      int64_t lda, int64_t ldb, int64_t ldc, int accumulate) {
  ref_matmul_i8(A, B, C, M, N, K, lda, ldb, ldc, accumulate);
}
#else
#include "ame_hw_model.inc"
#endif
