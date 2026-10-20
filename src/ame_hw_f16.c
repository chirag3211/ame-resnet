/* fp16 widening AME matmul (fp16 in, fp32 accumulate, fp16 out).  Bench-only: not wired into ame_dispatch.
 * RISC-V only (needs the matrix instructions). */
#ifdef __riscv
#include "ame_f16.h"

static const float ame_zero_row_f32[64] __attribute__((aligned(64)));

/* ---- faithful: instruction-for-instruction the sequence the compiler emits for riscv-dnn ---- */
void ame_hw_matmul_f16w_faithful(const uint16_t *A, const uint16_t *B, uint16_t *C,
                                 int64_t M, int64_t N, int64_t K, int64_t lda, int64_t ldb, int64_t ldc) {
  if (M <= 0 || N <= 0 || K <= 0) return;
  for (int64_t m0 = 0; m0 < M; ) {
    ame_mtypei_e16(); ame_mtypehi_fp16();
    int64_t tm = (int64_t)ame_settile_m((uint64_t)(M - m0));
    for (int64_t n0 = 0; n0 < N; ) {
      int64_t tn = (int64_t)ame_settile_n((uint64_t)(N - n0));
      ame_mtypei_e32(); ame_mtypehi_fp32();                 /* clear acc0 as an fp32 tile */
      ame_mfsub_clear();
      for (int64_t k0 = 0; k0 < K; ) {
        ame_mtypei_e16(); ame_mtypehi_fp16();               /* loads run as fp16 */
        int64_t tk = (int64_t)ame_settile_k((uint64_t)(K - k0));
        ame_ld_a16(A + m0 * lda + k0, (uint64_t)(lda * 2));
        ame_ld_b16(B + k0 * ldb + n0, (uint64_t)(ldb * 2));
        ame_mtypei_e16(); ame_mtypehi_fp16_fp32();          /* multiply widens fp16 -> fp32 */
        ame_mfwma_f16();
        k0 += tk;
      }
      ame_mtypei_e32(); ame_mtypehi_fp16_fp32();            /* narrow fp32 -> fp16 */
      ame_mfncvt_a1_a0();
      ame_mtypei_e16(); ame_mtypehi_fp16();                 /* store as fp16 */
      ame_st_c16_a1(C + m0 * ldc + n0, (uint64_t)(ldc * 2));
      n0 += tn;
    }
    m0 += tm;
  }
}

/* ---- hoisted: same arithmetic, redundant mtype writes removed.
 * What Spike (matrix branch 6add067) does, measured and read from matrix_unit.cc:
 *   - msettypei  replaces the whole of mtype with its immediate (clears the high fields);
 *     msettypehi ORs bits into the high fields; msetsew changes only the element-width bits.
 *   - the granted tile depends ONLY on the element width: e16 -> 8x8x8 (512 MACs), e32 -> 8x4x4 (128 MACs);
 *     the fp type fields do not change it.  So every msettile* must run in an e16 state.
 *   - mlae16/mlbe16/mfwma/mlce32 and msettilem/n/k all run in (e16, fp16->fp32).
 *   - mfncvt.f.fw.m needs (e32, fp16->fp32) and traps in the e16 state; msce16.m runs fine in the e32 state.
 * Hence: set (e16, fp16->fp32) once per call, and toggle only the element width with msetsew around the
 * narrowing step: 2 one-instruction writes per output tile, against the compiler's 12 type writes per
 * output tile plus 4 per K-step.  The accumulator is cleared by loading a zero row (mlce32, stride 0), as in
 * ame_hw_matmul_f32.  Validated on Spike only; the spec does not promise these states are interchangeable. */
void ame_hw_matmul_f16w_hoisted(const uint16_t *A, const uint16_t *B, uint16_t *C,
                                int64_t M, int64_t N, int64_t K, int64_t lda, int64_t ldb, int64_t ldc) {
  if (M <= 0 || N <= 0 || K <= 0) return;
  ame_mtypei_e16(); ame_mtypehi_fp16_fp32();                /* once per call */
  for (int64_t m0 = 0; m0 < M; ) {
    int64_t tm = (int64_t)ame_settile_m((uint64_t)(M - m0));
    for (int64_t n0 = 0; n0 < N; ) {
      int64_t tn = (int64_t)ame_settile_n((uint64_t)(N - n0));
      ame_ld_c32(ame_zero_row_f32, 0);                      /* mlce32.m acc0 <- 0 (fp32 accumulator) */
      for (int64_t k0 = 0; k0 < K; ) {
        int64_t tk = (int64_t)ame_settile_k((uint64_t)(K - k0));
        ame_ld_a16(A + m0 * lda + k0, (uint64_t)(lda * 2));
        ame_ld_b16(B + k0 * ldb + n0, (uint64_t)(ldb * 2));
        ame_mfwma_f16();
        k0 += tk;
      }
      ame_set_sew32();                                      /* narrowing needs e32 (msetsew keeps the fp fields) */
      ame_mfncvt_a1_a0();
      ame_st_c16_a1(C + m0 * ldc + n0, (uint64_t)(ldc * 2));
      ame_set_sew16();                                      /* back to the 8x8x8 state for the next tile */
      n0 += tn;
    }
    m0 += tm;
  }
}
#endif /* __riscv */
