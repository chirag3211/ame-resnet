/* ame16_kernels.h: fp16 GEMM and im2col-conv on the AME matrix unit, written to mirror riscv-dnn's
 * conv() (scalar im2col, then matmul) and matmul_rvm_v3 (tile loops over M, N, K; fp16 in, fp16 out).
 *
 * Layouts (same as riscv-dnn): activations NHWC [h][w][c], weights HWIO flattened to [kh*kw*cin][cout],
 * GEMM A = im2col [hout*wout][kh*kw*cin], B = weights, C = output [hout*wout][cout]. */
#ifndef AME16_KERNELS_H
#define AME16_KERNELS_H
#include "ame16_hw.h"

#ifndef AME16_KC
#define AME16_KC 128   /* max k-elements accumulated in the fp16 tile before a flush to fp32 (STATE.md: ~2e-3 rel err) */
#endif

static const f16 ame16_zero_row[16] __attribute__((aligned(64)));

/* C[M][N] (fp16) = A[M][K] * B[K][N] (fp16 inputs).
 * K <= KC : one fp16 accumulation, stored straight to C (no flush).
 * K  > KC : fp16 accumulation per KC-chunk, flushed to an fp32 sum in software, narrowed to fp16 at the end. */
static void ame16_matmul(const f16 *A, const f16 *B, f16 *C, int M, int N, int K, int KC) {
  static f16 Ct[8 * 16] __attribute__((aligned(64)));   /* flush buffer, 32-byte row stride */
  ame16_set_sew16(); ame16_set_fp16();
  for (int m0 = 0; m0 < M;) {
    int tm = (int)AME16_SETTILE_M(M - m0);
    for (int n0 = 0; n0 < N;) {
      int tn = (int)AME16_SETTILE_N(N - n0);
      if (K <= KC) {
        ame16_ld_c(ame16_zero_row, 0);                       /* clear acc0 */
        for (int k0 = 0; k0 < K;) {
          int tk = (int)AME16_SETTILE_K(K - k0);
          ame16_ld_a(A + (int64_t)m0 * K + k0, (uint64_t)K * 2);
          ame16_ld_b(B + (int64_t)k0 * N + n0, (uint64_t)N * 2);
          ame16_mfma();
          k0 += tk;
        }
        ame16_st_c(C + (int64_t)m0 * N + n0, (uint64_t)N * 2);
      } else {
        float acc[8 * 8];
        for (int i = 0; i < tm * tn; i++) acc[i] = 0.0f;
        for (int k0 = 0; k0 < K;) {
          int kend = k0 + KC < K ? k0 + KC : K;
          ame16_ld_c(ame16_zero_row, 0);
          while (k0 < kend) {
            int tk = (int)AME16_SETTILE_K(kend - k0);
            ame16_ld_a(A + (int64_t)m0 * K + k0, (uint64_t)K * 2);
            ame16_ld_b(B + (int64_t)k0 * N + n0, (uint64_t)N * 2);
            ame16_mfma();
            k0 += tk;
          }
          ame16_st_c(Ct, 32);
          for (int i = 0; i < tm; i++) for (int j = 0; j < tn; j++) acc[i * 8 + j] += ame16_h2f(Ct[i * 16 + j]);
        }
        for (int i = 0; i < tm; i++) for (int j = 0; j < tn; j++) C[(int64_t)(m0 + i) * N + n0 + j] = ame16_f2h(acc[i * 8 + j]);
      }
      n0 += tn;
    }
    m0 += tm;
  }
}

typedef struct {
  int hin, win, cin, hout, wout, cout, kh, kw;
  int stride_h, stride_w, pad_t, pad_l, dil_h, dil_w;
} ame16_conv_cfg;

/* riscv-dnn conv() step 1: zero the col buffer, scalar im2col (skips out-of-range taps). col = [hout*wout][kh*kw*cin] */
static void ame16_im2col(f16 *col, const f16 *src, const ame16_conv_cfg *c) {
  const int K = c->kh * c->kw * c->cin, M = c->hout * c->wout;
  memset(col, 0, (size_t)M * K * sizeof(f16));
  for (int i = 0; i < c->hout; i++)
    for (int j = 0; j < c->wout; j++)
      for (int ii = 0; ii < c->kh; ii++)
        for (int jj = 0; jj < c->kw; jj++) {
          int _i = i * c->stride_h - c->pad_t + ii * c->dil_h;
          int _j = j * c->stride_w - c->pad_l + jj * c->dil_w;
          if (_i < 0 || _i > c->hin - 1 || _j < 0 || _j > c->win - 1) continue;
          for (int kk = 0; kk < c->cin; kk++)
            col[(int64_t)(i * c->wout + j) * K + (ii * c->kw + jj) * c->cin + kk] = src[(int64_t)(_i * c->win + _j) * c->cin + kk];
        }
}

/* conv = im2col + matmul (same two steps as riscv-dnn's conv()) */
static void ame16_conv(f16 *dst, const f16 *src, const f16 *w, f16 *col, const ame16_conv_cfg *c, int KC) {
  ame16_im2col(col, src, c);
  ame16_matmul(col, w, dst, c->hout * c->wout, c->cout, c->kh * c->kw * c->cin, KC);
}
#endif
