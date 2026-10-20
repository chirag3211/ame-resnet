/* Host/Spike test of ame16_conv / ame16_matmul against an independent direct-convolution reference (double,
 * on the same fp16-rounded inputs). Layer shapes are ResNet-50 cases scaled down so the host emulation is fast.
 * Host:  gcc -O2 -I../include test_kernels.c -o test_kernels_host -lm && ./test_kernels_host
 * Spike: see Makefile (target spike-test). Prints per-case error and matrix-instruction counts (mfma column: host emulation only, 0 on Spike; instret column: Spike only). */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ame16_kernels.h"

static uint64_t rng = 0x9E3779B97F4A7C15ULL;
static double urand(void) { rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27; return (double)((rng * 0x2545F4914F6CDD1DULL) >> 11) * (1.0 / 9007199254740992.0); }
static double nrand(void) { double s = -6.0; for (int i = 0; i < 12; i++) s += urand(); return s; }
static int cdiv(int a, int b) { return (a + b - 1) / b; }

typedef struct { const char *name; ame16_conv_cfg c; } Case;

static int run_case(const Case *cs, int KC, int enforce) {
  const ame16_conv_cfg *c = &cs->c;
  const int K = c->kh * c->kw * c->cin, M = c->hout * c->wout, N = c->cout;
  f16 *src = malloc((size_t)c->hin * c->win * c->cin * 2), *w = malloc((size_t)K * N * 2);
  f16 *col = malloc((size_t)M * K * 2), *dst = malloc((size_t)M * N * 2);
  for (int i = 0; i < c->hin * c->win * c->cin; i++) src[i] = ame16_d2h(urand());               /* post-relu-like, >= 0 */
  for (int i = 0; i < K * N; i++) w[i] = ame16_d2h(nrand() / sqrt((double)K));
  memset(&ame16_stats, 0, sizeof ame16_stats);
  uint64_t t0 = ame16_instret();
  ame16_conv(dst, src, w, col, c, KC);
  uint64_t instret = ame16_instret() - t0;                 /* Spike: retired instructions incl. im2col; host: 0 */
  /* independent reference: direct convolution, no im2col */
  double se = 0, sr = 0, maxabs = 0;
  for (int i = 0; i < c->hout; i++) for (int j = 0; j < c->wout; j++) for (int o = 0; o < N; o++) {
    double s = 0;
    for (int ii = 0; ii < c->kh; ii++) for (int jj = 0; jj < c->kw; jj++) {
      int _i = i * c->stride_h - c->pad_t + ii * c->dil_h, _j = j * c->stride_w - c->pad_l + jj * c->dil_w;
      if (_i < 0 || _i >= c->hin || _j < 0 || _j >= c->win) continue;
      for (int kk = 0; kk < c->cin; kk++)
        s += ame16_h2d(src[(_i * c->win + _j) * c->cin + kk]) * ame16_h2d(w[((ii * c->kw + jj) * c->cin + kk) * N + o]);
    }
    double g = ame16_h2d(dst[(i * c->wout + j) * N + o]), e = fabs(g - s);
    if (e > maxabs) maxabs = e;
    se += e * e; sr += s * s;
  }
  double macs = (double)M * N * K;
  uint64_t ideal = (uint64_t)cdiv(M, 8) * cdiv(N, 8) * cdiv(K, 8);   /* tile-granular mfma count, 8x8x8 tiles */
  printf("%-26s M=%-5d N=%-5d K=%-5d KC=%-10d rel_rms=%.2e maxabs=%.2e | mfma=%llu (tile-ideal %llu, MACs/512=%.0f) instret=%llu\n",
         cs->name, M, N, K, KC, sqrt(se / (sr > 0 ? sr : 1)), maxabs,
         (unsigned long long)ame16_stats.mfma, (unsigned long long)ideal, macs / 512.0, (unsigned long long)instret);
  int ok = !enforce || sqrt(se / (sr > 0 ? sr : 1)) < 5e-3;   /* the no-flush contrast run is informational only */
#ifndef __riscv
  ok = ok && ame16_stats.mfma == ideal;                         /* tile count must match, for every run */
#endif
  free(src); free(w); free(col); free(dst);
  return ok;
}

int main(void) {
  printf("backend=%s\n", AME16_BACKEND);
  Case cases[] = {
    /* name                 hin win cin hout wout cout kh kw sh sw pt pl dh dw */
    {"1x1 s1 (bottleneck)",  {14, 14, 256, 14, 14,  64, 1, 1, 1, 1, 0, 0, 1, 1}},
    {"1x1 s2 (downsample)",  {14, 14, 128,  7,  7, 256, 1, 1, 2, 2, 0, 0, 1, 1}},
    {"3x3 s1 p1",            {14, 14, 128, 14, 14,  64, 3, 3, 1, 1, 1, 1, 1, 1}},
    {"3x3 s2 p1",            {28, 28,  64, 14, 14,  64, 3, 3, 2, 2, 1, 1, 1, 1}},
    {"7x7 s2 p3 (stem)",     {32, 32,   3, 16, 16,  64, 7, 7, 2, 2, 3, 3, 1, 1}},
  };
  int ok = 1;
  for (unsigned i = 0; i < sizeof cases / sizeof cases[0]; i++) {
    ok &= run_case(&cases[i], AME16_KC, 1);
    ok &= run_case(&cases[i], 1 << 30, 0);       /* no flush: pure fp16 accumulation, for contrast */
  }
  /* fc: 1 x 2048 x 1001 (odd N: edge tile) as a plain matmul */
  { const int M = 1, N = 1001, K = 2048;
    f16 *A = malloc(K * 2), *B = malloc((size_t)K * N * 2), *C = malloc(N * 2);
    for (int i = 0; i < K; i++) A[i] = ame16_d2h(urand());
    for (int i = 0; i < K * N; i++) B[i] = ame16_d2h(nrand() / sqrt((double)K));
    memset(&ame16_stats, 0, sizeof ame16_stats);
    uint64_t t0 = ame16_instret();
    ame16_matmul(A, B, C, M, N, K, AME16_KC);
    uint64_t instret = ame16_instret() - t0;
    double se = 0, sr = 0;
    for (int j = 0; j < N; j++) { double s = 0; for (int k = 0; k < K; k++) s += ame16_h2d(A[k]) * ame16_h2d(B[(size_t)k * N + j]); double e = ame16_h2d(C[j]) - s; se += e * e; sr += s * s; }
    uint64_t ideal = (uint64_t)cdiv(M, 8) * cdiv(N, 8) * cdiv(K, 8);
    printf("%-26s M=%-5d N=%-5d K=%-5d KC=%-10d rel_rms=%.2e | mfma=%llu (tile-ideal %llu) instret=%llu\n", "fc 2048->1001", M, N, K, AME16_KC, sqrt(se / sr), (unsigned long long)ame16_stats.mfma, (unsigned long long)ideal, (unsigned long long)instret);
    ok &= sqrt(se / sr) < 5e-3;
#ifndef __riscv
    ok &= ame16_stats.mfma == ideal;
#endif
    free(A); free(B); free(C); }
  printf(ok ? "ALL OK\n" : "FAIL\n");
  return ok ? 0 : 1;
}
