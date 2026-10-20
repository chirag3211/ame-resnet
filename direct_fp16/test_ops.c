/* Test of the scalar fp16 ops in ame16_ops.h at ResNet-50 sizes, against independent models written with
 * the double-precision helpers (not the float code path used by the ops). Prints per-op mismatch counts and,
 * on Spike, retired instructions per call.
 * Host:  make -C direct_fp16 ops-host-test        Spike: make -C direct_fp16 ops-spike-test */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ame16_ops.h"

static uint64_t rng = 0x243F6A8885A308D3ULL;
static double urand(void) { rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27; return (double)((rng * 0x2545F4914F6CDD1DULL) >> 11) * (1.0 / 9007199254740992.0); }
static double nrand(void) { double s = -6.0; for (int i = 0; i < 12; i++) s += urand(); return s; }
static f16 *rnd16(size_t n, double scale, double shift) { f16 *p = malloc(n * 2); for (size_t i = 0; i < n; i++) p[i] = ame16_d2h(nrand() * scale + shift); return p; }
static int total_bad = 0;

static void report(const char *name, const char *shape, size_t bad, uint64_t instret, size_t n, const char *extra) {
  printf("%-10s %-30s mismatches=%-6zu instret=%-10llu (%.1f/elem) %s\n", name, shape, bad, (unsigned long long)instret, n ? (double)instret / n : 0.0, extra);
  total_bad += bad != 0;
}
static size_t cmp_bits(const f16 *a, const f16 *b, size_t n) { size_t bad = 0; for (size_t i = 0; i < n; i++) bad += a[i] != b[i]; return bad; }

int main(void) {
  char sh[64], ex[96];
  printf("backend=%s\n", AME16_BACKEND);

  { size_t n = 56 * 56 * 256; f16 *x = rnd16(n, 1.0, 0), *y = malloc(n * 2), *r = malloc(n * 2);
    uint64_t t = ame16_instret(); ame16_relu(y, x, n, 0); t = ame16_instret() - t;
    for (size_t i = 0; i < n; i++) r[i] = ame16_h2d(x[i]) > 0 ? x[i] : 0;
    snprintf(sh, sizeof sh, "n=%zu (56x56x256)", n); report("relu", sh, cmp_bits(y, r, n), t, n, ""); free(x); free(y); free(r); }

  { size_t n = 56 * 56 * 256; f16 *a = rnd16(n, 1.0, 0), *b = rnd16(n, 1.0, 0), *y = malloc(n * 2), *r = malloc(n * 2);
    uint64_t t = ame16_instret(); ame16_add(y, a, b, n); t = ame16_instret() - t;
    for (size_t i = 0; i < n; i++) r[i] = ame16_d2h(ame16_h2d(a[i]) + ame16_h2d(b[i]));
    snprintf(sh, sizeof sh, "n=%zu (56x56x256)", n); report("add", sh, cmp_bits(y, r, n), t, n, ""); free(a); free(b); free(y); free(r); }

  { struct { int hw, c; } bn[] = {{3136, 256}, {49, 2048}, {12544, 64}};
    for (int q = 0; q < 3; q++) { int hw = bn[q].hw, c = bn[q].c; size_t n = (size_t)hw * c;
      f16 *x = rnd16(n, 1.0, 0), *al = rnd16(c, 0.3, 1.0), *be = rnd16(c, 0.5, 0), *y = malloc(n * 2), *r = malloc(n * 2);
      uint64_t t = ame16_instret(); ame16_batchnorm(y, x, al, be, hw, c); t = ame16_instret() - t;
      for (int j = 0; j < hw; j++) for (int k = 0; k < c; k++)   /* model: exact MAC -> fp32 -> fp16 */
        r[(size_t)j * c + k] = ame16_d2h((double)(float)(ame16_h2d(x[(size_t)j * c + k]) * ame16_h2d(al[k]) + ame16_h2d(be[k])));
      snprintf(sh, sizeof sh, "hw=%d c=%d", hw, c); report("batchnorm", sh, cmp_bits(y, r, n), t, n, ""); free(x); free(al); free(be); free(y); free(r); } }

  { int hin = 56, win = 56, c = 64, pt = 2, pb = 3, pl = 1, pr = 4; int ho = hin + pt + pb, wo = win + pl + pr; size_t ni = (size_t)hin * win * c, no = (size_t)ho * wo * c;
    f16 *x = rnd16(ni, 1.0, 0), *y = calloc(no, 2), *r = calloc(no, 2);
    uint64_t t = ame16_instret(); ame16_padding(y, x, hin, win, c, pt, pb, pl, pr); t = ame16_instret() - t;
    for (int i = 0; i < hin; i++) for (int j = 0; j < win; j++) for (int k = 0; k < c; k++) r[((size_t)(i + pt) * wo + j + pl) * c + k] = x[((size_t)i * win + j) * c + k];
    snprintf(sh, sizeof sh, "56x56x64 pad t2 b3 l1 r4"); report("padding", sh, cmp_bits(y, r, no), t, ni, ""); free(x); free(y); free(r); }

  struct { const char *tag; int hin, win, c, kh, kw, sh, sw, pt, pb, pl, pr; } pc[] = {
    {"maxpool stem 112x112x64 k3 s2 SAME", 112, 112, 64, 3, 3, 2, 2, 0, 1, 0, 1},
    {"maxpool odd 15x13x8 k3 s2 pad1",      15,  13,  8, 3, 3, 2, 2, 1, 1, 1, 1},
  };
  for (int q = 0; q < 2; q++) { int hin = pc[q].hin, win = pc[q].win, c = pc[q].c;
    int ho = (hin + pc[q].pt + pc[q].pb - pc[q].kh) / pc[q].sh + 1, wo = (win + pc[q].pl + pc[q].pr - pc[q].kw) / pc[q].sw + 1;
    size_t ni = (size_t)hin * win * c, no = (size_t)ho * wo * c;
    f16 *x = rnd16(ni, 1.0, 0), *y = malloc(no * 2), *r = malloc(no * 2);
    uint64_t t = ame16_instret(); ame16_maxpool(y, x, hin, win, c, ho, wo, pc[q].kh, pc[q].kw, pc[q].sh, pc[q].sw, pc[q].pt, pc[q].pl); t = ame16_instret() - t;
    for (int i = 0; i < ho; i++) for (int j = 0; j < wo; j++) for (int k = 0; k < c; k++) {
      double m = -1e300; int any = 0;
      for (int a = 0; a < pc[q].kh; a++) for (int b = 0; b < pc[q].kw; b++) {
        int yy = i * pc[q].sh + a - pc[q].pt, xx = j * pc[q].sw + b - pc[q].pl;
        if (yy < 0 || yy >= hin || xx < 0 || xx >= win) continue;
        double v = ame16_h2d(x[((size_t)yy * win + xx) * c + k]); if (!any || v > m) m = v; any = 1; }
      r[((size_t)i * wo + j) * c + k] = ame16_d2h(any ? m : 0); }
    snprintf(sh, sizeof sh, "-> %dx%dx%d", ho, wo, c); report(pc[q].tag, sh, cmp_bits(y, r, no), t, no, ""); free(x); free(y); free(r); }

  struct { const char *tag; int hin, win, c, kh, kw, sh, sw, pt, pb, pl, pr; } ac[] = {
    {"avgpool global 7x7x2048", 7, 7, 2048, 7, 7, 1, 1, 0, 0, 0, 0},
    {"avgpool 3x3 s2 pad1 9x9x8", 9, 9, 8, 3, 3, 2, 2, 1, 1, 1, 1},
  };
  for (int q = 0; q < 2; q++) { int hin = ac[q].hin, win = ac[q].win, c = ac[q].c;
    int ho = (hin + ac[q].pt + ac[q].pb - ac[q].kh) / ac[q].sh + 1, wo = (win + ac[q].pl + ac[q].pr - ac[q].kw) / ac[q].sw + 1;
    size_t ni = (size_t)hin * win * c, no = (size_t)ho * wo * c;
    f16 *x = rnd16(ni, 1.0, 0.5), *y = malloc(no * 2), *r = malloc(no * 2);
    uint64_t t = ame16_instret(); ame16_avgpool(y, x, hin, win, c, ho, wo, ac[q].kh, ac[q].kw, ac[q].sh, ac[q].sw, ac[q].pt, ac[q].pl); t = ame16_instret() - t;
    double worst = 0;
    for (int i = 0; i < ho; i++) for (int j = 0; j < wo; j++) for (int k = 0; k < c; k++) {
      double s16 = 0, s64 = 0; int cnt = 0;
      for (int a = 0; a < ac[q].kh; a++) for (int b = 0; b < ac[q].kw; b++) {
        int yy = i * ac[q].sh + a - ac[q].pt, xx = j * ac[q].sw + b - ac[q].pl;
        if (yy < 0 || yy >= hin || xx < 0 || xx >= win) continue;
        double v = ame16_h2d(x[((size_t)yy * win + xx) * c + k]); s16 = ame16_h2d(ame16_d2h(s16 + v)); s64 += v; cnt++; }
      double rc = ame16_h2d(ame16_d2h((double)(float)(1.0 / cnt)));
      r[((size_t)i * wo + j) * c + k] = ame16_d2h(s16 * rc);                 /* model A: fp16 running sum, bit-exact */
      double e = fabs(ame16_h2d(y[((size_t)i * wo + j) * c + k]) - s64 / cnt) / (s64 / cnt); if (e > worst) worst = e; }  /* model B: true mean */
    snprintf(ex, sizeof ex, "max rel err vs exact mean %.2e", worst);
    snprintf(sh, sizeof sh, "-> %dx%dx%d", ho, wo, c); report(ac[q].tag, sh, cmp_bits(y, r, no) + (worst > 1e-2), t, no, ex); free(x); free(y); free(r); }

  { size_t n = 1001; float *s = malloc(n * 4); f16 *h = malloc(n * 2), *h2 = malloc(n * 2); float *f = malloc(n * 4);
    for (size_t i = 0; i < n; i++) s[i] = (float)(nrand() * 3.0);
    uint64_t t = ame16_instret(); ame16_cast_f32_to_f16(h, s, n); t = ame16_instret() - t;
    size_t bad = 0; for (size_t i = 0; i < n; i++) bad += h[i] != ame16_d2h((double)s[i]);
    report("cast f32>16", "n=1001", bad, t, n, "");
    t = ame16_instret(); ame16_cast_f16_to_f32(f, h, n); t = ame16_instret() - t;
    bad = 0; for (size_t i = 0; i < n; i++) bad += (double)f[i] != ame16_h2d(h[i]);
    report("cast f16>32", "n=1001", bad, t, n, "");
    (void)h2; free(s); free(h); free(h2); free(f); }

  { size_t n = 1001; float *s = malloc(n * 4), *y = malloc(n * 4);
    for (size_t i = 0; i < n; i++) s[i] = (float)(nrand() * 4.0);
    uint64_t t = ame16_instret(); ame16_softmax(y, s, n); t = ame16_instret() - t;
    double mx = -1e300, sum = 0, worst = 0, ysum = 0; size_t am = 0, ym = 0;
    for (size_t i = 0; i < n; i++) if (s[i] > mx) { mx = s[i]; am = i; }
    for (size_t i = 0; i < n; i++) sum += exp(s[i] - mx);
    for (size_t i = 0; i < n; i++) { double p = exp(s[i] - mx) / sum, e = fabs(y[i] - p); if (e > worst) worst = e; ysum += y[i]; if (y[i] > y[ym]) ym = i; }
    snprintf(ex, sizeof ex, "max abs err %.2e, sum=%.6f", worst, ysum);
    report("softmax", "n=1001 fp32", (worst > 1e-6) + (fabs(ysum - 1) > 1e-4) + (am != ym), t, n, ex); free(s); free(y); }

  printf(total_bad ? "FAIL (%d op case(s))\n" : "ALL OK\n", total_bad);
  return total_bad != 0;
}
