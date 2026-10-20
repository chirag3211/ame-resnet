/* ame16_ops.h: scalar fp16 ports of riscv-dnn's non-matrix ops (relu, add, batchnorm, maxpool, avgpool,
 * padding, casts, softmax). Plain-pointer API, NHWC [h][w][c], fp16 stored as bits (f16 = uint16_t).
 *
 * Numerics follow riscv-dnn's vector code, not "ideal" math:
 *   - fp16 add/mul are done in float and rounded to fp16 once. float has 24 >= 2*11+2 bits, so this is
 *     bit-identical to a native fp16 add/mul (double rounding is innocuous at that width).
 *   - batchnorm = narrow16( fp32(src*alpha) + fp32(beta) )   (vfwmacc: widening MAC, then narrow once)
 *   - avgpool   = fp16 running sum (rounded after every add) * fp16(1/valid_count)   (vfadd.f16 + vfmul.vf)
 *   - softmax is fp32; riscv-dnn uses its own vector exp (exp.h), this port uses expf, so results agree to
 *     ~1e-6 relative, not bit for bit.
 * Differences from riscv-dnn that are NOT numerical: scalar loops instead of RVV (so instruction counts are
 * NOT comparable for these ops), and avgpool uses the window start `pad_l - sw0` (riscv-dnn has `pad_l - j`,
 * identical when stride_w == 1 or pad_l == 0, which covers ResNet-50's global avgpool). */
#ifndef AME16_OPS_H
#define AME16_OPS_H
#include <math.h>
#include <stddef.h>
#include "ame16_hw.h"

static inline float ame16_ld(const f16 *p) { return ame16_h2f(*p); }
static inline void ame16_stf(f16 *p, float x) { *p = ame16_f2h(x); }
static inline float ame16_r16(float x) { return ame16_h2f(ame16_f2h(x)); }   /* round float to fp16 and back */

/* dst = src > base ? src : base */
static void ame16_relu(f16 *dst, const f16 *src, size_t n, f16 base) {
  const float b = ame16_h2f(base);
  for (size_t i = 0; i < n; i++) { float x = ame16_ld(src + i); dst[i] = x > b ? src[i] : base; }
}

/* dst = a + b  (fp16 add) */
static void ame16_add(f16 *dst, const f16 *a, const f16 *b, size_t n) {
  for (size_t i = 0; i < n; i++) ame16_stf(dst + i, ame16_ld(a + i) + ame16_ld(b + i));
}

/* dst[p][ch] = narrow16( src[p][ch]*alpha[ch] + beta[ch] ) with the MAC in fp32; hw = h*w pixels, c channels */
static void ame16_batchnorm(f16 *dst, const f16 *src, const f16 *alpha, const f16 *beta, int hw, int c) {
  for (int j = 0; j < hw; j++)
    for (int k = 0; k < c; k++) {
      float r = ame16_ld(src + (size_t)j * c + k) * ame16_ld(alpha + k) + ame16_ld(beta + k);
      ame16_stf(dst + (size_t)j * c + k, r);
    }
}

/* copy src [hin][win][c] into the interior of a pre-zeroed dst [hin+pt+pb][win+pl+pr][c] */
static void ame16_padding(f16 *dst, const f16 *src, int hin, int win, int c, int pt, int pb, int pl, int pr) {
  const int wout = win + pl + pr;
  (void)pb;
  for (int i = 0; i < hin; i++)
    memcpy(dst + ((size_t)(i + pt) * wout + pl) * c, src + (size_t)i * win * c, (size_t)win * c * sizeof(f16));
}

/* max over the valid (non-padding) taps of each window; windows with no valid tap give 0 */
static void ame16_maxpool(f16 *dst, const f16 *src, int hin, int win, int c, int hout, int wout,
                          int kh, int kw, int sh, int sw, int pt, int pl) {
  for (int i = 0; i < hout; i++)
    for (int j = 0; j < wout; j++) {
      f16 *o = dst + ((size_t)i * wout + j) * c;
      int num = 0;
      for (int m = 0; m < kh; m++) {
        int sy = i * sh + m; if (sy < pt || sy >= pt + hin) continue;
        for (int n = 0; n < kw; n++) {
          int sx = j * sw + n; if (sx < pl || sx >= pl + win) continue;
          const f16 *p = src + ((size_t)(sy - pt) * win + (sx - pl)) * c;
          if (num == 0) memcpy(o, p, (size_t)c * sizeof(f16));
          else for (int k = 0; k < c; k++) if (ame16_ld(p + k) > ame16_ld(o + k)) o[k] = p[k];
          num++;
        }
      }
      if (num == 0) memset(o, 0, (size_t)c * sizeof(f16));
    }
}

/* mean over the valid taps: fp16 running sum, times fp16(1/valid_count) */
static void ame16_avgpool(f16 *dst, const f16 *src, int hin, int win, int c, int hout, int wout,
                          int kh, int kw, int sh, int sw, int pt, int pl) {
  for (int i = 0; i < hout; i++) {
    int sh0 = i * sh;
    int hs = sh0 > pt ? 0 : pt - sh0;
    int he = sh0 + kh > pt + hin ? pt + hin - sh0 : kh;
    for (int j = 0; j < wout; j++) {
      int sw0 = j * sw;
      int ws = sw0 > pl ? 0 : pl - sw0;
      int we = sw0 + kw > pl + win ? pl + win - sw0 : kw;
      int cnt = (he - hs) * (we - ws);
      float recip = ame16_r16((float)(1.0 / (cnt > 0 ? cnt : 1)));     /* double -> float -> fp16, as in riscv-dnn */
      for (int k = 0; k < c; k++) {
        float s = 0.0f;
        for (int m = hs; m < he; m++)
          for (int n = ws; n < we; n++)
            s = ame16_r16(s + ame16_ld(src + ((size_t)(sh0 + m - pt) * win + (sw0 + n - pl)) * c + k));
        ame16_stf(dst + ((size_t)i * wout + j) * c + k, s * recip);
      }
    }
  }
}

static void ame16_cast_f32_to_f16(f16 *dst, const float *src, size_t n) { for (size_t i = 0; i < n; i++) ame16_stf(dst + i, src[i]); }
static void ame16_cast_f16_to_f32(float *dst, const f16 *src, size_t n) { for (size_t i = 0; i < n; i++) dst[i] = ame16_ld(src + i); }

/* fp32 softmax over n values (max-subtracted) */
static void ame16_softmax(float *dst, const float *src, size_t n) {
  float mx = src[0];
  for (size_t i = 1; i < n; i++) if (src[i] > mx) mx = src[i];
  float sum = 0.0f;
  for (size_t i = 0; i < n; i++) { dst[i] = expf(src[i] - mx); sum += dst[i]; }
  for (size_t i = 0; i < n; i++) dst[i] /= sum;
}
#endif
