/* ame16_hw.h: fp16 matrix-unit access for the hand-written ("direct") ResNet-50.
 *   __riscv : real AME instructions (same encodings as tests/ame_fp16_err.c, which ran on Spike)
 *   host    : software EMULATION of the same operations (index/shape/accuracy checks only; NOT a Spike measurement)
 * Tile semantics (as used in tests/ame_fp16_err.c): A tile = tm x tk (row stride given), B tile = tk x tn,
 * acc/C tile = tm x tn, fp16 accumulator, mfma_hf: C += A*B with rounding to fp16 after every k-step. */
#ifndef AME16_HW_H
#define AME16_HW_H
#include <stdint.h>
#include <string.h>

typedef uint16_t f16;                       /* IEEE half, stored as bits */

/* ---------- fp16 <-> double/float, round-to-nearest-even, no dependency on _Float16 ---------- */
static inline f16 ame16_d2h(double x) {
  uint64_t b; memcpy(&b, &x, 8);
  uint16_t s = (uint16_t)((b >> 63) << 15);
  int e = (int)((b >> 52) & 0x7ff);
  uint64_t m = b & ((1ULL << 52) - 1);
  if (e == 0x7ff) return (uint16_t)(s | 0x7c00 | (m ? 0x200 : 0));
  if (e == 0) return s;
  e -= 1023;
  if (e > 15) return (uint16_t)(s | 0x7c00);
  uint64_t full = (1ULL << 52) | m;
  int shift = 42;
  if (e < -14) shift += (-14 - e);
  if (shift >= 63) return s;
  uint64_t r = full >> shift, rem = full & ((1ULL << shift) - 1), half = 1ULL << (shift - 1);
  if (rem > half || (rem == half && (r & 1))) r++;
  if (e < -14) return (uint16_t)(s | r);
  int he = e + 15;
  if (r == 2048) { he++; r = 1024; }
  if (he >= 31) return (uint16_t)(s | 0x7c00);
  return (uint16_t)(s | (he << 10) | (r & 1023));
}
static inline double ame16_h2d(f16 h) {
  int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
  double v;
  if (e == 0) v = m * (1.0 / 16777216.0);
  else if (e == 31) v = m ? 0.0 / 0.0 : 1e300;
  else { v = 1.0 + m / 1024.0; int x = e - 15; while (x > 0) { v *= 2; x--; } while (x < 0) { v *= 0.5; x++; } }
  return s ? -v : v;
}

typedef struct { uint64_t mfma, ld_a, ld_b, ld_c, st_c, settile; } ame16_stats_t;
static ame16_stats_t ame16_stats;           /* advanced by the host emulation only; on Spike use ame16_instret() */

#ifdef __riscv
#include "ame_insn.h"
static inline void ame16_set_sew16(void) { __asm__ volatile(".insn r 0x77, 6, 0x01, x0, x0, x1" ::: "memory"); }
static inline void ame16_set_fp16(void)  { __asm__ volatile(".insn r 0x77, 6, 0x01, x0, x7, x1" ::: "memory"); }
AME_MEM(ame16_ld_a, 0x02, 1, x0)   /* mlae16.m tr0  */
AME_MEM(ame16_ld_b, 0x04, 1, x1)   /* mlbe16.m tr1  */
AME_MEM(ame16_ld_c, 0x00, 1, x0)   /* mlce16.m acc0 */
AME_MEM(ame16_st_c, 0x01, 1, x0)   /* msce16.m acc0 */
static inline void ame16_mfma(void) { __asm__ volatile(".insn r 0x77, 1, 0x11, x16, x0, x1" ::: "memory"); }
static inline uint64_t ame16_instret(void) { uint64_t v; __asm__ volatile("rdinstret %0" : "=r"(v)); return v; }
#define AME16_SETTILE_M(r) ame_settile_m((uint64_t)(r))
#define AME16_SETTILE_K(r) ame_settile_k((uint64_t)(r))
#define AME16_SETTILE_N(r) ame_settile_n((uint64_t)(r))
#define AME16_BACKEND "spike-hw"
#if defined(__riscv_zfh)
static inline float ame16_h2f(f16 h) { _Float16 f; memcpy(&f, &h, 2); return (float)f; }   /* fcvt.s.h */
static inline f16 ame16_f2h(float x) { _Float16 f = (_Float16)x; f16 h; memcpy(&h, &f, 2); return h; }  /* fcvt.h.s */
#else
static inline float ame16_h2f(f16 h) { return (float)ame16_h2d(h); }   /* slow: would inflate instruction counts */
static inline f16 ame16_f2h(float x) { return ame16_d2h((double)x); }
#endif
#else  /* ---------------- host emulation ---------------- */
static int E_tm, E_tk, E_tn;
static f16 E_a[16 * 16], E_b[16 * 16], E_c[16 * 16];
static inline void ame16_set_sew16(void) {}
static inline void ame16_set_fp16(void) {}
static inline uint64_t AME16_SETTILE_M(uint64_t r) { ame16_stats.settile++; E_tm = r < 8 ? (int)r : 8; return E_tm; }
static inline uint64_t AME16_SETTILE_K(uint64_t r) { ame16_stats.settile++; E_tk = r < 8 ? (int)r : 8; return E_tk; }
static inline uint64_t AME16_SETTILE_N(uint64_t r) { ame16_stats.settile++; E_tn = r < 8 ? (int)r : 8; return E_tn; }
#define AME16_ROW(p, i, st) ((const f16 *)((const char *)(p) + (int64_t)(i) * (int64_t)(st)))
static inline void ame16_ld_a(const void *p, uint64_t st) { ame16_stats.ld_a++; for (int i = 0; i < E_tm; i++) for (int j = 0; j < E_tk; j++) E_a[i * 16 + j] = AME16_ROW(p, i, st)[j]; }
static inline void ame16_ld_b(const void *p, uint64_t st) { ame16_stats.ld_b++; for (int i = 0; i < E_tk; i++) for (int j = 0; j < E_tn; j++) E_b[i * 16 + j] = AME16_ROW(p, i, st)[j]; }
static inline void ame16_ld_c(const void *p, uint64_t st) { ame16_stats.ld_c++; for (int i = 0; i < E_tm; i++) for (int j = 0; j < E_tn; j++) E_c[i * 16 + j] = AME16_ROW(p, i, st)[j]; }
static inline void ame16_st_c(void *p, uint64_t st) { ame16_stats.st_c++; for (int i = 0; i < E_tm; i++) for (int j = 0; j < E_tn; j++) ((f16 *)((char *)p + (int64_t)i * (int64_t)st))[j] = E_c[i * 16 + j]; }
static inline void ame16_mfma(void) {
  ame16_stats.mfma++;
  for (int i = 0; i < E_tm; i++) for (int j = 0; j < E_tn; j++) {
    f16 c = E_c[i * 16 + j];
    for (int k = 0; k < E_tk; k++) c = ame16_d2h(ame16_h2d(E_a[i * 16 + k]) * ame16_h2d(E_b[k * 16 + j]) + ame16_h2d(c));
    E_c[i * 16 + j] = c;
  }
}
static inline uint64_t ame16_instret(void) { return 0; }
static inline float ame16_h2f(f16 h) { return (float)ame16_h2d(h); }
static inline f16 ame16_f2h(float x) { return ame16_d2h((double)x); }
#define AME16_BACKEND "host-emulation"
#endif
#endif
