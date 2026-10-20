/* One-matmul benchmark: the fp32 AME kernel vs the fp16-widening AME kernels on the same shape.
 *   -DBENCH_KERNEL=0  ame_hw_matmul_f32           fp32 in/out        (needs -DUSE_AME, src/ame_hw.c, src/ref_matmul.c)
 *   -DBENCH_KERNEL=1  ame_hw_matmul_f16w_faithful fp16 in, fp32 acc, fp16 out; compiler-style type switching
 *   -DBENCH_KERNEL=2  ame_hw_matmul_f16w_hoisted  same arithmetic, redundant mtype writes removed
 *   -DBENCH_M= -DBENCH_K= -DBENCH_N= (default 64)   -DBENCH_PAD= extra leading-dimension padding (default 0)
 * The kernel runs twice (cold, warm); the result and the instret figure are from the warm run.
 * Prints integers only, so it also works with the tiny bare-metal printf. */
#include <stdint.h>
#ifdef BAREMETAL
extern int printf(const char *, ...);
#else
#include <stdio.h>
#endif
#include "ame_kernel.h"
#include "ame_f16.h"

/* command-line names are BENCH_*: plain -DM/-DN/-DK would clash with the parameter names in the kernel headers */
#ifndef BENCH_M
#define BENCH_M 64
#endif
#ifndef BENCH_K
#define BENCH_K 64
#endif
#ifndef BENCH_N
#define BENCH_N 64
#endif
#ifndef BENCH_PAD
#define BENCH_PAD 0
#endif
#ifndef BENCH_KERNEL
#define BENCH_KERNEL 1
#endif
#define M BENCH_M          /* defined only after the headers above are included */
#define K BENCH_K
#define N BENCH_N
#define PAD BENCH_PAD
#define KERNEL BENCH_KERNEL
#define LDA (K + PAD)
#define LDB (N + PAD)
#define LDC (N + PAD)

static uint32_t rng = 12345u;
static uint32_t nextr(void) { rng = rng * 1664525u + 1013904223u; return rng >> 8; }
static float rnd_f(void) { return ((int)(nextr() % 2001) - 1000) / 1000.0f; }   /* [-1, 1] */

static inline uint64_t rd_instret(void) { uint64_t v; __asm__ volatile("rdinstret %0" : "=r"(v)); return v; }

/* IEEE fp16 <-> fp32 (round to nearest even), software, independent of compiler fp16 support */
static __attribute__((unused)) uint16_t f2h(float f) {
  union { float f; uint32_t u; } v = { f };
  uint32_t x = v.u, sign = (x >> 16) & 0x8000u, m = x & 0x7fffffu;
  int e = (int)((x >> 23) & 0xff) - 127 + 15;
  if (e <= 0) {
    if (e < -10) return (uint16_t)sign;
    m |= 0x800000u;
    int sh = 14 - e;
    uint32_t r = m >> sh, rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
    if (rem > half || (rem == half && (r & 1))) r++;
    return (uint16_t)(sign | r);
  }
  if (e >= 31) return (uint16_t)(sign | 0x7c00u);
  uint32_t r = ((uint32_t)e << 10) | (m >> 13), rem = m & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) r++;
  return (uint16_t)(sign | r);
}
static __attribute__((unused)) float h2f(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16, m = h & 0x3ffu, u;
  int e = (h >> 10) & 0x1f;
  if (e == 0) {
    if (m == 0) u = sign;
    else { e = 1; while (!(m & 0x400u)) { m <<= 1; e--; } m &= 0x3ffu; u = sign | ((uint32_t)(e + 127 - 15) << 23) | (m << 13); }
  } else if (e == 31) u = sign | 0x7f800000u | (m << 13);
  else u = sign | ((uint32_t)(e + 127 - 15) << 23) | (m << 13);
  union { uint32_t u; float f; } v = { u };
  return v.f;
}

#if KERNEL == 0
static float    gA[M * LDA] __attribute__((aligned(64))), gB[K * LDB] __attribute__((aligned(64))), gC[M * LDC] __attribute__((aligned(64)));
static const char *kname = "f32";
#else
static uint16_t hA[M * LDA] __attribute__((aligned(64))), hB[K * LDB] __attribute__((aligned(64))), hC[M * LDC] __attribute__((aligned(64)));
static const char *kname = (KERNEL == 1) ? "f16w_faithful" : "f16w_hoisted";
#endif

static void run(void) {
#if KERNEL == 0
  ame_hw_matmul_f32(gA, gB, gC, M, N, K, LDA, LDB, LDC, 0);
#elif KERNEL == 1
  ame_hw_matmul_f16w_faithful(hA, hB, hC, M, N, K, LDA, LDB, LDC);
#else
  ame_hw_matmul_f16w_hoisted(hA, hB, hC, M, N, K, LDA, LDB, LDC);
#endif
}

int main(void) {
  for (int i = 0; i < M; i++) for (int k = 0; k < K; k++) {
    float a = rnd_f();
#if KERNEL == 0
    gA[i * LDA + k] = a;
#else
    hA[i * LDA + k] = f2h(a);
#endif
  }
  for (int k = 0; k < K; k++) for (int j = 0; j < N; j++) {
    float b = rnd_f();
#if KERNEL == 0
    gB[k * LDB + j] = b;
#else
    hB[k * LDB + j] = f2h(b);
#endif
  }
  run();                                   /* cold run: page-faults / first touch under pk */
  uint64_t t0 = rd_instret();
  run();                                   /* warm run: measured */
  uint64_t instret = rd_instret() - t0;

  long bad = 0; double maxerr = 0;
#ifndef BENCH_NOCHECK        /* -DBENCH_NOCHECK skips the O(MNK) reference check for big shapes */
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      double ref = 0;
      for (int k = 0; k < K; k++) {
#if KERNEL == 0
        ref += (double)gA[i * LDA + k] * (double)gB[k * LDB + j];
        double got = gC[i * LDC + j];
        double tol = 1e-3 * (ref < 0 ? -ref : ref) + 1e-3;      /* fp32 path: loose on purpose, as test_matmul.c */
        if (k == K - 1) { double d = got - ref; if (d < 0) d = -d; if (d > maxerr) maxerr = d; if (d > tol) bad++; }
#else
        ref += (double)h2f(hA[i * LDA + k]) * (double)h2f(hB[k * LDB + j]);
        if (k == K - 1) {
          double got = h2f(hC[i * LDC + j]);
          double d = got - ref; if (d < 0) d = -d;
          double tol = 1e-3 * (ref < 0 ? -ref : ref) + 1e-4;    /* fp16 output rounding is <= 2^-11 relative */
          if (d > maxerr) maxerr = d;
          if (d > tol) bad++;
        }
#endif
      }
    }
#endif
  printf("BENCH kernel=%s M=%d N=%d K=%d pad=%d %s bad=%d maxerr_ppm=%d macs=%d instret=%d\n",
         kname, M, N, K, PAD,
#ifdef BENCH_NOCHECK
         "UNCHECKED",
#else
         bad ? "FAIL" : "PASS",
#endif
         (int)bad, (int)(maxerr * 1e6),
         (int)((uint64_t)M * N * K), (int)instret);
  return bad ? 1 : 0;
}
