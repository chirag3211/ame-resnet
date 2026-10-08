/* fp16 error test: matmul through mfma_hf (fp16 accumulator in the matrix unit) with K-chunking.
 *
 * Question: with K = 768 (ViT), 1152 (Moonshine FFN) and 3072 (ViT fc2), how large is the error if the
 * accumulator is fp16 for CHUNK k-elements and the partial result is then flushed (msce16, convert,
 * fp32 add in software)? chunk = K means no flush at all (pure fp16 accumulation).
 * The flush cost (store + convert + add per output element) is INCLUDED in instr/MAC.
 *
 * Per (K, data mode) it prints:
 *   floor   : error of the fp16-ROUNDED INPUTS alone (double reference on rounded inputs vs on the original
 *             fp32 inputs). Nothing that keeps fp16 inputs can beat this, whatever the accumulator does.
 *   fp32acc : control, fp32 sequential accumulation of the rounded inputs (what an fp16-input /
 *             fp32-accumulate unit would give).
 *   chunk=c : mfma_hf with c k-elements per fp16 accumulation, fp32 flush between chunks.
 * Error is measured against the double reference on the rounded inputs ("accum err"), except 'floor'.
 *
 * Real run (Spike), needs zfh in the ISA and in the compiler (the flush then uses fcvt.s.h):
 *   riscv64-unknown-elf-gcc -O2 -march=rv64gc_zfh -mabi=lp64d -Iinclude tests/ame_fp16_err.c -o build/ame_fp16_err.elf -lm
 *   $HOME/riscv-stc/bin/spike --isa=rv64imafdcv_zfh_zicntr_matrix $HOME/riscv/riscv64-unknown-elf/bin/pk build/ame_fp16_err.elf
 * Host run (software EMULATION of mfma_hf, NOT a measurement of Spike; assumes the accumulator is rounded to
 * fp16 after every k-step, sequentially over k, as read from MXU_VFP_VV_LOOP in the Spike source):
 *   gcc -O2 tests/ame_fp16_err.c -Iinclude -o build/ame_fp16_err_host -lm && ./build/ame_fp16_err_host
 * Options: -DM_=32 -DN_=32 (output size; error depends on K and data, not on M,N), -DKLIST="768,1152,3072". */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#ifndef M_
#define M_ 32
#endif
#ifndef N_
#define N_ 32
#endif
#ifndef KLIST
#define KLIST 768, 1152, 3072
#endif
#define KMAX 3072
#define ZSTRIDE 32            /* tile buffers: 16 halfs = 32 bytes per row */

/* ---------- fp16 <-> double (round-to-nearest-even), no dependency on _Float16 ---------- */
static uint16_t d2h(double x) {
  uint64_t b; memcpy(&b, &x, 8);
  uint16_t s = (uint16_t)((b >> 63) << 15);
  int e = (int)((b >> 52) & 0x7ff);
  uint64_t m = b & ((1ULL << 52) - 1);
  if (e == 0x7ff) return (uint16_t)(s | 0x7c00 | (m ? 0x200 : 0));
  if (e == 0) return s;                              /* zero / double subnormal -> 0 */
  e -= 1023;
  if (e > 15) return (uint16_t)(s | 0x7c00);
  uint64_t full = (1ULL << 52) | m;
  int shift = 42;                                    /* 52 - 10 */
  if (e < -14) shift += (-14 - e);
  if (shift >= 63) return s;
  uint64_t r = full >> shift, rem = full & ((1ULL << shift) - 1), half = 1ULL << (shift - 1);
  if (rem > half || (rem == half && (r & 1))) r++;
  if (e < -14) return (uint16_t)(s | r);             /* subnormal; r==1024 is the smallest normal, bits match */
  int he = e + 15;
  if (r == 2048) { he++; r = 1024; }
  if (he >= 31) return (uint16_t)(s | 0x7c00);
  return (uint16_t)(s | (he << 10) | (r & 1023));
}
static double h2d(uint16_t h) {
  int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
  double v;
  if (e == 0) v = m * (1.0 / 16777216.0);            /* m * 2^-24 */
  else if (e == 31) v = m ? 0.0 / 0.0 : 1e300;
  else { v = 1.0 + m / 1024.0; int x = e - 15; while (x > 0) { v *= 2; x--; } while (x < 0) { v *= 0.5; x++; } }
  return s ? -v : v;
}

/* ---------- matrix-unit access: real instructions on RISC-V, emulation on the host ---------- */
#ifdef __riscv
#include "ame_insn.h"
static inline void set_sew16(void) { __asm__ volatile(".insn r 0x77, 6, 0x01, x0, x0, x1" ::: "memory"); }
static inline void set_fp16(void)  { __asm__ volatile(".insn r 0x77, 6, 0x01, x0, x7, x1" ::: "memory"); }
AME_MEM(ld_a16, 0x02, 1, x0)   /* mlae16.m tr0  */
AME_MEM(ld_b16, 0x04, 1, x1)   /* mlbe16.m tr1  */
AME_MEM(ld_c16, 0x00, 1, x0)   /* mlce16.m acc0 */
AME_MEM(st_c16, 0x01, 1, x0)   /* msce16.m acc0 */
static inline void mfma_hf(void) { __asm__ volatile(".insn r 0x77, 1, 0x11, x16, x0, x1" ::: "memory"); }
static inline uint64_t instret(void) { uint64_t v; __asm__ volatile("csrr %0, instret" : "=r"(v)); return v; }
#define BACKEND "spike-hw"
#else
static int E_tm, E_tk, E_tn;
static uint16_t E_a[16 * 16], E_b[16 * 16], E_c[16 * 16];
static void set_sew16(void) {}
static void set_fp16(void) {}
static uint64_t ame_settile_m(uint64_t r) { E_tm = r < 8 ? (int)r : 8; return E_tm; }
static uint64_t ame_settile_k(uint64_t r) { E_tk = r < 8 ? (int)r : 8; return E_tk; }
static uint64_t ame_settile_n(uint64_t r) { E_tn = r < 8 ? (int)r : 8; return E_tn; }
#define ROW(p, i, st) ((const uint16_t *)((const char *)(p) + (uint64_t)(i) * (st)))
static void ld_a16(const void *p, uint64_t st) { for (int i = 0; i < E_tm; i++) for (int j = 0; j < E_tk; j++) E_a[i * 16 + j] = ROW(p, i, st)[j]; }
static void ld_b16(const void *p, uint64_t st) { for (int i = 0; i < E_tk; i++) for (int j = 0; j < E_tn; j++) E_b[i * 16 + j] = ROW(p, i, st)[j]; }
static void ld_c16(const void *p, uint64_t st) { for (int i = 0; i < E_tm; i++) for (int j = 0; j < E_tn; j++) E_c[i * 16 + j] = ROW(p, i, st)[j]; }
static void st_c16(void *p, uint64_t st) { for (int i = 0; i < E_tm; i++) for (int j = 0; j < E_tn; j++) ((uint16_t *)((char *)p + (uint64_t)i * st))[j] = E_c[i * 16 + j]; }
static void mfma_hf(void) {
  for (int i = 0; i < E_tm; i++) for (int j = 0; j < E_tn; j++) {
    uint16_t c = E_c[i * 16 + j];
    for (int k = 0; k < E_tk; k++) c = d2h(h2d(E_a[i * 16 + k]) * h2d(E_b[k * 16 + j]) + h2d(c));
    E_c[i * 16 + j] = c;
  }
}
static uint64_t instret(void) { return 0; }
#define BACKEND "host-emulation"
#endif

/* flush conversion: fcvt.s.h when the compiler has zfh, else bit-twiddling (slower, would inflate instr/MAC) */
#if defined(__riscv) && defined(__riscv_zfh)
static inline float h2f_fast(uint16_t h) { _Float16 f; memcpy(&f, &h, 2); return (float)f; }
#define FLUSH_CVT "fcvt.s.h (zfh)"
#else
static inline float h2f_fast(uint16_t h) { return (float)h2d(h); }
#define FLUSH_CVT "software (NOT fcvt; instr/MAC overstated)"
#endif

static const uint16_t zero_row[256] __attribute__((aligned(64)));

/* C[M][N] (fp32) = A[M][K] * B[K][N]; A, B are fp16 bits. fp16 accumulation for KC elements, then fp32 flush. */
static void matmul_fp16_chunked(const uint16_t *A, const uint16_t *B, float *C, int M, int N, int K, int KC) {
  static uint16_t Ct[16 * 16] __attribute__((aligned(64)));
  set_sew16(); set_fp16();
  for (int m0 = 0; m0 < M;) {
    int tm = (int)ame_settile_m((uint64_t)(M - m0));
    for (int n0 = 0; n0 < N;) {
      int tn = (int)ame_settile_n((uint64_t)(N - n0));
      float acc[16 * 16];
      for (int i = 0; i < tm; i++) for (int j = 0; j < tn; j++) acc[i * 16 + j] = 0.0f;
      for (int k0 = 0; k0 < K;) {
        int kend = k0 + KC < K ? k0 + KC : K;
        ld_c16(zero_row, 0);                                   /* clear acc0 (stride-0 load of zeros) */
        while (k0 < kend) {
          int tk = (int)ame_settile_k((uint64_t)(kend - k0));
          ld_a16(A + (int64_t)m0 * K + k0, (uint64_t)K * 2);
          ld_b16(B + (int64_t)k0 * N + n0, (uint64_t)N * 2);
          mfma_hf();
          k0 += tk;
        }
        st_c16(Ct, ZSTRIDE);
        for (int i = 0; i < tm; i++) for (int j = 0; j < tn; j++) acc[i * 16 + j] += h2f_fast(Ct[i * 16 + j]);
      }
      for (int i = 0; i < tm; i++) for (int j = 0; j < tn; j++) C[(int64_t)(m0 + i) * N + n0 + j] = acc[i * 16 + j];
      n0 += tn;
    }
    m0 += tm;
  }
}

/* ---------- data ---------- */
static uint64_t rng = 0x9E3779B97F4A7C15ULL;
static double urand(void) {                                    /* [0,1) */
  rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
  return (double)((rng * 0x2545F4914F6CDD1DULL) >> 11) * (1.0 / 9007199254740992.0);
}
static double nrand(void) { double s = -6.0; for (int i = 0; i < 12; i++) s += urand(); return s; }   /* ~N(0,1) */

static float A0[M_ * KMAX], B0[KMAX * N_];                     /* original fp32 inputs */
static uint16_t A16[M_ * KMAX], B16[KMAX * N_];                /* rounded to fp16 */
static float Cq[M_ * N_];
static double Rr[M_ * N_], Ro[M_ * N_];                        /* reference on rounded / original inputs */

static void fill(int K, int positive) {
  for (int i = 0; i < M_ * K; i++) A0[i] = positive ? (float)urand() : (float)nrand();
  for (int i = 0; i < K * N_; i++) B0[i] = positive ? (float)urand() : (float)(nrand() / __builtin_sqrt((double)K));
  for (int i = 0; i < M_ * K; i++) A16[i] = d2h((double)A0[i]);
  for (int i = 0; i < K * N_; i++) B16[i] = d2h((double)B0[i]);
}
static void reference(int K) {
  for (int i = 0; i < M_; i++) for (int j = 0; j < N_; j++) {
    double sr = 0, so = 0;
    for (int k = 0; k < K; k++) { sr += h2d(A16[i * K + k]) * h2d(B16[k * N_ + j]); so += (double)A0[i * K + k] * (double)B0[k * N_ + j]; }
    Rr[i * N_ + j] = sr; Ro[i * N_ + j] = so;
  }
}
static double dabs(double x) { return x < 0 ? -x : x; }
static void report(const char *tag, int K, const char *mode, const double *ref, const double *got_d, const float *got_f,
                   uint64_t instr) {
  double maxabs = 0, se = 0, sr = 0, refmax = 0, seo = 0, sro = 0, maxo = 0;
  for (int i = 0; i < M_ * N_; i++) {
    double g = got_d ? got_d[i] : (double)got_f[i], e = dabs(g - ref[i]), eo = dabs(g - Ro[i]);
    if (e > maxabs) maxabs = e;
    if (eo > maxo) maxo = eo;
    se += e * e; sr += ref[i] * ref[i]; seo += eo * eo; sro += Ro[i] * Ro[i];
    if (dabs(ref[i]) > refmax) refmax = dabs(ref[i]);
  }
  double relrms = __builtin_sqrt(se / (sr > 0 ? sr : 1)), macs = (double)M_ * N_ * K;
  double totrms = __builtin_sqrt(seo / (sro > 0 ? sro : 1));
  /* accum err: vs the double reference on the rounded inputs; total err: vs the double reference on the ORIGINAL
   * fp32 inputs (what a PASS check against the fp32 golden would see) */
  printf("ERR K=%-4d %-8s %-10s accum: maxabs=%.2e rel_rms=%.2e | total: maxabs=%.2e rel_rms=%.2e | |ref|max=%.3g",
         K, mode, tag, maxabs, relrms, maxo, totrms, refmax);
  if (instr) printf(" | instr=%llu instr/MAC=%.4f", (unsigned long long)instr, (double)instr / macs);
  printf("\n");
}

int main(void) {
  printf("backend=%s  M=%d N=%d  flush conversion: %s\n", BACKEND, M_, N_, FLUSH_CVT);
#ifdef __riscv
  set_sew16(); set_fp16();
  printf("mtype=0x%lx\n", (unsigned long)ame_mtype());
#endif
  {
    uint64_t tm = ame_settile_m(100), tk = ame_settile_k(100), tn = ame_settile_n(100);
    printf("granted tile (e16) m=%lu k=%lu n=%lu\n", (unsigned long)tm, (unsigned long)tk, (unsigned long)tn);
  }
  static const int Ks[] = {KLIST};
  static const int chunks[] = {8, 16, 32, 64, 128, 256, 512, 0};   /* 0 = whole K (no flush) */
  for (int mode = 0; mode < 2; mode++) {
    const char *mname = mode ? "positive" : "signed";
    for (unsigned ki = 0; ki < sizeof Ks / sizeof Ks[0]; ki++) {
      int K = Ks[ki];
      fill(K, mode); reference(K);
      report("floor", K, mname, Ro, Rr, 0, 0);
      /* control: fp32 accumulation of the rounded inputs */
      for (int i = 0; i < M_; i++) for (int j = 0; j < N_; j++) {
        float s = 0; for (int k = 0; k < K; k++) s += (float)h2d(A16[i * K + k]) * (float)h2d(B16[k * N_ + j]);
        Cq[i * N_ + j] = s;
      }
      report("fp32acc", K, mname, Rr, 0, Cq, 0);
      for (unsigned ci = 0; ci < sizeof chunks / sizeof chunks[0]; ci++) {
        int KC = chunks[ci] ? chunks[ci] : K;
        if (KC > K) continue;
        char tag[24]; snprintf(tag, sizeof tag, chunks[ci] ? "chunk=%d" : "chunk=K", KC);
        uint64_t t0 = instret();
        matmul_fp16_chunked(A16, B16, Cq, M_, N_, K, KC);
        uint64_t t1 = instret();
        report(tag, K, mname, Rr, 0, Cq, t1 - t0);
      }
    }
  }
  return 0;
}
