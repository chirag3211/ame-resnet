/* Phase A: kernel API vs. an independent double-precision / exact reference.
 * Runs on the host (make test-native) and on Spike (make test-spike). */
#include <stdio.h>
#include <stdint.h>
#include "ame_kernel.h"

#define MAXD 80
static float    fA[MAXD * MAXD], fB[MAXD * MAXD], fC[MAXD * MAXD];
static int8_t   iA[MAXD * MAXD], iB[MAXD * MAXD];
static int32_t  iC[MAXD * MAXD];

static uint32_t rng = 12345u;
static uint32_t next(void) { rng = rng * 1664525u + 1013904223u; return rng >> 8; }
static float rnd_f(void) { return ((int)(next() % 2001) - 1000) / 1000.0f; }  /* [-1,1] */
static int8_t rnd_i(void) { return (int8_t)((int)(next() % 255) - 127); }

static int test_f32(int M, int N, int K, int acc) {
  int lda = K + 3, ldb = N + 2, ldc = N + 5;      /* deliberately non-contiguous */
  if (lda * M > MAXD * MAXD || ldb * K > MAXD * MAXD || ldc * M > MAXD * MAXD) return 0;
  for (int i = 0; i < MAXD * MAXD; i++) { fA[i] = rnd_f(); fB[i] = rnd_f(); fC[i] = rnd_f(); }
  static double ref[MAXD * MAXD];
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      double s = acc ? fC[i * ldc + j] : 0.0;
      for (int k = 0; k < K; k++) s += (double)fA[i * lda + k] * (double)fB[k * ldb + j];
      ref[i * N + j] = s;
    }
  ame_matmul_f32(fA, fB, fC, M, N, K, lda, ldb, ldc, acc);
  int bad = 0;
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      double d = fC[i * ldc + j] - ref[i * N + j];
      if (d < 0) d = -d;
      if (d > 1e-3) bad++;
    }
  if (bad) printf("FAIL f32 M=%d N=%d K=%d acc=%d (%d bad)\n", M, N, K, acc, bad);
  return bad;
}

static int test_i8(int M, int N, int K, int acc) {
  int lda = K + 1, ldb = N + 4, ldc = N + 1;
  if (lda * M > MAXD * MAXD || ldb * K > MAXD * MAXD || ldc * M > MAXD * MAXD) return 0;
  for (int i = 0; i < MAXD * MAXD; i++) { iA[i] = rnd_i(); iB[i] = rnd_i(); iC[i] = (int32_t)(next() % 1000); }
  static int64_t ref[MAXD * MAXD];
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      int64_t s = acc ? iC[i * ldc + j] : 0;
      for (int k = 0; k < K; k++) s += (int64_t)iA[i * lda + k] * iB[k * ldb + j];
      ref[i * N + j] = s;
    }
  ame_matmul_i8(iA, iB, iC, M, N, K, lda, ldb, ldc, acc);
  int bad = 0;
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
      if (iC[i * ldc + j] != ref[i * N + j]) bad++;
  if (bad) printf("FAIL i8  M=%d N=%d K=%d acc=%d (%d bad)\n", M, N, K, acc, bad);
  return bad;
}

int main(void) {
  /* includes sizes that are NOT multiples of any plausible tile size */
  static const int sz[][3] = {
    {1,1,1}, {1,7,3}, {5,1,9}, {4,4,4}, {8,8,8}, {16,16,16}, {17,16,33},
    {15,31,7}, {32,32,32}, {33,17,5}, {64,64,16}, {3,64,64}, {64,3,64}
  };
  int fails = 0, n = sizeof(sz) / sizeof(sz[0]);
  ame_stats_reset();
  for (int t = 0; t < n; t++)
    for (int acc = 0; acc < 2; acc++) {
      fails += test_f32(sz[t][0], sz[t][1], sz[t][2], acc) != 0;
      fails += test_i8 (sz[t][0], sz[t][1], sz[t][2], acc) != 0;
    }
  const ame_stats_t *s = ame_stats_get();
  printf("backend=%s calls=%lu macs=%lu\n", ame_backend_name(),
         (unsigned long)s->calls, (unsigned long)s->macs);
  printf(fails ? "FAIL (%d cases)\n" : "PASS\n", fails);
  return fails ? 1 : 0;
}
