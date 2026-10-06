/* Phase B: call the MLIR-compiled @matmul (which calls back into ame_matmul_f32). */
#include <stdio.h>
#include <stdint.h>
#include "ame_kernel.h"

typedef struct { float *alloc, *align; int64_t off, sizes[2], strides[2]; } MemRef2D;
extern void _mlir_ciface_matmul(MemRef2D *A, MemRef2D *B, MemRef2D *C);

#define M 8
#define K 16
#define N 12

int main(void) {
  static float A[M * K], B[K * N], C[M * N], R[M * N];
  for (int i = 0; i < M * K; i++) A[i] = (float)((i * 5) % 7) - 3.0f;
  for (int i = 0; i < K * N; i++) B[i] = (float)((i * 3) % 5) - 2.0f;
  for (int i = 0; i < M * N; i++) { C[i] = 0.5f; R[i] = 0.5f; }
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
      for (int k = 0; k < K; k++) R[i * N + j] += A[i * K + k] * B[k * N + j];

  MemRef2D mA = {A, A, 0, {M, K}, {K, 1}};
  MemRef2D mB = {B, B, 0, {K, N}, {N, 1}};
  MemRef2D mC = {C, C, 0, {M, N}, {N, 1}};
  ame_stats_reset();
  _mlir_ciface_matmul(&mA, &mB, &mC);

  int bad = 0;
  for (int i = 0; i < M * N; i++) {
    float d = C[i] - R[i];
    if (d < 0) d = -d;
    if (d > 1e-4f) bad++;
  }
  const ame_stats_t *s = ame_stats_get();
  printf("backend=%s calls=%lu macs=%lu (expect calls=1 macs=%d)\n",
         ame_backend_name(), (unsigned long)s->calls, (unsigned long)s->macs, M * N * K);
  if (s->calls != 1) bad++;
  printf(bad ? "FAIL mlir matmul\n" : "PASS mlir matmul\n");
  return bad ? 1 : 0;
}
