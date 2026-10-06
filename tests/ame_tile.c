#include <stdio.h>
#include <stdint.h>
#include "ame_insn.h"

static float A[8*8] __attribute__((aligned(64)));
static float B[8*8] __attribute__((aligned(64)));
static float Z[8*8] __attribute__((aligned(64)));
static float D[8*8] __attribute__((aligned(64)));

int main(void) {
  printf("mlenb=%lu mrlenb=%lu mamul=%lu\n",
         (unsigned long)ame_mlenb(), (unsigned long)ame_mrlenb(), (unsigned long)ame_mamul());
  ame_set_sew32();
  ame_set_fp32();
  printf("mtype=0x%lx\n", (unsigned long)ame_mtype());
  uint64_t tm = ame_settile_m(100), tk = ame_settile_k(100), tn = ame_settile_n(100);
  printf("granted m=%lu k=%lu n=%lu\n", (unsigned long)tm, (unsigned long)tk, (unsigned long)tn);

  for (int i = 0; i < 8; i++) for (int k = 0; k < 8; k++) A[i*8+k] = (float)((i+1)*(k+1));
  for (int k = 0; k < 8; k++) for (int j = 0; j < 8; j++) B[k*8+j] = (float)(k + 2*j + 1);
  for (int i = 0; i < 64; i++) { Z[i] = 0.0f; D[i] = -1.0f; }

  ame_ld_a32(A, 32);
  ame_ld_b32(B, 32);
  ame_ld_c32(Z, 32);
  ame_mfma_f();
  ame_st_c32(D, 32);

  int bad = 0, outside = 0;
  for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) {
    if (i < (int)tm && j < (int)tn) {
      float ref = 0.0f;
      for (int k = 0; k < (int)tk; k++) ref += A[i*8+k] * B[k*8+j];
      if (D[i*8+j] != ref) { if (bad < 4) printf("MISMATCH [%d][%d] got %g want %g\n", i, j, D[i*8+j], ref); bad++; }
    } else if (D[i*8+j] != -1.0f) outside++;
  }
  printf("bad=%d outside_writes=%d\n", bad, outside);
  printf(bad == 0 && outside == 0 ? "PASS tile\n" : "FAIL tile\n");
  return 0;
}
