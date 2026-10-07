/* fp16 accumulator probe for the riscv-stc AME Spike. Put in tests/, build:
 *   riscv64-unknown-elf-gcc -O2 -march=rv64gc -mabi=lp64d -Iinclude tests/ame_fp16_probe.c -o build/ame_fp16_probe.elf
 *   $HOME/riscv-stc/bin/spike --isa=rv64imafdcv_zicntr_matrix $HOME/riscv/riscv64-unknown-elf/bin/pk build/ame_fp16_probe.elf
 * FP16VAL = value written to the fp16 mtype field (check MTYPE_FP16 in matrix_unit.h). Default 1; override with -DFP16VAL=n. */
#include <stdio.h>
#include <stdint.h>
#include "ame_insn.h"
#ifndef FP16VAL
#define FP16VAL 1
#endif
#define STR2(x) #x
#define STR(x) STR2(x)
static inline void set_sew16(void) { __asm__ volatile(".insn r 0x77, 6, 0x01, x0, x0, x1" ::: "memory"); }
static inline void set_fp16(void)  { __asm__ volatile(".insn r 0x77, 6, 0x01, x0, x7, x" STR(FP16VAL) ::: "memory"); }
AME_MEM(ld_a16, 0x02, 1, x0)   /* mlae16.m tr0  */
AME_MEM(ld_b16, 0x04, 1, x1)   /* mlbe16.m tr1  */
AME_MEM(ld_c16, 0x00, 1, x0)   /* mlce16.m acc0 */
AME_MEM(st_c16, 0x01, 1, x0)   /* msce16.m acc0 */
static inline void mfma_hf(void) { __asm__ volatile(".insn r 0x77, 1, 0x11, x16, x0, x1" ::: "memory"); }

static uint16_t A[32*32] __attribute__((aligned(64))), B[32*32] __attribute__((aligned(64)));
static uint16_t Z[32*32] __attribute__((aligned(64))), D[32*32] __attribute__((aligned(64)));

static float h2f(uint16_t h) {            /* fp16 bits -> float (normals, subnormals, inf) */
  int s = h >> 15, e = (h >> 10) & 31, m = h & 1023; float v;
  if (e == 0) v = m * (1.0f / 16777216.0f);
  else if (e == 31) v = m ? 0.0f / 0.0f : 1e30f;
  else { v = 1.0f + m / 1024.0f; for (int i = e; i > 15; i--) v *= 2; for (int i = e; i < 15; i++) v *= 0.5f; }
  return s ? -v : v;
}

int main(void) {
  set_sew16(); set_fp16();
  printf("mtype=0x%lx (FP16VAL=" STR(FP16VAL) ")\n", (unsigned long)ame_mtype());
  uint64_t tm = ame_settile_m(100), tk = ame_settile_k(100), tn = ame_settile_n(100);
  printf("granted m=%lu k=%lu n=%lu (e16)\n", (unsigned long)tm, (unsigned long)tk, (unsigned long)tn);
  for (int i = 0; i < 32*32; i++) { A[i] = B[i] = 0x3C00; Z[i] = 0; D[i] = 0xFFFF; }   /* 0x3C00 = 1.0 */
  uint64_t stride = 64;
  int totals[] = {tk, 1024, 2048, 2560, 4096};
  for (int t = 0; t < 5; t++) {
    int iters = totals[t] / (int)tk; if (iters < 1) iters = 1;
    ld_c16(Z, stride);
    for (int it = 0; it < iters; it++) { ld_a16(A, stride); ld_b16(B, stride); mfma_hf(); }
    st_c16(D, stride);
    printf("K=%5d: C[0][0]=%g (exact %d)\n", iters * (int)tk, h2f(D[0]), iters * (int)tk);
  }
  return 0;
}
