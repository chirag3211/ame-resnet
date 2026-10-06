#include <stdio.h>
#include <stdint.h>
#include "ame_insn.h"
static float A[64] __attribute__((aligned(64)));
static float B[64] __attribute__((aligned(64)));
static float Z[64] __attribute__((aligned(64)));
static float D[64] __attribute__((aligned(64)));

#define MEAS(NAME, BODY) do { unsigned long a, b; \
  __asm__ volatile("rdinstret %0\n\t" BODY "\n\trdinstret %1" \
                   : "=&r"(a), "=&r"(b) : "r"(p), "r"(s) : "memory"); \
  printf("  %-10s delta=%lu\n", NAME, b - a); } while (0)

static void measure(const char *tag) {
  uint64_t s = 32; const float *p;
  unsigned long a, b, g;
  printf("-- %s\n", tag);
  __asm__ volatile("rdinstret %0\n\trdinstret %1" : "=&r"(a), "=&r"(b));
  printf("  %-10s delta=%lu\n", "baseline", b - a);
  p = A; MEAS("mlae32", ".insn r 0x77, 2, 0x02, x0, %2, %3");
  p = B; MEAS("mlbe32", ".insn r 0x77, 2, 0x04, x1, %2, %3");
  p = Z; MEAS("mlce32", ".insn r 0x77, 2, 0x00, x0, %2, %3");
  p = A; MEAS("mfma", ".insn r 0x77, 2, 0x11, x16, x0, x1");
  p = D; MEAS("msce32", ".insn r 0x77, 2, 0x01, x0, %2, %3");
  uint64_t req = 100;
  __asm__ volatile("rdinstret %0\n\t.insn r 0x77, 6, 0x02, %2, %3, x0\n\trdinstret %1"
                   : "=&r"(a), "=&r"(b), "=&r"(g) : "r"(req));
  printf("  %-10s delta=%lu\n", "settilek", b - a);
}

int main(void) {
  for (int i = 0; i < 64; i++) { A[i] = i; B[i] = i; Z[i] = 0; D[i] = -1; }
  ame_set_sew32(); ame_set_fp32();
  uint64_t tm = ame_settile_m(100), tk = ame_settile_k(100), tn = ame_settile_n(100);
  printf("granted m=%lu k=%lu n=%lu\n", (unsigned long)tm, (unsigned long)tk, (unsigned long)tn);
  measure("full tile");
  tm = ame_settile_m(2); tk = ame_settile_k(1); tn = ame_settile_n(2);
  printf("granted m=%lu k=%lu n=%lu\n", (unsigned long)tm, (unsigned long)tk, (unsigned long)tn);
  measure("small tile");
  return 0;
}
