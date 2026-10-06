#include <stdio.h>
#include <stdint.h>
#include "ame_insn.h"
static float big[16 * 1024] __attribute__((aligned(4096)));   /* 16 untouched pages in .bss */
int main(void) {
  ame_set_sew32(); ame_set_fp32();
  ame_settile_m(100); ame_settile_k(100); ame_settile_n(100);
  uint64_t s = 16;
  for (int pass = 0; pass < 2; pass++) {
    for (int i = 0; i < 4; i++) {
      const float *p = big + i * 1024;
      unsigned long a, b;
      __asm__ volatile("rdinstret %0\n\t.insn r 0x77, 2, 0x02, x0, %2, %3\n\trdinstret %1"
                       : "=&r"(a), "=&r"(b) : "r"(p), "r"(s) : "memory");
      printf("%s page %d: mlae32 delta=%lu\n", pass ? "warm" : "cold", i, b - a);
    }
  }
  return 0;
}
