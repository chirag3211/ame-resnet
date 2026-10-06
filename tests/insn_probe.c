#include <stdio.h>
#include "ame_insn.h"
__attribute__((used, noinline)) void all_insns(float *a, float *b, float *c, signed char *s) {
  ame_set_sew32(); ame_set_sew8(); ame_set_fp32(); ame_set_int8();
  (void)ame_settile_m(4); (void)ame_settile_k(4); (void)ame_settile_n(4);
  ame_ld_a32(a, 16); ame_ld_b32(b, 16); ame_ld_c32(c, 16); ame_st_c32(c, 16);
  ame_ld_a8(s, 4); ame_ld_b8(s, 4);
  ame_mfma_f(); ame_mqma_b();
}
int main(void) { printf("mlenb=%lu mrlenb=%lu mamul=%lu\n",
  (unsigned long)ame_mlenb(), (unsigned long)ame_mrlenb(), (unsigned long)ame_mamul()); return 0; }
