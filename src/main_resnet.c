/* Phase C driver -- TEMPLATE. Assumes torch-mlir exported `forward` with the
 * weights inlined as constants, i.e.  forward(input[1,3,H,W]) -> logits[1,1000],
 * called through the C interface. If the real signature differs (e.g. weights
 * passed as arguments), adjust this file once we have seen the IR.
 *
 * Build: make resnet-spike  (see README)   -DINPUT_H / -DINPUT_W set by the Makefile.
 * The input is embedded with `objcopy -I binary` from build/input.bin.
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "ame_kernel.h"

#ifndef INPUT_H
#define INPUT_H 224
#endif
#ifndef INPUT_W
#define INPUT_W 224
#endif

typedef struct { float *alloc, *align; int64_t off, sizes[4]; int64_t strides[4]; } MemRef4D;
typedef struct { float *alloc, *align; int64_t off, sizes[2]; int64_t strides[2]; } MemRef2D;

/* with --llvm-request-c-wrappers: result memref returned via pointer (first arg) */
extern void _mlir_ciface_forward(MemRef2D *out, MemRef4D *in);

extern const unsigned char _binary_build_input_bin_start[];

static float g_in[1 * 3 * INPUT_H * INPUT_W] __attribute__((aligned(64)));
static float g_out[1000] __attribute__((aligned(64)));

int main(void) {
  memcpy(g_in, _binary_build_input_bin_start, sizeof(g_in));
  MemRef4D in = {g_in, g_in, 0, {1, 3, INPUT_H, INPUT_W},
                 {3 * INPUT_H * INPUT_W, INPUT_H * INPUT_W, INPUT_W, 1}};
  MemRef2D out = {g_out, g_out, 0, {1, 1000}, {1000, 1}};

  extern unsigned long ame_instret_in_matmul, ame_ksteps, ame_otiles, ame_instret_in_copy, ame_copy_calls;
  for (int run = 0; run < 2; run++) {
    ame_stats_reset();
    ame_instret_in_matmul = ame_ksteps = ame_otiles = ame_instret_in_copy = ame_copy_calls = 0;
    unsigned long ir0, ir1;
    __asm__ volatile("rdinstret %0" : "=r"(ir0));
    _mlir_ciface_forward(&out, &in);
    __asm__ volatile("rdinstret %0" : "=r"(ir1));
    fprintf(stderr, "RUN%d INSTRET forward=%lu in_matmul=%lu in_copy=%lu copy_calls=%lu ksteps=%lu otiles=%lu\n",
            run, ir1 - ir0, ame_instret_in_matmul, ame_instret_in_copy, ame_copy_calls, ame_ksteps, ame_otiles);
  }

  /* the callee may return its own buffer: read through out.align + out.off */
  const float *logits = out.align + out.off;
  int top = 0;
  for (int i = 0; i < 1000; i++) {
    uint32_t bits;
    memcpy(&bits, &logits[i], 4);
    printf("LOGIT %d 0x%08x\n", i, (unsigned)bits);
    if (logits[i] > logits[top]) top = i;
  }
  const ame_stats_t *s = ame_stats_get();
  printf("TOP1 %d\n", top);
  printf("STATS backend=%s matmul_calls=%lu macs=%lu\n", ame_backend_name(),
         (unsigned long)s->calls, (unsigned long)s->macs);
  return 0;
}
