/* Hand-written fp16 ResNet-50 (TF official v2, pre-activation) on the AME matrix unit, structured like riscv-dnn's
 * test/models/resnet50/test.c: conv_base / conv_bn_relu / conv_add_bn_relu, NHWC, fp16, one image index N per run.
 * conv = scalar im2col + AME fp16 matmul (ame16_kernels.h); everything else = scalar fp16 ops (ame16_ops.h).
 * Output: riscv-dnn's `result N v0,v1,...` line (fp16 softmax bits), plus LAYER/SUMMARY profiling lines.
 * Run: Spike `pk resnet50_direct.elf [N]`, host `./resnet50_direct_host [N]` (emulated matrix unit; AME16_DUMP_DIR=dir
 * additionally writes Relu_k / add_k / MaxPool / Mean tensors for tools/check_layers.py). */
#include <stdio.h>
#include <stdlib.h>
#include "ame16_kernels.h"
#include "ame16_ops.h"
#include "weights_inc.h"

#define ACT_MAX (112 * 112 * 64)            /* = 56*56*256 = 802816 elements: largest activation */
#define COL_MAX (12544 * 147 + 100000)      /* stem im2col 1,843,968; 3x3 stage-1 1,806,336 */
#define A64 __attribute__((aligned(64)))
static f16 buf_conv[ACT_MAX] A64, buf_bn[ACT_MAX] A64, buf_R[ACT_MAX] A64, buf_T1[ACT_MAX] A64, buf_T2[ACT_MAX] A64;
static f16 buf_SC[ACT_MAX] A64, buf_A0[ACT_MAX] A64, buf_A1[ACT_MAX] A64, buf_pool[56 * 56 * 64] A64, col[COL_MAX] A64;
static f16 buf_avg[2048] A64, buf_fc[1008] A64, buf_bias16[1008] A64, buf_ba[1008] A64, buf_out16[1008] A64;
static float buf_f32a[1008], buf_f32b[1008];

/* ---------------- profiling (rdinstret on Spike; all zero on the host) ---------------- */
enum { C_IM2COL, C_MATMUL, C_BN, C_RELU, C_ADD, C_N };
typedef struct { char name[32]; int cin, cout, k, s, hin, M, N, K; uint64_t c[C_N]; } Rec;
static Rec rec[64]; static int nrec;
static uint64_t t_maxpool, t_avgpool, t_fc, t_tail, t_total;
#define TIMED(acc, stmt) do { uint64_t _t0 = ame16_instret(); stmt; (acc) += ame16_instret() - _t0; } while (0)

#ifndef __riscv
#include <string.h>
static void dump(const char *tag, int k, const f16 *p, size_t n) {
  const char *dir = getenv("AME16_DUMP_DIR"); if (!dir) return;
  char path[512]; if (k >= 0) snprintf(path, sizeof path, "%s/%s_%d.bin", dir, tag, k); else snprintf(path, sizeof path, "%s/%s.bin", dir, tag);
  FILE *f = fopen(path, "wb"); if (!f) { perror(path); exit(1); } fwrite(p, 2, n, f); fclose(f);
}
#else
static void dump(const char *tag, int k, const f16 *p, size_t n) { (void)tag; (void)k; (void)p; (void)n; }
#endif

/* conv (+ kernel w [kh][kw][cin][cout]) -> dst, recording into r. Returns hout (== wout, all layers are square). */
static int run_conv(Rec *r, f16 *dst, const f16 *src, const f16 *w, int hin, int cin, int cout, int kh, int s, int pad) {
  int hout = (hin + 2 * pad - kh) / s + 1;
  ame16_conv_cfg c = {hin, hin, cin, hout, hout, cout, kh, kh, s, s, pad, pad, 1, 1};
  r->cin = cin; r->cout = cout; r->k = kh; r->s = s; r->hin = hin; r->M = hout * hout; r->N = cout; r->K = kh * kh * cin;
  TIMED(r->c[C_IM2COL], ame16_im2col(col, src, &c));
  TIMED(r->c[C_MATMUL], ame16_matmul(col, w, dst, hout * hout, cout, kh * kh * cin, AME16_KC));
  return hout;
}
static void run_bn_relu(Rec *r, f16 *relu_out, const f16 *in, int bi, int hw, int c) {
  TIMED(r->c[C_BN], ame16_batchnorm(buf_bn, in, g_bn_alpha[bi], g_bn_beta[bi], hw, c));
  TIMED(r->c[C_RELU], ame16_relu(relu_out, buf_bn, (size_t)hw * c, 0));
  dump("Relu", bi, relu_out, (size_t)hw * c);
}

static void resnet50(const f16 *img) {
  const struct { int planes, blocks, stride; } stages[4] = {{64, 3, 1}, {128, 4, 2}, {256, 6, 2}, {512, 3, 2}};
  uint64_t t0 = ame16_instret();
  int ci = 0, bi = 0, ai = 0;
  Rec *r;

  /* stage 0: 7x7 s2 conv, maxpool 3x3 s2 SAME (pad bottom/right only), then block-1's BN+ReLU */
  r = &rec[nrec++]; snprintf(r->name, sizeof r->name, "conv2d");
  int h = run_conv(r, buf_conv, img, g_conv_w[ci++], 224, 3, 64, 7, 2, 3);                    /* 112 */
  int hp = (h + 1 - 3) / 2 + 1;                                                              /* 56 */
  TIMED(t_maxpool, ame16_maxpool(buf_pool, buf_conv, h, h, 64, hp, hp, 3, 3, 2, 2, 0, 0));
  dump("MaxPool", -1, buf_pool, (size_t)hp * hp * 64);
  run_bn_relu(r, buf_R, buf_pool, bi++, hp * hp, 64);
  h = hp; int c = 64;

  f16 *S = 0, *Aprev = 0;
  for (int st = 0; st < 4; st++) {
    const int p = stages[st].planes;
    for (int b = 0; b < stages[st].blocks; b++) {
      const int first = b == 0, sd = first ? stages[st].stride : 1;
      const f16 *Ssrc;
      if (first) {                                                                            /* projection shortcut (conv_base) */
        r = &rec[nrec++]; snprintf(r->name, sizeof r->name, "conv2d_%d.shortcut", ci);
        run_conv(r, buf_SC, buf_R, g_conv_w[ci++], h, c, 4 * p, 1, sd, 0); Ssrc = buf_SC;
      } else Ssrc = Aprev;
      r = &rec[nrec++]; snprintf(r->name, sizeof r->name, "conv2d_%d.c1", ci);                 /* 1x1 -> p, BN+ReLU */
      run_conv(r, buf_conv, buf_R, g_conv_w[ci++], h, c, p, 1, 1, 0);
      run_bn_relu(r, buf_T1, buf_conv, bi++, h * h, p);
      r = &rec[nrec++]; snprintf(r->name, sizeof r->name, "conv2d_%d.c2", ci);                 /* 3x3 (stride here) -> p, BN+ReLU */
      int h2 = run_conv(r, buf_conv, buf_T1, g_conv_w[ci++], h, p, p, 3, sd, 1);
      run_bn_relu(r, buf_T2, buf_conv, bi++, h2 * h2, p);
      r = &rec[nrec++]; snprintf(r->name, sizeof r->name, "conv2d_%d.c3+add", ci);             /* 1x1 -> 4p, add shortcut, next BN+ReLU */
      run_conv(r, buf_conv, buf_T2, g_conv_w[ci++], h2, p, 4 * p, 1, 1, 0);
      f16 *Anew = (ai & 1) ? buf_A1 : buf_A0;
      size_t n = (size_t)h2 * h2 * 4 * p;
      TIMED(r->c[C_ADD], ame16_add(Anew, buf_conv, Ssrc, n));
      dump("add", ai, Anew, n); ai++;
      run_bn_relu(r, buf_R, Anew, bi++, h2 * h2, 4 * p);
      Aprev = Anew; h = h2; c = 4 * p;
    }
  }
  (void)S;

  TIMED(t_avgpool, ame16_avgpool(buf_avg, buf_R, h, h, c, 1, 1, h, h, 1, 1, 0, 0));
  dump("Mean", -1, buf_avg, 2048);
  TIMED(t_fc, ame16_matmul(buf_avg, g_dense_w, buf_fc, 1, 1001, 2048, AME16_KC));
  TIMED(t_tail, {
    ame16_cast_f32_to_f16(buf_bias16, g_dense_b, 1001);
    ame16_add(buf_ba, buf_fc, buf_bias16, 1001);
    ame16_cast_f16_to_f32(buf_f32a, buf_ba, 1001);
    ame16_softmax(buf_f32b, buf_f32a, 1001);
    ame16_cast_f32_to_f16(buf_out16, buf_f32b, 1001);
  });
  t_total = ame16_instret() - t0;
  if (ci != 53 || bi != 49 || ai != 16) { printf("INTERNAL ERROR: conv=%d bn=%d add=%d\n", ci, bi, ai); exit(2); }
}

int main(int argc, char **argv) {
  int N = argc > 1 ? atoi(argv[1]) : 0;
  if (N < 0 || N >= NUM_PICS) { printf("image index must be in [0,%d)\n", NUM_PICS - 1); return 1; }
  printf("backend=%s KC=%d image=%d\n", AME16_BACKEND, AME16_KC, N);
  resnet50(g_pics + (size_t)N * 224 * 224 * 3);

  uint64_t sum[C_N] = {0};
  for (int i = 0; i < nrec; i++) {
    Rec *r = &rec[i];
    printf("LAYER,%d,%s,hin=%d,cin=%d,cout=%d,k=%d,s=%d,M=%d,N=%d,K=%d,im2col=%llu,matmul=%llu,bn=%llu,relu=%llu,add=%llu\n", i, r->name, r->hin, r->cin, r->cout, r->k, r->s, r->M, r->N, r->K,
           (unsigned long long)r->c[0], (unsigned long long)r->c[1], (unsigned long long)r->c[2], (unsigned long long)r->c[3], (unsigned long long)r->c[4]);
    for (int k = 0; k < C_N; k++) sum[k] += r->c[k];
  }
  printf("SUMMARY,total=%llu,im2col=%llu,matmul=%llu,bn=%llu,relu=%llu,add=%llu,maxpool=%llu,avgpool=%llu,fc_matmul=%llu,tail=%llu\n", (unsigned long long)t_total,
         (unsigned long long)sum[0], (unsigned long long)sum[1], (unsigned long long)sum[2], (unsigned long long)sum[3], (unsigned long long)sum[4],
         (unsigned long long)t_maxpool, (unsigned long long)t_avgpool, (unsigned long long)t_fc, (unsigned long long)t_tail);
  printf("result %d ", N);
  for (int i = 0; i < 1001; i++) printf("%d,", buf_out16[i]);
  printf("\nEnd\n");
  return 0;
}
