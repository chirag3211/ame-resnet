#ifndef WEIGHTS_INC_H
#define WEIGHTS_INC_H
#include <stdint.h>
#define NUM_PICS 69
extern const uint16_t *const g_conv_w[53];      /* conv2d, conv2d_1 .. conv2d_52 kernels, fp16 [kh][kw][cin][cout] */
extern const uint16_t *const g_bn_alpha[49];    /* folded batchnorm scale, fp16 */
extern const uint16_t *const g_bn_beta[49];     /* folded batchnorm shift, fp16 */
extern const uint16_t *const g_dense_w;         /* fp16 [2048][1001] */
extern const float *const g_dense_b;            /* fp32 [1001] */
extern const uint16_t *const g_pics;            /* fp16 [69][224][224][3] */
#endif
