/* Software model of a tile-matrix programming model (NOT the real AME encodings).
 * Replace the primitives with .insn wrappers once the spec + Spike fork exist.
 * Primitives take the ACTIVE tile extent (m,n,k <= tile size), so edges need no padding. */
#ifndef AME_MODEL_H
#define AME_MODEL_H
#include <stdint.h>

#ifndef AME_TM
#define AME_TM 8
#endif
#ifndef AME_TN
#define AME_TN 8
#endif
#ifndef AME_TK_F32
#define AME_TK_F32 8
#endif
#ifndef AME_TK_I8
#define AME_TK_I8 16
#endif

#define AME_DEFINE_TILE(SUF, TA, TB, TC, TK) \
  static TC acc_##SUF[AME_TM][AME_TN]; \
  static TA ta_##SUF[AME_TM][TK]; \
  static TB tb_##SUF[TK][AME_TN]; \
  static inline void ame_zero_acc_##SUF(int m, int n) { \
    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) acc_##SUF[i][j] = 0; \
  } \
  static inline void ame_load_acc_##SUF(const TC *p, int64_t ld, int m, int n) { \
    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) acc_##SUF[i][j] = p[i*ld+j]; \
  } \
  static inline void ame_load_a_##SUF(const TA *p, int64_t ld, int m, int k) { \
    for (int i = 0; i < m; i++) for (int l = 0; l < k; l++) ta_##SUF[i][l] = p[i*ld+l]; \
  } \
  static inline void ame_load_b_##SUF(const TB *p, int64_t ld, int k, int n) { \
    for (int l = 0; l < k; l++) for (int j = 0; j < n; j++) tb_##SUF[l][j] = p[l*ld+j]; \
  } \
  static inline void ame_mac_##SUF(int m, int n, int k) { \
    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) { \
      TC s = acc_##SUF[i][j]; \
      for (int l = 0; l < k; l++) s += (TC)ta_##SUF[i][l] * (TC)tb_##SUF[l][j]; \
      acc_##SUF[i][j] = s; \
    } \
  } \
  static inline void ame_store_acc_##SUF(TC *p, int64_t ld, int m, int n) { \
    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) p[i*ld+j] = acc_##SUF[i][j]; \
  }

AME_DEFINE_TILE(f32, float,  float,  float,   AME_TK_F32)
AME_DEFINE_TILE(i8,  int8_t, int8_t, int32_t, AME_TK_I8)
#endif
