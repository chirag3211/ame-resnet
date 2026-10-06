/* MLIR's default calling convention for an external function expands every
 * memref argument into its descriptor fields:
 *   (allocated_ptr, aligned_ptr, offset, size0, size1, stride0, stride1)
 * for rank 2. The generated shims (build/generated_shims.c) take 3 such
 * memrefs (A, B, C) and forward to the implementations in memref_shims.c. */
#ifndef MEMREF_SHIMS_H
#define MEMREF_SHIMS_H
#include <stdint.h>

#define MR2_PARAMS(p) void *p##_alloc, void *p##_align, int64_t p##_off, \
                      int64_t p##_s0, int64_t p##_s1, int64_t p##_st0, int64_t p##_st1
#define MR2_ARGS(p) p##_alloc, p##_align, p##_off, p##_s0, p##_s1, p##_st0, p##_st1

void ame_matmul_f32_memref(MR2_PARAMS(a), MR2_PARAMS(b), MR2_PARAMS(c));
void ame_matmul_i8_memref(MR2_PARAMS(a), MR2_PARAMS(b), MR2_PARAMS(c));

#define AME_SHIM(NAME, IMPL) \
  void NAME(MR2_PARAMS(a), MR2_PARAMS(b), MR2_PARAMS(c)) { \
    IMPL(MR2_ARGS(a), MR2_ARGS(b), MR2_ARGS(c)); }

#endif
