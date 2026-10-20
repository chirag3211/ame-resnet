/* fp16 AME matmul with fp32 accumulation (widening), the same instruction family riscv-dnn's
 * matmul_batch1 compiles to.  Spec v0.5 (riscv-stc Spike `matrix` branch).
 *
 * Every encoding below was decoded from the instruction words that the riscv-stc LLVM fork emitted
 * for riscv-dnn's kernel (llvm-objdump of smoke.o) and cross-checked against Spike's disassembler.
 * Fixed tile assignment: A -> tr0, B -> tr1, accumulator -> acc0, narrowed fp16 result -> acc1.
 * Needs ame_insn.h (tile-size setters + the AME_MEM macro). */
#ifndef AME_F16_H
#define AME_F16_H
#include <stdint.h>
#include "ame_insn.h"

/* msettypei / msettypehi: funct3 = 4 / 5, funct7 = 0x01.  The 10-bit immediate occupies the rs1 slot
 * (values < 32 only need rs1).  rd = x0, so the previous mtype is discarded. */
#define AME_MTYPEI(NAME, IMM)  static inline void NAME(void) { \
    __asm__ volatile(".insn r 0x77, 4, 0x01, x0, x" #IMM ", x0" ::: "memory"); }
#define AME_MTYPEHI(NAME, IMM) static inline void NAME(void) { \
    __asm__ volatile(".insn r 0x77, 5, 0x01, x0, x" #IMM ", x0" ::: "memory"); }
AME_MTYPEI (ame_mtypei_e16, 1)            /* msettypei  e16           */
AME_MTYPEI (ame_mtypei_e32, 2)            /* msettypei  e32           */
AME_MTYPEHI(ame_mtypehi_fp16, 1)          /* msettypehi bu, fp16      */
AME_MTYPEHI(ame_mtypehi_fp32, 4)          /* msettypehi bu, fp32      */
AME_MTYPEHI(ame_mtypehi_fp16_fp32, 5)     /* msettypehi bu, fp16, fp32  (fp16 in, fp32 out: widening) */

/* msetsew: changes only the element-width bits of mtype and keeps the other fields (unlike msettypei, which
 * replaces the whole of mtype and so clears the high fields; msettypehi ORs bits into them).  Same encoding
 * family as ame_set_sew32 in ame_insn.h: funct3 = 6, funct7 = 0x01, value in the rs2 slot (e16 = 1, e32 = 2). */
static inline void ame_set_sew16(void) { __asm__ volatile(".insn r 0x77, 6, 0x01, x0, x0, x1" ::: "memory"); } /* msetsew e16 */

/* loads/stores: rs1 = base address, rs2 = row stride in BYTES */
AME_MEM(ame_ld_a16,     0x02, 1, x0)      /* mlae16.m tr0          */
AME_MEM(ame_ld_b16,     0x04, 1, x1)      /* mlbe16.m tr1          */
AME_MEM(ame_st_c16_a1,  0x01, 1, x1)      /* msce16.m acc1         */

static inline void ame_mfwma_f16(void)  { __asm__ volatile(".insn r 0x77, 4, 0x13, x16, x0, x1" ::: "memory"); } /* mfwma.mm acc0, tr0, tr1 */
static inline void ame_mfsub_clear(void){ __asm__ volatile(".insn r 0x77, 4, 0x15, x0, x16, x0" ::: "memory"); } /* mfsub.mm acc0, acc0, acc0 */
static inline void ame_mfncvt_a1_a0(void){ __asm__ volatile(".insn r 0x77, 4, 0x33, x1, x0, x6" ::: "memory"); }  /* mfncvt.f.fw.m acc1, acc0 */

/* C[MxN] = A[MxK] * B[KxN]; A, B, C are raw IEEE fp16 bit patterns; ld* in ELEMENTS; overwrites C (no accumulate).
 * faithful : the exact type-switching sequence the fork's compiler emits for riscv-dnn's kernel.
 * hoisted  : the same arithmetic with redundant mtype writes removed (see src/ame_hw_f16.c). */
void ame_hw_matmul_f16w_faithful(const uint16_t *A, const uint16_t *B, uint16_t *C,
                                 int64_t M, int64_t N, int64_t K, int64_t lda, int64_t ldb, int64_t ldc);
void ame_hw_matmul_f16w_hoisted (const uint16_t *A, const uint16_t *B, uint16_t *C,
                                 int64_t M, int64_t N, int64_t K, int64_t lda, int64_t ldb, int64_t ldc);
#endif
