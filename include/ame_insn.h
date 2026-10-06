/* AME instruction wrappers via .insn r (opcode OP-M32 = 0x77).
 * Encodings derived from Spike's MATCH_* values; UNVERIFIED until tools/dasm_check.py passes.
 * Fixed tile assignment for now: A -> tr0, B -> tr1, C -> acc0. */
#ifndef AME_INSN_H
#define AME_INSN_H
#include <stdint.h>

/* read-only CSRs */
static inline uint64_t ame_mlenb(void)  { uint64_t v; __asm__ volatile("csrr %0, 0xc44" : "=r"(v)); return v; }
static inline uint64_t ame_mrlenb(void) { uint64_t v; __asm__ volatile("csrr %0, 0xc45" : "=r"(v)); return v; }
static inline uint64_t ame_mamul(void)  { uint64_t v; __asm__ volatile("csrr %0, 0xc46" : "=r"(v)); return v; }
static inline uint64_t ame_mtype(void)  { uint64_t v; __asm__ volatile("csrr %0, 0xc40" : "=r"(v)); return v; }

/* mtype field setters: funct3=6, funct7=1, rs1 slot = field id, rs2 slot = value */
static inline void ame_set_sew32(void) { __asm__ volatile(".insn r 0x77, 6, 0x01, x0, x0, x2" ::: "memory"); } /* msetsew e32 */
static inline void ame_set_sew8(void)  { __asm__ volatile(".insn r 0x77, 6, 0x01, x0, x0, x0" ::: "memory"); } /* msetsew e8  */
static inline void ame_set_fp32(void)  { __asm__ volatile(".insn r 0x77, 6, 0x01, x0, x8, x1" ::: "memory"); } /* msetfp fp32 */
static inline void ame_set_int8(void)  { __asm__ volatile(".insn r 0x77, 6, 0x01, x0, x2, x1" ::: "memory"); } /* msetint int8 */

/* tile shape: returns the size actually granted */
#define AME_SETTILE(NAME, F3) \
  static inline uint64_t NAME(uint64_t req) { uint64_t g; \
    __asm__ volatile(".insn r 0x77, " #F3 ", 0x02, %0, %1, x0" : "=r"(g) : "r"(req)); return g; }
AME_SETTILE(ame_settile_m, 5)
AME_SETTILE(ame_settile_k, 6)
AME_SETTILE(ame_settile_n, 4)

/* loads/stores: rs1 = base address, rs2 = row stride in BYTES */
#define AME_MEM(NAME, F7, F3, RD) \
  static inline void NAME(const void *p, uint64_t stride) { \
    __asm__ volatile(".insn r 0x77, " #F3 ", " #F7 ", " #RD ", %0, %1" :: "r"(p), "r"(stride) : "memory"); }
AME_MEM(ame_ld_a32, 0x02, 2, x0)   /* mlae32.m tr0   */
AME_MEM(ame_ld_b32, 0x04, 2, x1)   /* mlbe32.m tr1   */
AME_MEM(ame_ld_c32, 0x00, 2, x0)   /* mlce32.m acc0  */
AME_MEM(ame_st_c32, 0x01, 2, x0)   /* msce32.m acc0  */
AME_MEM(ame_ld_a8,  0x02, 0, x0)   /* mlae8.m  tr0   */
AME_MEM(ame_ld_b8,  0x04, 0, x1)   /* mlbe8.m  tr1   */

/* multiply-accumulate: acc0 += tr0 * tr1 ; rd slot x16 = ma=1, md=acc0 ; rs1 = ms1, rs2 = ms2 */
static inline void ame_mfma_f(void) { __asm__ volatile(".insn r 0x77, 2, 0x11, x16, x0, x1" ::: "memory"); } /* mfma.f.mm */
static inline void ame_mqma_b(void) { __asm__ volatile(".insn r 0x77, 0, 0x14, x16, x16, x1" ::: "memory"); } /* mqma.b.mm (sn=1 -> rs1 slot +16) */
#endif
