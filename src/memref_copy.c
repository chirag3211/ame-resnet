/* Minimal replacement for MLIR's runtime memrefCopy (strided, any rank <= 8). */
#include <stdint.h>
#include <string.h>

typedef struct { int64_t rank; void *desc; } UnrankedMemRef;
typedef struct { char *alloc; char *align; int64_t offset; int64_t dims[]; } MemRefDesc;

static void memrefCopy_impl(int64_t elemSize, UnrankedMemRef *src, UnrankedMemRef *dst) {
  const MemRefDesc *s = (const MemRefDesc *)src->desc;
  MemRefDesc *d = (MemRefDesc *)dst->desc;
  int64_t rank = src->rank;
  if (rank > 8) __builtin_trap();

  const int64_t *sz = s->dims, *ss = s->dims + rank, *ds = d->dims + rank;
  int64_t total = 1;
  for (int64_t i = 0; i < rank; i++) total *= sz[i];
  if (total == 0) return;

  const char *sp = s->align + s->offset * elemSize;
  char *dp = d->align + d->offset * elemSize;
  if (rank == 0) { memmove(dp, sp, (size_t)elemSize); return; }

  int64_t idx[8] = {0};
  for (;;) {
    int64_t so = 0, doff = 0;
    for (int64_t i = 0; i < rank; i++) { so += idx[i] * ss[i]; doff += idx[i] * ds[i]; }
    memmove(dp + doff * elemSize, sp + so * elemSize, (size_t)elemSize);
    int64_t k = rank - 1;
    while (k >= 0 && ++idx[k] == sz[k]) idx[k--] = 0;
    if (k < 0) break;
  }
}

unsigned long ame_instret_in_copy, ame_copy_calls;
void memrefCopy(int64_t elemSize, UnrankedMemRef *src, UnrankedMemRef *dst) {
  unsigned long t0, t1;
  __asm__ volatile("rdinstret %0" : "=r"(t0));
  memrefCopy_impl(elemSize, src, dst);
  __asm__ volatile("rdinstret %0" : "=r"(t1));
  ame_instret_in_copy += t1 - t0;
  ame_copy_calls++;
}
