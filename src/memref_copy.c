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

  /* Merge the innermost dims that are contiguous in BOTH src and dst (size-1 dims never matter)
   * into one run, then memmove one run at a time instead of one element at a time. */
  int64_t run = 1, k = rank - 1;
  while (k >= 0) {
    if (sz[k] == 1) { k--; continue; }
    if (ss[k] == run && ds[k] == run) { run *= sz[k]; k--; } else break;
  }
  const int64_t n = k + 1;               /* outer dims still to iterate: 0..n-1 */
  const size_t runBytes = (size_t)(run * elemSize);
  int64_t idx[8] = {0};
  int64_t so = 0, doff = 0;
  for (;;) {
    memmove(dp + doff * elemSize, sp + so * elemSize, runBytes);
    int64_t j = n - 1;
    for (; j >= 0; j--) {
      so += ss[j]; doff += ds[j];
      if (++idx[j] < sz[j]) break;
      so -= ss[j] * sz[j]; doff -= ds[j] * sz[j]; idx[j] = 0;
    }
    if (j < 0) break;
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
