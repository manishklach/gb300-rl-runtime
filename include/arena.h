#pragma once
#include <stdint.h>
#include <stddef.h>

#define BITMAP_WORDS ((1UL << 20) / 64)  /* supports up to 1M blocks */

/* Hugepage-backed KV block arena.
 *
 * All memory is pre-faulted at init so the GPU never hits a page fault.
 * Allocation is O(1) via a word-level bitmap with CLZ. */
typedef struct {
  uint8_t  *base;
  size_t    capacity;
  size_t    block_size;
  uint64_t  bitmap[BITMAP_WORDS];
} KVArena;

/* Initialize arena: mmap with MAP_HUGETLB, touch every page, zero bitmap.
 * Aborts on failure. */
void  arena_init(KVArena *a, size_t capacity, size_t block_size);

/* Acquire one block.  Returns block index (0-based) or -1 on exhaustion. */
int64_t arena_acquire(KVArena *a);

/* Release a previously acquired block. */
void    arena_release(KVArena *a, int64_t idx);

/* Return a pointer to the start of block idx. */
static inline uint8_t *
arena_block_ptr(KVArena *a, int64_t idx) {
  return a->base + idx * a->block_size;
}

/* Destroy arena (munmap). */
void  arena_destroy(KVArena *a);
