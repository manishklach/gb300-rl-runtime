#include "arena.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

void
arena_init(KVArena *a, size_t capacity, size_t block_size) {
  a->capacity   = capacity;
  a->block_size = block_size;

  a->base = mmap(NULL, capacity, PROT_READ | PROT_WRITE,
                 MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB,
                 -1, 0);
  if (a->base == MAP_FAILED) {
    perror("arena_init: mmap hugepages");
    fprintf(stderr, "falling back to 4K pages\n");
    a->base = mmap(NULL, capacity, PROT_READ | PROT_WRITE,
                   MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (a->base == MAP_FAILED) {
      perror("arena_init: mmap 4K");
      abort();
    }
  }

  /* touch-fault every page */
  memset(a->base, 0, capacity);
  memset(a->bitmap, 0, sizeof(a->bitmap));
}

int64_t
arena_acquire(KVArena *a) {
  for (int w = 0; w < BITMAP_WORDS; w++) {
    uint64_t bits = ~a->bitmap[w];
    if (bits) {
      int b = __builtin_ctzll(bits);
      a->bitmap[w] |= (1ULL << b);
      return w * 64 + b;
    }
  }
  return -1;
}

void
arena_release(KVArena *a, int64_t idx) {
  if (idx < 0)
    return;
  int w = (int)(idx / 64);
  int b = (int)(idx % 64);
  a->bitmap[w] &= ~(1ULL << b);
}

void
arena_destroy(KVArena *a) {
  if (a->base)
    munmap(a->base, a->capacity);
  a->base = NULL;
}
