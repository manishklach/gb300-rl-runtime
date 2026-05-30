#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#define HP_INIT_MAGIC 0xDEADBEEF

typedef struct {
    uint64_t malloc_count;
    uint64_t calloc_count;
    uint64_t realloc_count;
    uint64_t free_count;
    uint64_t cuda_malloc_count;
    uint64_t cuda_free_count;
    uint64_t page_fault_count;
    int      active;
} HotpathGuard;

void hp_guard_init(HotpathGuard *g);
void hp_guard_activate(HotpathGuard *g);
void hp_guard_deactivate(HotpathGuard *g);

void hp_track_malloc(HotpathGuard *g, size_t n);
void hp_track_calloc(HotpathGuard *g, size_t n, size_t s);
void hp_track_realloc(HotpathGuard *g, void *p, size_t n);
void hp_track_free(HotpathGuard *g, void *p);
void hp_track_cuda_malloc(HotpathGuard *g, size_t n);
void hp_track_cuda_free(HotpathGuard *g);
void hp_track_page_fault(HotpathGuard *g);

#define HP_GUARD_MALLOC(g, n)     hp_track_malloc(g, n)
#define HP_GUARD_CALLOC(g, n, s)  hp_track_calloc(g, n, s)
#define HP_GUARD_REALLOC(g, p, n) hp_track_realloc(g, p, n)
#define HP_GUARD_FREE(g, p)       hp_track_free(g, p)
#define HP_GUARD_CUDA_MALLOC(g, n) hp_track_cuda_malloc(g, n)
#define HP_GUARD_CUDA_FREE(g)     hp_track_cuda_free(g)
#define HP_GUARD_PF(g)            hp_track_page_fault(g)
