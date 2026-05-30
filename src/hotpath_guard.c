#include "hotpath_guard.h"
#include <string.h>

void
hp_guard_init(HotpathGuard *g)
{
    memset(g, 0, sizeof(*g));
}

void
hp_guard_activate(HotpathGuard *g)
{
    g->active = 1;
}

void
hp_guard_deactivate(HotpathGuard *g)
{
    g->active = 0;
}

static void
hp_violation(const char *what)
{
    fprintf(stderr, "HOT PATH VIOLATION: %s in hot path\n", what);
}

void
hp_track_malloc(HotpathGuard *g, size_t n)
{
    g->malloc_count++;
    if (g->active) hp_violation("malloc");
}

void
hp_track_calloc(HotpathGuard *g, size_t n, size_t s)
{
    (void)n; (void)s;
    g->calloc_count++;
    if (g->active) hp_violation("calloc");
}

void
hp_track_realloc(HotpathGuard *g, void *p, size_t n)
{
    (void)p; (void)n;
    g->realloc_count++;
    if (g->active) hp_violation("realloc");
}

void
hp_track_free(HotpathGuard *g, void *p)
{
    (void)p;
    g->free_count++;
    if (g->active) hp_violation("free");
}

void
hp_track_cuda_malloc(HotpathGuard *g, size_t n)
{
    (void)n;
    g->cuda_malloc_count++;
    if (g->active) hp_violation("cudaMalloc");
}

void
hp_track_cuda_free(HotpathGuard *g)
{
    g->cuda_free_count++;
    if (g->active) hp_violation("cudaFree");
}

void
hp_track_page_fault(HotpathGuard *g)
{
    g->page_fault_count++;
    if (g->active) hp_violation("page fault");
}
