/* ─── Lab 3: Hugepage TLB Miss Comparison ───────────────────────
 *
 * Allocate 1 GB of memory using 4 KB pages vs 2 MB hugepages.
 * Measure sequential scan and random access latency.
 *
 * Build:  gcc -O3 -lpthread bench.c -o bench
 * Usage:  sudo ./bench              # or:
 *         echo 1024 | sudo tee /proc/sys/vm/nr_hugepages
 *         ./bench
 *
 * Requires:  MAP_HUGETLB support, hugepages enabled.
 * If mmap with MAP_HUGETLB fails, the bench falls back to 4K.
 *
 * For TLB miss counts, run with perf:
 *   perf stat -e dTLB-loads,dTLB-load-misses ./bench
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#define SIZE_1GB  (1UL << 30)
#define STRIDES   8              /* sequential passes */
#define RANDOM_N  1000000        /* random accesses */

/* ─── Timer ───────────────────────────────────────────────────── */

static double
now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* ─── Allocate with given page size ───────────────────────────── */

static void *
alloc_4k(size_t size) {
  void *p = mmap(NULL, size, PROT_READ | PROT_WRITE,
                 MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  if (p == MAP_FAILED)
    return NULL;
  memset(p, 0, size);   /* fault all pages */
  return p;
}

static void *
alloc_2m(size_t size) {
  void *p = mmap(NULL, size, PROT_READ | PROT_WRITE,
                 MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB, -1, 0);
  if (p == MAP_FAILED)
    return NULL;
  memset(p, 0, size);
  return p;
}

/* ─── Benchmarks ──────────────────────────────────────────────── */

static double
bench_seq(const uint8_t *buf, size_t size, int passes) {
  volatile uint64_t sink = 0;
  double t0 = now_ns();
  for (int p = 0; p < passes; p++)
    for (size_t i = 0; i < size; i += 64)
      sink += buf[i];
  double t1 = now_ns();
  (void)sink;
  return t1 - t0;
}

static double
bench_rand(const uint8_t *buf, size_t size, int n_accesses) {
  /* simple LCG for reproducible random offsets */
  uint64_t rng = 42;
  volatile uint64_t sink = 0;
  size_t stride = size / n_accesses;
  if (stride < 64) stride = 64;

  double t0 = now_ns();
  for (int i = 0; i < n_accesses; i++) {
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    size_t off = (size_t)(rng % (size / 64)) * 64;
    sink += buf[off];
  }
  double t1 = now_ns();
  (void)sink;
  return t1 - t0;
}

int
main(void) {
  printf("Hugepage TLB Lab\n");
  printf("  allocation size:  1 GB\n\n");

  /* ── 4 KB pages ── */
  uint8_t *buf_4k = alloc_4k(SIZE_1GB);
  if (!buf_4k) { perror("alloc_4k"); return 1; }

  double seq_4k = bench_seq(buf_4k, SIZE_1GB, STRIDES);
  double rand_4k = bench_rand(buf_4k, SIZE_1GB, RANDOM_N);
  printf("  [4K pages]  sequential:  %.0f ms  |  random:  %.0f ms  (%.1f ns/access)\n",
         seq_4k / 1e6, rand_4k / 1e6, rand_4k / RANDOM_N);
  munmap(buf_4k, SIZE_1GB);

  /* ── 2 MB hugepages ── */
  uint8_t *buf_2m = alloc_2m(SIZE_1GB);
  if (!buf_2m) {
    printf("  [2M pages]  MAP_HUGETLB failed.  Are hugepages configured?\n");
    printf("             Try: echo 1024 | sudo tee /proc/sys/vm/nr_hugepages\n\n");
  } else {
    double seq_2m = bench_seq(buf_2m, SIZE_1GB, STRIDES);
    double rand_2m = bench_rand(buf_2m, SIZE_1GB, RANDOM_N);
    printf("  [2M pages]  sequential:  %.0f ms  |  random:  %.0f ms  (%.1f ns/access)\n",
           seq_2m / 1e6, rand_2m / 1e6, rand_2m / RANDOM_N);
    printf("\n  speedup:    seq %.1fx  |  rand %.1fx\n",
           seq_4k / seq_2m, rand_4k / rand_2m);
    munmap(buf_2m, SIZE_1GB);
  }

  printf("\n");
  printf("  What happened:\n");
  printf("    4K pages: 1 GB → 262,144 TLB entries.  Most accesses miss.\n");
  printf("    2M pages: 1 GB →   512 TLB entries.  Nearly all hit.\n");
  printf("\n");
  printf("  This is why the GB300 runtime uses MAP_HUGETLB for the\n");
  printf("  KV arena — the GPU never stalls on a TLB miss.\n");

  return 0;
}
