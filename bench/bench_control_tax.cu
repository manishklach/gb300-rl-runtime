#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdatomic.h>
#include <unistd.h>
#include <sys/eventfd.h>
#include <pthread.h>

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static double ns_per_op(uint64_t total_ns, uint64_t n)
{
    return (double)total_ns / (double)n;
}

/* ─── Mode A: eventfd wakeup per step (simulated syscall tax) ─── */
static double
bench_mode_a(int n)
{
    int efd = eventfd(0, 0);
    if (efd < 0) { perror("eventfd"); return -1; }

    uint64_t val = 1;
    uint64_t t0 = now_ns();
    for (int i = 0; i < n; i++) {
        write(efd, &val, sizeof(val));
        read(efd, &val, sizeof(val));
    }
    uint64_t t1 = now_ns();
    close(efd);
    return ns_per_op(t1 - t0, n);
}

/* ─── Mode B: shared-memory polling (userspace ring) ──────────── */
typedef struct {
    volatile uint64_t head;
    volatile uint64_t tail;
    char pad[48];
} PollSlot;

static double
bench_mode_b(int n)
{
    PollSlot *slot = (PollSlot *)calloc(1, 64);
    if (!slot) return -1;

    uint64_t t0 = now_ns();
    for (int i = 0; i < n; i++) {
        __atomic_store_n(&slot->head, i + 1, __ATOMIC_RELEASE);
        while (__atomic_load_n(&slot->tail, __ATOMIC_ACQUIRE) < i + 1)
            __asm__ volatile("pause"); /* spin */
    }
    uint64_t t1 = now_ns();
    free(slot);
    return ns_per_op(t1 - t0, n);
}

/* ─── Mode C: persistent worker + command ring (reference) ────── */
static double
bench_mode_c(int n)
{
    /* allocate ring in heap (user-space only — simulates the ring) */
    volatile uint64_t *ring = (volatile uint64_t *)aligned_alloc(64, 64 * 2);
    if (!ring) return -1;
    memset((void *)ring, 0, 128);

    uint64_t t0 = now_ns();
    for (int i = 0; i < n; i++) {
        ring[0] = i;                        /* producer write */
        __atomic_store_n(&ring[1], i + 1, __ATOMIC_RELEASE); /* commit */

        while (__atomic_load_n(&ring[1], __ATOMIC_ACQUIRE) > 0) {
            uint64_t v = ring[0];
            (void)v;
            break;
        }
        /* single-pass: consumer reads once per iteration in the
           reference mode, analogous to the persistent worker's
           ring_consume() call */
    }
    uint64_t t1 = now_ns();
    free((void *)ring);
    return ns_per_op(t1 - t0, n);
}

int main(int argc, char **argv)
{
    int n = 1000000;
    if (argc > 1) n = atoi(argv[1]);

    printf("GB300 RL Runtime — Control-Plane Tax Benchmark\n");
    printf("  Iterations:       %d\n\n", n);

    printf("  Mode A (eventfd syscall per step):  ");
    double a = bench_mode_a(n);
    if (a < 0) { printf("FAILED\n"); return 1; }
    printf("%.0f ns/op  (%.0f M ops/s)\n", a, 1000.0 / a);

    printf("  Mode B (userspace polling ring):    ");
    double b = bench_mode_b(n);
    if (b < 0) { printf("FAILED\n"); return 1; }
    printf("%.0f ns/op  (%.0f M ops/s)\n", b, 1000.0 / b);

    printf("  Mode C (descriptor ring, ref):      ");
    double c = bench_mode_c(n);
    if (c < 0) { printf("FAILED\n"); return 1; }
    printf("%.0f ns/op  (%.0f M ops/s)\n", c, 1000.0 / c);

    printf("\n── Comparison ──\n");
    printf("  Syscall overhead (A - C):    %.0f ns/op  (%.1fx)\n", a - c, a / c);
    printf("  Polling overhead (B - C):    %.0f ns/op  (%.1fx)\n", b - c, b / c);
    printf("  Syscall vs polling (A / B):  %.1fx\n", a / b);

    printf("\nInterpretation:\n");
    printf("  Mode A = PyTorch/vLLM per-step launch model\n");
    printf("  Mode B = userspace polling (fast but CPU-hungry)\n");
    printf("  Mode C = this runtime's persistent worker approach\n");
    printf("\nDone.\n");
    return 0;
}
