/* ─── Lab 4: Syscall Wakeup vs Shared-Memory Polling ────────────
 *
 * Compare the cost of waking up a consumer thread via:
 *
 *   A) eventfd (syscall)    — write() to an eventfd wakes the reader
 *   B) shared-memory poll   — consumer busy-waits on a volatile variable
 *
 * This directly measures why the GB300 runtime avoids kernel
 * intervention in the hot path: each syscall costs ~1-5 µs.
 *
 * Build:  gcc -O3 -lpthread bench.c -o bench
 * Usage:  ./bench [iterations]
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>
#include <sys/eventfd.h>
#include <unistd.h>
#include <stdatomic.h>

/* ─── Shared state for poll-based wakeup ──────────────────────── */

typedef struct {
  volatile uint64_t  flag      _Alignas(64);
  char               _pad[56];
} poll_state;

/* ─── Timer ───────────────────────────────────────────────────── */

static double
now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* ═══════════════════════════════════════════════════════════════
 *  Version A: eventfd wakeup
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
  int       efd;
  uint64_t  count;
} efd_arg;

static void *
efd_consumer(void *arg) {
  efd_arg *ea = (efd_arg *)arg;
  uint64_t val;
  for (uint64_t i = 0; i < ea->count; i++)
    read(ea->efd, &val, sizeof(val));       /* blocking syscall */
  return NULL;
}

static double
bench_eventfd(uint64_t n) {
  int efd = eventfd(0, 0);
  if (efd < 0) { perror("eventfd"); return -1; }

  efd_arg ea = { efd, n };
  pthread_t t;
  pthread_create(&t, NULL, efd_consumer, &ea);

  double t0 = now_ns();
  for (uint64_t i = 0; i < n; i++) {
    uint64_t one = 1;
    write(efd, &one, sizeof(one));          /* waking syscall */
  }
  pthread_join(t, NULL);
  double t1 = now_ns();

  close(efd);
  return t1 - t0;
}

/* ═══════════════════════════════════════════════════════════════
 *  Version B: shared-memory polling (busy-wait)
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
  poll_state *ps;
  uint64_t    count;
} poll_arg;

static void *
poll_consumer(void *arg) {
  poll_arg *pa = (poll_arg *)arg;
  for (uint64_t i = 0; i < pa->count; ) {
    if (pa->ps->flag) {
      pa->ps->flag = 0;
      i++;
    }
    /* no yield — tight poll, measuring worst-case CPU cost */
  }
  return NULL;
}

static double
bench_poll(uint64_t n) {
  poll_state ps = {0};
  poll_arg pa = { &ps, n };
  pthread_t t;
  pthread_create(&t, NULL, poll_consumer, &pa);

  double t0 = now_ns();
  for (uint64_t i = 0; i < n; i++) {
    ps.flag = 1;                            /* no syscall */
    /* producer doesn't wait — consumer will see it */
  }
  pthread_join(t, NULL);
  double t1 = now_ns();

  return t1 - t0;
}

/* ═══════════════════════════════════════════════════════════════
 *  Version C: polling with sched_yield (hybrid)
 * ═══════════════════════════════════════════════════════════════ */

static void *
yield_consumer(void *arg) {
  poll_arg *pa = (poll_arg *)arg;
  for (uint64_t i = 0; i < pa->count; ) {
    if (pa->ps->flag) {
      pa->ps->flag = 0;
      i++;
    } else {
      sched_yield();                        /* light-weight syscall */
    }
  }
  return NULL;
}

static double
bench_yield(uint64_t n) {
  poll_state ps = {0};
  poll_arg pa = { &ps, n };
  pthread_t t;
  pthread_create(&t, NULL, yield_consumer, &pa);

  double t0 = now_ns();
  for (uint64_t i = 0; i < n; i++) {
    ps.flag = 1;
  }
  pthread_join(t, NULL);
  double t1 = now_ns();

  return t1 - t0;
}

/* ═══════════════════════════════════════════════════════════════ */

int
main(int argc, char **argv) {
  uint64_t n = (argc > 1) ? (uint64_t)atol(argv[1]) : 100000ULL;

  printf("Syscall vs Polling Lab\n");
  printf("  iterations:  %lu\n\n", n);

  double ns_efd   = bench_eventfd(n);
  printf("  [eventfd]    syscall wakeup:     %.0f ms  (%.1f ns/op)\n",
         ns_efd / 1e6, ns_efd / n);

  double ns_poll  = bench_poll(n);
  printf("  [busy-poll]  shared-memory poll: %.0f ms  (%.1f ns/op)\n",
         ns_poll / 1e6, ns_poll / n);

  double ns_yield = bench_yield(n);
  printf("  [yield]      poll + sched_yield: %.0f ms  (%.1f ns/op)\n",
         ns_yield / 1e6, ns_yield / n);

  printf("\n");
  printf("  speedup (poll vs eventfd):  %.1fx\n", ns_efd / ns_poll);
  printf("  speedup (poll vs yield):    %.1fx\n", ns_yield / ns_poll);
  printf("\n");
  printf("  What happened:\n");
  printf("    eventfd:  1 write() + 1 read() syscall per wakeup.\n");
  printf("              Each syscall is ~1-5 µs (mode switch + scheduler).\n");
  printf("    busy-poll: no kernel entry.  Consumer spins forever.\n");
  printf("               Fastest, but burns CPU when idle.\n");
  printf("    yield:     trade-off: spin a bit, then yield to scheduler.\n");
  printf("\n");
  printf("  The GB300 runtime uses polling in the persistent GPU\n");
  printf("  worker (no host wakeup per token) and the CPU polls\n");
  printf("  completions at ~100 Hz (not per-token).\n");

  return 0;
}
