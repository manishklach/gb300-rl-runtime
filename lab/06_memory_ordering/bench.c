/* ─── Lab 6: Memory Ordering and Publication ────────────────────
 *
 * Demonstrates the descriptor publication rule used by the runtime:
 *
 *   producer: write payload -> release-store tail/flag
 *   consumer: acquire-load tail/flag -> read payload
 *
 * The BROKEN mode deliberately publishes the sequence number before the
 * payload. That creates a visible window where the consumer can observe
 * "new work" but still read stale descriptor contents.
 *
 * Build:  gcc -O3 -pthread bench.c -o bench
 * Usage:  ./bench [iterations]
 */

#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
static inline uint64_t rdclock(void) { return __rdtsc(); }
#else
static inline uint64_t rdclock(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
#endif

typedef struct {
  uint64_t seq;
  uint64_t payload;
} PayloadSlot;

typedef struct {
  PayloadSlot slot;
  volatile uint64_t published;
  volatile uint64_t consumed;
  volatile int stop;
  uint64_t stale_reads;
  uint64_t max_window_cycles;
  uint64_t iterations;
} SharedState;

typedef struct {
  SharedState *shared;
  int use_release_acquire;
} ThreadArg;

static void *
producer_thread(void *arg)
{
  ThreadArg *ta = (ThreadArg *)arg;
  SharedState *shared = ta->shared;

  for (uint64_t seq = 1; seq <= shared->iterations; seq++) {
    while (__atomic_load_n(&shared->consumed, __ATOMIC_ACQUIRE) != (seq - 1))
      __asm__ volatile("pause");

    if (ta->use_release_acquire) {
      shared->slot.seq = seq;
      shared->slot.payload = seq;
      __atomic_store_n(&shared->published, seq, __ATOMIC_RELEASE);
    } else {
      __atomic_store_n(&shared->published, seq, __ATOMIC_RELAXED);
      for (int i = 0; i < 64; i++)
        __asm__ volatile("" ::: "memory");
      shared->slot.seq = seq;
      shared->slot.payload = seq;
    }
  }

  __atomic_store_n(&shared->stop, 1, __ATOMIC_RELEASE);
  return NULL;
}

static void *
consumer_thread(void *arg)
{
  ThreadArg *ta = (ThreadArg *)arg;
  SharedState *shared = ta->shared;
  uint64_t last_seen = 0;

  for (;;) {
    const int done = __atomic_load_n(&shared->stop, __ATOMIC_ACQUIRE);
    const uint64_t published = ta->use_release_acquire
                                   ? __atomic_load_n(&shared->published, __ATOMIC_ACQUIRE)
                                   : __atomic_load_n(&shared->published, __ATOMIC_RELAXED);

    if (published != last_seen) {
      const uint64_t t0 = rdclock();
      const uint64_t payload_seq = shared->slot.seq;
      const uint64_t payload = shared->slot.payload;
      const uint64_t t1 = rdclock();

      if (payload_seq != published || payload != published) {
        shared->stale_reads++;
        const uint64_t window = t1 - t0;
        if (window > shared->max_window_cycles)
          shared->max_window_cycles = window;
      } else {
        last_seen = published;
        __atomic_store_n(&shared->consumed, published, __ATOMIC_RELEASE);
      }
    } else if (done) {
      break;
    } else {
      sched_yield();
    }
  }

  return NULL;
}

static void
run_mode(const char *label, int use_release_acquire, uint64_t iterations)
{
  SharedState shared;
  memset(&shared, 0, sizeof(shared));
  shared.iterations = iterations;

  pthread_t prod;
  pthread_t cons;
  ThreadArg arg = { .shared = &shared, .use_release_acquire = use_release_acquire };

  const uint64_t t0 = rdclock();
  pthread_create(&prod, NULL, producer_thread, &arg);
  pthread_create(&cons, NULL, consumer_thread, &arg);
  pthread_join(prod, NULL);
  pthread_join(cons, NULL);
  const uint64_t t1 = rdclock();

  printf("  [%s]\n", label);
  printf("    stale observations:  %llu\n", (unsigned long long)shared.stale_reads);
  printf("    max visibility gap:  %llu ticks\n",
         (unsigned long long)shared.max_window_cycles);
  printf("    wall clock:          %llu ticks\n",
         (unsigned long long)(t1 - t0));
}

int
main(int argc, char **argv)
{
  const uint64_t iterations = (argc > 1) ? strtoull(argv[1], NULL, 10) : 1000000ULL;

  printf("Memory Ordering Lab\n");
  printf("  iterations: %llu\n\n", (unsigned long long)iterations);

  run_mode("BROKEN relaxed publish", 0, iterations);
  run_mode("CORRECT release/acquire publish", 1, iterations);

  printf("\nWhat this shows:\n");
  printf("  The broken mode publishes the sequence before the payload, so\n");
  printf("  the consumer can observe a new flag and stale data together.\n");
  printf("  The correct mode follows the ring invariant used in the runtime:\n");
  printf("  payload first, then a release-store; consumer does an acquire-load\n");
  printf("  before dereferencing the payload.\n");
  printf("\n");
  printf("  On x86, acquire/release is often cheap because TSO already helps.\n");
  printf("  On weaker machines and GPUs, the ordering contract is what keeps\n");
  printf("  descriptors from being observed half-published.\n");
  return 0;
}
