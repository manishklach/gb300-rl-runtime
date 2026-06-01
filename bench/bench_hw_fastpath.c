#include "hw_desc.h"
#include "hw_ring.h"
#include "hw_worker_sim.h"
#include "infer_submit.h"
#include "mmio.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    hw_worker_sim_t *worker;
} hw_bench_thread_arg_t;

static void *
hw_bench_worker_main(void *arg)
{
    hw_bench_thread_arg_t *thread_arg = (hw_bench_thread_arg_t *)arg;
    hw_worker_sim_run(thread_arg->worker);
    return NULL;
}

static uint64_t
now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static int
cmp_u64(const void *a, const void *b)
{
    const uint64_t va = *(const uint64_t *)a;
    const uint64_t vb = *(const uint64_t *)b;
    return (va > vb) - (va < vb);
}

static hw_desc_t
make_decode_desc(uint32_t rollout_id)
{
    hw_desc_t desc;
    memset(&desc, 0, sizeof(desc));
    desc.opcode = DESC_OP_DECODE;
    desc.rollout_id = rollout_id;
    desc.kv_arena_id = 1u;
    desc.kv_offset = (uint64_t)rollout_id << 12;
    desc.delta_offset = (uint64_t)rollout_id << 8;
    desc.seq_len = rollout_id & 7u;
    desc.max_tokens = 128u;
    desc.user_data = rollout_id;
    desc.checksum_or_cookie = desc.kv_offset ^ desc.delta_offset;
    return desc;
}

static void
run_batch_benchmark(uint32_t batch_size, uint32_t iterations)
{
    hw_ring_t *cmdq = hw_ring_create();
    hw_ring_t *doneq = hw_ring_create();
    volatile uint32_t stop = 0;
    volatile uint32_t doorbell = 0;
    hw_worker_sim_t worker;
    hw_bench_thread_arg_t thread_arg;
    pthread_t worker_thread;
    uint64_t *latencies = NULL;
    uint64_t t_submit0;
    uint64_t t_submit1;
    uint64_t t_drain1;
    uint32_t submitted = 0;
    uint32_t completed = 0;
    uint32_t i;

    if (!cmdq || !doneq)
        goto cleanup;

    latencies = (uint64_t *)malloc((size_t)iterations * sizeof(*latencies));
    if (!latencies)
        goto cleanup;

    memset(&worker, 0, sizeof(worker));
    worker.cmdq = cmdq;
    worker.doneq = doneq;
    worker.stop = &stop;
    thread_arg.worker = &worker;
    pthread_create(&worker_thread, NULL, hw_bench_worker_main, &thread_arg);

    t_submit0 = now_ns();
    for (i = 0; i < iterations; i++) {
        const uint64_t start = now_ns();
        hw_desc_t desc = make_decode_desc(i + 1u);
        while (hw_ring_push(cmdq, &desc) != 0) {
            hw_desc_t out;
            while (hw_ring_pop(doneq, &out) == 0)
                completed++;
#if defined(__x86_64__)
            __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__)
            __asm__ volatile("yield" ::: "memory");
#endif
        }
        submitted++;

        if (((i + 1u) % batch_size) == 0u || (i + 1u) == iterations)
            mmio_write32(&doorbell, hw_ring_load_acquire(&cmdq->tail));

        if (((i + 1u) & 255u) == 0u) {
            hw_desc_t out;
            while (hw_ring_pop(doneq, &out) == 0)
                completed++;
        }

        latencies[i] = now_ns() - start;
    }
    t_submit1 = now_ns();

    while (completed < iterations) {
        hw_desc_t out;
        if (hw_ring_pop(doneq, &out) == 0) {
            completed++;
            continue;
        }
#if defined(__x86_64__)
        __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__)
        __asm__ volatile("yield" ::: "memory");
#endif
    }
    t_drain1 = now_ns();

    qsort(latencies, iterations, sizeof(*latencies), cmp_u64);
    printf("  batch=%u\n", batch_size);
    printf("    doorbell writes:     %u\n", (iterations + batch_size - 1u) / batch_size);
    printf("    submit throughput:   %.0f desc/s\n",
           (double)submitted * 1.0e9 / (double)(t_submit1 - t_submit0));
    printf("    completion throughput: %.0f desc/s\n",
           (double)completed * 1.0e9 / (double)(t_drain1 - t_submit0));
    printf("    submit latency p50:  %lu ns\n",
           (unsigned long)latencies[iterations / 2u]);
    printf("    submit latency p99:  %lu ns\n",
           (unsigned long)latencies[(iterations * 99u) / 100u]);
    printf("    decoded tokens:      %lu\n",
           (unsigned long)worker.decoded_tokens);
    printf("    final doorbell:      %u\n", mmio_read32(&doorbell));

    {
        hw_desc_t stop_desc;
        memset(&stop_desc, 0, sizeof(stop_desc));
        stop_desc.opcode = DESC_OP_STOP;
        while (hw_ring_push(cmdq, &stop_desc) != 0) {
#if defined(__x86_64__)
            __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__)
            __asm__ volatile("yield" ::: "memory");
#endif
        }
        mmio_write32(&doorbell, hw_ring_load_acquire(&cmdq->tail));
    }

    pthread_join(worker_thread, NULL);

cleanup:
    free(latencies);
    hw_ring_destroy(cmdq);
    hw_ring_destroy(doneq);
}

static void
run_submit_api_smoke(void)
{
    hw_ring_t *cmdq = hw_ring_create();
    hw_ring_t *doneq = hw_ring_create();
    volatile uint32_t doorbell = 0;
    infer_hw_ctx_t ctx;

    if (!cmdq || !doneq)
        goto cleanup;

    memset(&ctx, 0, sizeof(ctx));
    ctx.cmdq = cmdq;
    ctx.doneq = doneq;
    ctx.doorbell = &doorbell;
    ctx.kv_arena_id = 11u;
    ctx.gpu_group_id = 3u;

    printf("  infer_submit_decode smoke:\n");
    printf("    status: %d\n",
           infer_submit_decode(&ctx, 1u, 0x1000u, 0x2000u, 7u, 16u, 64u));
    printf("    doorbell after submit: %u\n", mmio_read32(&doorbell));

cleanup:
    hw_ring_destroy(cmdq);
    hw_ring_destroy(doneq);
}

int
main(int argc, char **argv)
{
    uint32_t iterations = 200000u;
    if (argc > 1)
        iterations = (uint32_t)strtoul(argv[1], NULL, 10);

    printf("GB300 RL Runtime - Hardware Fastpath Benchmark\n");
    printf("  iterations: %u\n\n", iterations);

    run_submit_api_smoke();
    printf("\n  Doorbell batching sweep:\n");
    run_batch_benchmark(1u, iterations);
    run_batch_benchmark(8u, iterations);
    run_batch_benchmark(32u, iterations);
    run_batch_benchmark(64u, iterations);
    return 0;
}
