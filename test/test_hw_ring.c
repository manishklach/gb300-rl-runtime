#include "hw_desc.h"
#include "hw_ring.h"
#include "hw_worker_sim.h"
#include "infer_submit.h"

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static void
test_desc_size(void)
{
    printf("  test_desc_size ... ");
    assert(sizeof(hw_desc_t) == 64u);
    printf("OK\n");
}

static void
test_hw_ring_empty_full_wraparound(void)
{
    printf("  test_hw_ring_empty_full_wraparound ... ");
    hw_ring_t *ring = hw_ring_create();
    assert(ring);
    assert(hw_ring_empty(ring));

    for (uint32_t i = 0; i < HW_RING_SIZE; i++) {
        hw_desc_t desc;
        memset(&desc, 0, sizeof(desc));
        desc.rollout_id = i;
        assert(hw_ring_push(ring, &desc) == 0);
    }
    assert(hw_ring_full(ring));

    for (uint32_t i = 0; i < HW_RING_SIZE / 2u; i++) {
        hw_desc_t out;
        assert(hw_ring_pop(ring, &out) == 0);
        assert(out.rollout_id == i);
    }

    for (uint32_t i = 0; i < HW_RING_SIZE / 2u; i++) {
        hw_desc_t desc;
        memset(&desc, 0, sizeof(desc));
        desc.rollout_id = 100000u + i;
        assert(hw_ring_push(ring, &desc) == 0);
    }

    for (uint32_t i = HW_RING_SIZE / 2u; i < HW_RING_SIZE; i++) {
        hw_desc_t out;
        assert(hw_ring_pop(ring, &out) == 0);
        assert(out.rollout_id == i);
    }
    for (uint32_t i = 0; i < HW_RING_SIZE / 2u; i++) {
        hw_desc_t out;
        assert(hw_ring_pop(ring, &out) == 0);
        assert(out.rollout_id == 100000u + i);
    }

    assert(hw_ring_empty(ring));
    hw_ring_destroy(ring);
    printf("OK\n");
}

static void
test_producer_observes_consumer_progress(void)
{
    printf("  test_producer_observes_consumer_progress ... ");
    hw_ring_t *ring = hw_ring_create();
    assert(ring);

    for (uint32_t i = 0; i < HW_RING_SIZE; i++) {
        hw_desc_t desc;
        memset(&desc, 0, sizeof(desc));
        desc.rollout_id = i;
        assert(hw_ring_push(ring, &desc) == 0);
    }

    hw_desc_t out;
    assert(hw_ring_pop(ring, &out) == 0);
    assert(out.rollout_id == 0u);

    memset(&out, 0, sizeof(out));
    out.rollout_id = 999999u;
    assert(hw_ring_push(ring, &out) == 0);

    for (uint32_t i = 1; i < HW_RING_SIZE; i++) {
        assert(hw_ring_pop(ring, &out) == 0);
        assert(out.rollout_id == i);
    }
    assert(hw_ring_pop(ring, &out) == 0);
    assert(out.rollout_id == 999999u);

    hw_ring_destroy(ring);
    printf("OK\n");
}

static void
test_infer_submit_decode_updates_doorbell(void)
{
    printf("  test_infer_submit_decode_updates_doorbell ... ");
    hw_ring_t *cmdq = hw_ring_create();
    hw_ring_t *doneq = hw_ring_create();
    volatile uint32_t doorbell = 0;
    infer_hw_ctx_t ctx;
    hw_desc_t out;

    assert(cmdq && doneq);
    memset(&ctx, 0, sizeof(ctx));
    ctx.cmdq = cmdq;
    ctx.doneq = doneq;
    ctx.doorbell = &doorbell;
    ctx.gpu_group_id = 7u;
    ctx.kv_arena_id = 42u;

    assert(infer_submit_decode(&ctx, 99u, 0x1000u, 0x2000u, 5u, 17u, 64u) == 0);
    assert(doorbell == 1u);
    assert(hw_ring_pop(cmdq, &out) == 0);
    assert(out.opcode == DESC_OP_DECODE);
    assert(out.rollout_id == 99u);
    assert(out.kv_arena_id == 42u);
    assert(out.prefix_id == 5u);
    assert((out.flags & DESC_FLAG_COW_PREFIX) != 0u);
    assert(out.seq_len == 17u);
    assert(out.max_tokens == 64u);

    hw_ring_destroy(cmdq);
    hw_ring_destroy(doneq);
    printf("OK\n");
}

static void
test_doneq_overflow_detected(void)
{
    printf("  test_doneq_overflow_detected ... ");
    hw_ring_t *doneq = hw_ring_create();
    assert(doneq);

    for (uint32_t i = 0; i < HW_RING_SIZE; i++) {
        hw_desc_t desc;
        memset(&desc, 0, sizeof(desc));
        desc.rollout_id = i;
        assert(hw_ring_push(doneq, &desc) == 0);
    }

    {
        hw_desc_t desc;
        memset(&desc, 0, sizeof(desc));
        assert(hw_ring_push(doneq, &desc) == -1);
    }

    hw_ring_destroy(doneq);
    printf("OK\n");
}

typedef struct {
    hw_worker_sim_t *worker;
} worker_thread_arg_t;

static void *
worker_thread_main(void *arg)
{
    worker_thread_arg_t *thread_arg = (worker_thread_arg_t *)arg;
    hw_worker_sim_run(thread_arg->worker);
    return NULL;
}

static void
test_hw_worker_sim_completion(void)
{
    printf("  test_hw_worker_sim_completion ... ");
    hw_ring_t *cmdq = hw_ring_create();
    hw_ring_t *doneq = hw_ring_create();
    volatile uint32_t stop = 0;
    hw_worker_sim_t worker;
    worker_thread_arg_t thread_arg;
    pthread_t thread;
    hw_desc_t desc;
    hw_desc_t out;

    assert(cmdq && doneq);
    memset(&worker, 0, sizeof(worker));
    worker.cmdq = cmdq;
    worker.doneq = doneq;
    worker.stop = &stop;
    thread_arg.worker = &worker;
    assert(pthread_create(&thread, NULL, worker_thread_main, &thread_arg) == 0);

    memset(&desc, 0, sizeof(desc));
    desc.opcode = DESC_OP_DECODE;
    desc.rollout_id = 77u;
    desc.seq_len = 31u;
    desc.max_tokens = 96u;
    assert(hw_ring_push(cmdq, &desc) == 0);

    while (hw_ring_pop(doneq, &out) != 0) {
#if defined(__x86_64__)
        __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__)
        __asm__ volatile("yield" ::: "memory");
#endif
    }

    assert(out.rollout_id == 77u);
    assert((out.flags & (DESC_FLAG_NEEDS_REWARD | DESC_FLAG_DONE)) != 0u);
    assert(worker.decoded_tokens > 0u);
    assert(worker.completions == 1u);

    memset(&desc, 0, sizeof(desc));
    desc.opcode = DESC_OP_STOP;
    assert(hw_ring_push(cmdq, &desc) == 0);
    pthread_join(thread, NULL);

    hw_ring_destroy(cmdq);
    hw_ring_destroy(doneq);
    printf("OK\n");
}

int
main(void)
{
    printf("GB300 RL Runtime - Hardware Fastpath Tests\n\n");
    test_desc_size();
    test_hw_ring_empty_full_wraparound();
    test_producer_observes_consumer_progress();
    test_infer_submit_decode_updates_doorbell();
    test_doneq_overflow_detected();
    test_hw_worker_sim_completion();
    printf("\nAll hardware fastpath tests passed.\n");
    return 0;
}
