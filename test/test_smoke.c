#include "ring.h"
#include "completion.h"
#include "request.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static void test_ring_empty(void)
{
    printf("  test_ring_empty ... ");
    CommandRing *ring = ring_create();
    assert(ring);

    Descriptor d;
    assert(ring_consume(ring, &d) == 0);

    ring_destroy(ring);
    printf("OK\n");
}

static void test_ring_full(void)
{
    printf("  test_ring_full ... ");
    CommandRing *ring = ring_create();
    assert(ring);

    for (uint32_t i = 0; i < RING_SIZE; i++) {
        uint32_t pos = ring_acquire(ring, 1);
        assert(pos != UINT32_MAX);
        ring->slots[pos].seq_id = i;
        ring_commit(ring, 1);
    }
    assert(ring_acquire(ring, 1) == UINT32_MAX);

    for (uint32_t i = 0; i < RING_SIZE; i++) {
        Descriptor d;
        assert(ring_consume(ring, &d) == 1);
        assert(d.seq_id == i);
    }
    assert(ring_acquire(ring, 1) != UINT32_MAX);

    ring_destroy(ring);
    printf("OK\n");
}

static void test_ring_wraparound(void)
{
    printf("  test_ring_wraparound ... ");
    CommandRing *ring = ring_create();
    assert(ring);

    for (uint32_t i = 0; i < RING_SIZE / 2; i++) {
        uint32_t pos = ring_acquire(ring, 1);
        assert(pos != UINT32_MAX);
        ring->slots[pos].seq_id = i;
        ring_commit(ring, 1);
    }

    for (uint32_t i = 0; i < RING_SIZE / 2; i++) {
        Descriptor d;
        assert(ring_consume(ring, &d) == 1);
        assert(d.seq_id == i);
    }

    for (uint32_t i = 0; i < RING_SIZE; i++) {
        uint32_t pos = ring_acquire(ring, 1);
        assert(pos != UINT32_MAX);
        ring->slots[pos].seq_id = 10000 + i;
        ring_commit(ring, 1);
    }

    for (uint32_t i = 0; i < RING_SIZE; i++) {
        Descriptor d;
        assert(ring_consume(ring, &d) == 1);
        assert(d.seq_id == 10000 + i);
    }

    ring_destroy(ring);
    printf("OK\n");
}

static void test_producer_observes_consumer_progress(void)
{
    printf("  test_producer_observes_consumer_progress ... ");
    CommandRing *ring = ring_create();
    assert(ring);

    for (uint32_t i = 0; i < RING_SIZE; i++) {
        uint32_t pos = ring_acquire(ring, 1);
        assert(pos != UINT32_MAX);
        ring->slots[pos].seq_id = i;
        ring_commit(ring, 1);
    }

    Descriptor d;
    assert(ring_consume(ring, &d) == 1);
    assert(d.seq_id == 0);

    uint32_t pos = ring_acquire(ring, 1);
    assert(pos == 0);
    ring->slots[pos].seq_id = 99999;
    ring_commit(ring, 1);

    for (uint32_t i = 1; i < RING_SIZE; i++) {
        assert(ring_consume(ring, &d) == 1);
        assert(d.seq_id == i);
    }
    assert(ring_consume(ring, &d) == 1);
    assert(d.seq_id == 99999);

    ring_destroy(ring);
    printf("OK\n");
}

static void test_ring_no_permanent_full(void)
{
    printf("  test_ring_no_permanent_full ... ");
    CommandRing *ring = ring_create();
    assert(ring);

    for (uint32_t round = 0; round < 3; round++) {
        Descriptor d;
        for (uint32_t i = 0; i < RING_SIZE; i++) {
            uint32_t pos = ring_acquire(ring, 1);
            assert(pos != UINT32_MAX);
            ring->slots[pos].seq_id = round * RING_SIZE + i;
            ring_commit(ring, 1);
        }
        assert(ring_acquire(ring, 1) == UINT32_MAX);
        for (uint32_t i = 0; i < RING_SIZE; i++) {
            assert(ring_consume(ring, &d) == 1);
        }
        assert(ring_consume(ring, &d) == 0);
    }

    ring_destroy(ring);
    printf("OK\n");
}

static void test_completion_ring_overflow(void)
{
    printf("  test_completion_ring_overflow ... ");
    CompletionRing ring;
    memset(&ring, 0, sizeof(ring));

    Completion c = { .seq_id = 1, .token_id = 7, .kv_block_offset = 0, .reward_cookie = 9, .cycles_taken = 11 };
    for (uint32_t i = 0; i < COMP_RING_SIZE; i++)
        assert(comp_ring_push(&ring, &c) == 0);

    assert(comp_ring_push(&ring, &c) == -1);
    assert(ring.overflow.value == 1);

    assert(comp_ring_poll(&ring, &c) == 1);
    assert(comp_ring_push(&ring, &c) == 0);
    printf("OK\n");
}

static void test_done_ring_overflow(void)
{
    printf("  test_done_ring_overflow ... ");
    DoneRing ring;
    memset(&ring, 0, sizeof(ring));

    RolloutDone done = { .request_id = 1, .rollout_id = 2, .tokens_generated = 3, .reward = 0.5f, .status = 0 };
    for (uint32_t i = 0; i < REQUEST_RING_SIZE; i++)
        assert(done_ring_push(&ring, &done) == 0);

    assert(done_ring_push(&ring, &done) == -1);
    assert(ring.overflow.value == 1);

    assert(done_ring_pop(&ring, &done) == 1);
    assert(done_ring_push(&ring, &done) == 0);
    printf("OK\n");
}

int main(void)
{
    printf("GB300 RL Runtime — CPU Smoke Tests\n\n");
    test_ring_empty();
    test_ring_full();
    test_ring_wraparound();
    test_producer_observes_consumer_progress();
    test_ring_no_permanent_full();
    test_completion_ring_overflow();
    test_done_ring_overflow();
    printf("\nAll smoke tests passed.\n");
    return 0;
}
