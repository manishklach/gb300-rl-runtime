#include "pipeline.h"
#include "metrics.h"
#include "hotpath_guard.h"
#include "reward.h"
#include "kv_prefix.h"
#include "ring.h"
#include "arena.h"
#include "completion.h"
#include "attention_decode.h"
#include "model_state.h"
#include "query_producer.h"
#include "sample.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

__global__ void decode_worker(CommandRing*, KVArena*, CompletionRing*,
                              SampleState*, const __half*, float*, uint32_t,
                              uint64_t*);

typedef struct {
    RolloutPipeline  pipeline;
    RuntimeMetrics   metrics;
    HotpathGuard     hp_guard;
    KVPrefixTable    kv_prefix_table;
    CommandRing     *cmd_ring;
    CompletionRing  *comp_ring;
    KVArena          kv_arena;
    RewardRing       reward_ring;
    uint64_t        *d_step_count;
    SampleState     *d_sample_st;
    float           *d_hidden_buf;
    __half          *d_query_proj;
    __half          *d_query_buf;
    float           *d_output_buf;
    uint32_t         io_slots;
    int              num_sms;
    int              dev_id;
} BenchContext;

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static int bench_init(BenchContext *ctx, int dev_id, int n_rollouts)
{
    (void)n_rollouts;
    memset(ctx, 0, sizeof(*ctx));
    ctx->dev_id = dev_id;
    cudaSetDevice(dev_id);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev_id);
    ctx->num_sms = prop.multiProcessorCount;

    pipeline_init(&ctx->pipeline);
    metrics_init(&ctx->metrics);
    hp_guard_init(&ctx->hp_guard);
    kv_prefix_table_init(&ctx->kv_prefix_table);

    ctx->cmd_ring  = ring_create();
    ctx->comp_ring = (CompletionRing *)ring_create();
    if (!ctx->cmd_ring || !ctx->comp_ring) {
        fprintf(stderr, "failed to create rings\n");
        return -1;
    }

    arena_init(&ctx->kv_arena, 256UL << 20, 16384);
    reward_ring_init(&ctx->reward_ring);

    cudaMalloc(&ctx->d_step_count, sizeof(uint64_t));
    cudaMemset(ctx->d_step_count, 0, sizeof(uint64_t));
    ctx->io_slots = RING_SIZE;
    if (model_state_init(&ctx->d_hidden_buf, ctx->io_slots) != 0) {
        fprintf(stderr, "failed to initialize model state\n");
        return -1;
    }
    if (query_producer_init(&ctx->d_query_proj) != 0) {
        fprintf(stderr, "failed to initialize query producer\n");
        return -1;
    }
    cudaMalloc(&ctx->d_query_buf, ctx->io_slots * DECODE_FIXED_HEAD_DIM * sizeof(__half));
    cudaMalloc(&ctx->d_output_buf, ctx->io_slots * DECODE_FIXED_HEAD_DIM * sizeof(float));
    cudaMemset(ctx->d_output_buf, 0, ctx->io_slots * DECODE_FIXED_HEAD_DIM * sizeof(float));

    cudaMalloc(&ctx->d_sample_st, sizeof(SampleState));
    SampleState h_st;
    memset(&h_st, 0, sizeof(h_st));
    h_st.rng_state[0] = 42;
    h_st.rng_state[1] = 42 ^ 0x9e3779b97f4a7c15ULL;
    h_st.rng_state[2] = (42 << 17) ^ 0x3c6ef372fe94f82aULL;
    h_st.rng_state[3] = ~42ULL;
    h_st.temperature   = 1.0f;
    h_st.top_k         = 50;
    h_st.top_p         = 0.9f;
    h_st.vocab_size    = MAX_VOCAB_SIZE;
    cudaMemcpy(ctx->d_sample_st, &h_st, sizeof(SampleState), cudaMemcpyHostToDevice);

    int smem_size = 3 * 16384;
    decode_worker<<<ctx->num_sms, 32, smem_size>>>(
        ctx->cmd_ring, &ctx->kv_arena, ctx->comp_ring,
        ctx->d_sample_st, ctx->d_query_buf, ctx->d_output_buf, ctx->io_slots,
        ctx->d_step_count);

    return 0;
}

static void bench_shutdown(BenchContext *ctx)
{
    Descriptor sentinel;
    memset(&sentinel, 0, sizeof(sentinel));
    sentinel.seq_id = UINT64_MAX;
    int sent = 0;
    while (sent < ctx->num_sms) {
        uint32_t pos = ring_acquire(ctx->cmd_ring, 1);
        if (pos == UINT32_MAX)
            continue;
        ctx->cmd_ring->slots[pos] = sentinel;
        ring_commit(ctx->cmd_ring, 1);
        sent++;
    }

    cudaDeviceSynchronize();

    cudaFree(ctx->d_step_count);
    cudaFree(ctx->d_sample_st);
    model_state_destroy(ctx->d_hidden_buf);
    query_producer_destroy(ctx->d_query_proj);
    cudaFree(ctx->d_query_buf);
    cudaFree(ctx->d_output_buf);
    ring_destroy(ctx->cmd_ring);
    ring_destroy((CommandRing *)ctx->comp_ring);
    arena_destroy(&ctx->kv_arena);
}

static int dispatch_decode_step(BenchContext *ctx, uint32_t rollout_id,
                                uint32_t step, uint32_t kv_block,
                                uint32_t total_tokens)
{
    (void)total_tokens;
    uint32_t pos = ring_acquire(ctx->cmd_ring, 1);
    if (pos == UINT32_MAX) {
        METRIC_INC(ctx->metrics, ring_full_spins);
        Completion c;
        while (comp_ring_poll(ctx->comp_ring, &c))
            METRIC_INC(ctx->metrics, descriptors_consumed);
        pos = ring_acquire(ctx->cmd_ring, 1);
        if (pos == UINT32_MAX)
            return -1;
    }

    Descriptor desc;
    uint32_t slot = step & (ctx->io_slots - 1U);
    if (model_state_prepare_slot(ctx->d_hidden_buf, ctx->io_slots, rollout_id,
                                 step, slot) != 0)
        return -1;
    if (query_producer_prepare_slot(ctx->d_hidden_buf, ctx->d_query_buf,
                                    ctx->d_query_proj, ctx->io_slots,
                                    slot) != 0)
        return -1;
    desc.seq_id            = rollout_id;
    desc.kv_block_offset   = kv_block;
    desc.num_kv_blocks     = 1;
    desc.attention_flags   = 0;
    desc.pad               = 0;
    desc.output_token_offset = step;
    desc.reward_cookie     = (uint64_t)rollout_id << 32 | step;
    ctx->cmd_ring->slots[pos] = desc;
    ring_commit(ctx->cmd_ring, 1);
    METRIC_INC(ctx->metrics, descriptors_posted);
    return 0;
}

static int drain_completions(BenchContext *ctx)
{
    int n = 0;
    Completion c;
    while (comp_ring_poll(ctx->comp_ring, &c)) {
        METRIC_INC(ctx->metrics, descriptors_consumed);
        n++;
    }
    return n;
}

int main(int argc, char **argv)
{
    int n_rollouts = 1000;
    int tokens_per_rollout = 128;
    int dev_id = 0;

    int opt;
    while ((opt = getopt(argc, argv, "r:t:d:")) != -1) {
        switch (opt) {
        case 'r': n_rollouts        = atoi(optarg); break;
        case 't': tokens_per_rollout = atoi(optarg); break;
        case 'd': dev_id            = atoi(optarg); break;
        }
    }

    int n_tokens = n_rollouts * tokens_per_rollout;

    printf("GB300 RL Runtime — Pipeline Benchmark\n");
    printf("  Rollouts:         %d\n", n_rollouts);
    printf("  Tokens/rollout:   %d\n", tokens_per_rollout);
    printf("  Total tokens:     %d\n", n_tokens);
    printf("  Device:           %d\n\n", dev_id);

    BenchContext ctx;
    if (bench_init(&ctx, dev_id, n_rollouts) != 0)
        return 1;

    hp_guard_activate(&ctx.hp_guard);

    uint64_t t0 = now_ns();
    PageFaultSnapshot pf_start, pf_end, pf_delta;
    metrics_snapshot_page_faults(&pf_start);

    for (int r = 0; r < n_rollouts; r++) {
        uint32_t rid;
        if (rollout_alloc(&ctx.pipeline.slab, &rid) != 0) {
            fprintf(stderr, "rollout_alloc failed at %d\n", r);
            break;
        }

        rollout_t *ro = rollout_get(&ctx.pipeline.slab, rid);
        ro->max_tokens = tokens_per_rollout;
        ro->rng_seed   = 42 + r;

        rollout_transition(ro, ROLL_FREE, ROLL_PREFILL_READY);
        pipeline_push(&ctx.pipeline, Q_PREFILL, rid);

        rollout_transition(ro, ROLL_PREFILL_READY, ROLL_DECODING);
        pipeline_push(&ctx.pipeline, Q_DECODE, rid);

        int kv_block = r % 128;
        for (int t = 0; t < tokens_per_rollout; t++) {
            if (dispatch_decode_step(&ctx, rid, t, kv_block, tokens_per_rollout) != 0) {
                METRIC_INC(&ctx.metrics, pipeline_overflow);
                break;
            }
            kv_block = (kv_block + 1) % 128;
        }

        rollout_transition(ro, ROLL_DECODING, ROLL_REWARD_PENDING);
        pipeline_push(&ctx.pipeline, Q_REWARD, rid);

        RewardDesc rd;
        rd.rollout_id     = rid;
        rd.token_start    = 0;
        rd.token_count    = tokens_per_rollout;
        rd.reward_model_id = 0;
        rd.reward         = reward_score_mock(NULL, tokens_per_rollout);
        rd.flags          = 0;
        reward_push(&ctx.reward_ring, &rd);

        rollout_transition(ro, ROLL_REWARD_PENDING, ROLL_TRAJECTORY_READY);
        pipeline_push(&ctx.pipeline, Q_TRAJECTORY, rid);

        rollout_transition(ro, ROLL_TRAJECTORY_READY, ROLL_DONE);
        pipeline_push(&ctx.pipeline, Q_DONE, rid);

        METRIC_INC(&ctx.metrics, rollouts_completed);
    }

    uint64_t expected_completions = METRIC_READ(&ctx.metrics, descriptors_posted);
    uint64_t comps = 0;
    while (comps < expected_completions)
        comps += (uint64_t)drain_completions(&ctx);

    hp_guard_deactivate(&ctx.hp_guard);

    uint64_t t1 = now_ns();
    uint64_t wall_ns = t1 - t0;
    metrics_snapshot_page_faults(&pf_end);
    metrics_diff_page_faults(&pf_delta, &pf_start, &pf_end);

    ctx.metrics.completion_overflow_attempts = ctx.comp_ring->overflow.value;

    metrics_fprintf(stdout, &ctx.metrics, wall_ns, n_tokens, n_rollouts);

    printf("\n  Completions drained:  %lu\n", (unsigned long)comps);
    printf("  Wrapper-tracked mallocs: %lu\n",
           (unsigned long)ctx.hp_guard.malloc_count);
    printf("  Wrapper-tracked cudaMallocs: %lu\n",
           (unsigned long)ctx.hp_guard.cuda_malloc_count);
    printf("  Wrapper-tracked page faults: %lu\n",
           (unsigned long)ctx.hp_guard.page_fault_count);
    if (pf_delta.supported) {
        printf("  OS page faults (minor/major): %lu / %lu\n",
               (unsigned long)pf_delta.minor_faults,
               (unsigned long)pf_delta.major_faults);
    }

    if (ctx.hp_guard.malloc_count == 0 && ctx.hp_guard.cuda_malloc_count == 0)
        printf("  Hot path wrappers:    WRAPPER-CLEAN (not a global guarantee)\n");
    else
        printf("  Hot path wrappers:    WRAPPER VIOLATIONS DETECTED\n");

    printf("\nDone.\n");

    bench_shutdown(&ctx);
    return 0;
}
