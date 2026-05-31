#include "pipeline.h"
#include "metrics.h"
#include "hotpath_guard.h"
#include "reward.h"
#include "kv_prefix.h"
#include "trace.h"
#include "ring.h"
#include "arena.h"
#include "completion.h"
#include "decode_batch.h"
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
    TraceRing        trace;
    CommandRing     *cmd_ring;
    CompletionRing  *comp_ring;
    KVArena          kv_arena;
    RewardRing       reward_ring;
    uint64_t        *d_step_count;
    SampleState     *d_sample_st;
    ModelStateBuffers model_state;
    __half          *d_query_proj;
    __half          *d_query_buf;
    float           *d_output_buf;
    uint32_t         io_slots;
    int              num_sms;
    int              dev_id;
} BenchContext;

#define DECODE_BATCH_LIMIT 32U

static void trace_pipeline_window(const RolloutPipeline *pipeline, TraceRing *trace,
                                  uint32_t max_decode_batch)
{
    PipelineSnapshot snap;
    pipeline_snapshot(pipeline, &snap);
    trace_push(trace, TRACE_PIPELINE_PUSH, snap.queue_occupancy[Q_DECODE],
               pipeline_stage_target_batch(pipeline, Q_DECODE, max_decode_batch));
}

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
    trace_init(&ctx->trace);

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
    if (model_state_init(&ctx->model_state, ctx->io_slots) != 0) {
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
    model_state_destroy(&ctx->model_state);
    query_producer_destroy(ctx->d_query_proj);
    cudaFree(ctx->d_query_buf);
    cudaFree(ctx->d_output_buf);
    ring_destroy(ctx->cmd_ring);
    ring_destroy((CommandRing *)ctx->comp_ring);
    arena_destroy(&ctx->kv_arena);
}

static int dispatch_decode_window(BenchContext *ctx, uint32_t rollout_id,
                                  uint32_t step_start, uint32_t step_count,
                                  uint32_t kv_block_base,
                                  uint32_t total_tokens)
{
    DecodeDispatchBatch batch;
    (void)total_tokens;
    decode_batch_reset(&batch);

    if (step_count > DECODE_DESCRIPTOR_BATCH_LIMIT)
        step_count = DECODE_DESCRIPTOR_BATCH_LIMIT;

    while (batch.count < step_count) {
        Descriptor desc;
        uint32_t step = step_start + batch.count;
        uint32_t kv_block = (kv_block_base + batch.count) % 128U;
        uint32_t slot = step & (ctx->io_slots - 1U);
        if (model_state_prepare_slot(&ctx->model_state, ctx->io_slots, rollout_id,
                                     step, slot) != 0)
            return -1;
        if (query_producer_prepare_slot(ctx->model_state.hidden_buf, ctx->d_query_buf,
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
        desc.batch_size        = 0;
        desc.batch_index       = 0;
        if (decode_batch_push(&batch, &desc) != 0)
            return -1;
    }

    if (decode_batch_submit(ctx->cmd_ring, &batch) != 0) {
        METRIC_INC(ctx->metrics, ring_full_spins);
        Completion c;
        while (comp_ring_poll(ctx->comp_ring, &c)) {
            METRIC_INC(ctx->metrics, descriptors_consumed);
            trace_push(&ctx->trace, TRACE_COMPLETION_POLLED,
                       (uint32_t)(c.seq_id & 0xFFFFFFFF), c.token_id);
        }
        if (decode_batch_submit(ctx->cmd_ring, &batch) != 0)
            return -1;
    }
    for (uint32_t i = 0; i < batch.count; i++) {
        trace_push(&ctx->trace, TRACE_DESC_POSTED, rollout_id,
                   batch.descs[i].output_token_offset);
        trace_push(&ctx->trace, TRACE_DESC_COMMITTED, rollout_id,
                   batch.descs[i].output_token_offset);
    }
    __atomic_add_fetch(&ctx->metrics.descriptors_posted, batch.count, __ATOMIC_RELAXED);
    return 0;
}

static int drain_completions(BenchContext *ctx)
{
    int n = 0;
    Completion c;
    while (comp_ring_poll(ctx->comp_ring, &c)) {
        METRIC_INC(ctx->metrics, descriptors_consumed);
        trace_push(&ctx->trace, TRACE_DESC_CONSUMED,
                   (uint32_t)(c.seq_id & 0xFFFFFFFF), c.token_id);
        trace_push(&ctx->trace, TRACE_COMPLETION_POLLED,
                   (uint32_t)(c.seq_id & 0xFFFFFFFF), c.token_id);
        n++;
    }
    return n;
}

static void note_decode_batch_metrics(RuntimeMetrics *metrics, uint32_t batch_size)
{
    METRIC_INC(metrics, decode_batches);
    __atomic_add_fetch(&metrics->decode_rollouts_scheduled, batch_size, __ATOMIC_RELAXED);
    uint64_t peak = METRIC_READ(metrics, decode_batch_peak);
    while (peak < batch_size &&
           !__atomic_compare_exchange_n(&metrics->decode_batch_peak, &peak,
                                        batch_size, 0, __ATOMIC_RELAXED,
                                        __ATOMIC_RELAXED)) {
    }
}

static void finish_rollout(BenchContext *ctx, uint32_t rid, uint32_t tokens_per_rollout)
{
    rollout_t *ro = rollout_get(&ctx->pipeline.slab, rid);
    if (!ro)
        return;

    rollout_transition(ro, ROLL_DECODING, ROLL_REWARD_PENDING);
    pipeline_push(&ctx->pipeline, Q_REWARD, rid);

    RewardDesc rd;
    rd.rollout_id     = rid;
    rd.token_start    = 0;
    rd.token_count    = tokens_per_rollout;
    rd.reward_model_id = 0;
    rd.reward         = reward_score_mock(NULL, tokens_per_rollout);
    rd.flags          = 0;
    reward_push(&ctx->reward_ring, &rd);
    METRIC_INC(&ctx->metrics, reward_handoffs);
    trace_push(&ctx->trace, TRACE_REWARD_POSTED, rid, tokens_per_rollout);
    trace_push(&ctx->trace, TRACE_REWARD_SCORED, rid, tokens_per_rollout);

    rollout_transition(ro, ROLL_REWARD_PENDING, ROLL_TRAJECTORY_READY);
    pipeline_push(&ctx->pipeline, Q_TRAJECTORY, rid);

    rollout_transition(ro, ROLL_TRAJECTORY_READY, ROLL_DONE);
    pipeline_push(&ctx->pipeline, Q_DONE, rid);

    METRIC_INC(&ctx->metrics, rollouts_completed);
    trace_push(&ctx->trace, TRACE_TRAJECTORY_DONE, rid, tokens_per_rollout);
    rollout_free(&ctx->pipeline.slab, rid);
    trace_push(&ctx->trace, TRACE_ROLLOUT_FREE, rid, 0);
}

static int schedule_decode_batches(BenchContext *ctx, uint32_t batch_max,
                                   uint32_t tokens_per_rollout)
{
    uint32_t batch = pipeline_stage_target_batch(&ctx->pipeline, Q_DECODE, batch_max);
    uint32_t rollout_ids[DECODE_BATCH_LIMIT];
    uint32_t launched = 0;

    if (batch > DECODE_BATCH_LIMIT)
        batch = DECODE_BATCH_LIMIT;
    if (batch == 0)
        return 0;

    for (uint32_t i = 0; i < batch; i++) {
        if (pipeline_schedule(&ctx->pipeline, Q_DECODE, &rollout_ids[launched]) != 0)
            break;
        launched++;
    }
    if (launched == 0)
        return 0;

    note_decode_batch_metrics(&ctx->metrics, launched);
    trace_push(&ctx->trace, TRACE_DECODE_BATCH, launched, tokens_per_rollout);

    for (uint32_t i = 0; i < launched; i++) {
        const uint32_t rid = rollout_ids[i];
        int kv_block = (int)(rid % 128U);
        for (uint32_t t = 0; t < tokens_per_rollout; t += DECODE_DESCRIPTOR_BATCH_LIMIT) {
            uint32_t window = tokens_per_rollout - t;
            if (window > DECODE_DESCRIPTOR_BATCH_LIMIT)
                window = DECODE_DESCRIPTOR_BATCH_LIMIT;
            if (dispatch_decode_window(ctx, rid, t, window, (uint32_t)kv_block,
                                       tokens_per_rollout) != 0) {
                METRIC_INC(&ctx->metrics, pipeline_overflow);
                break;
            }
            kv_block = (kv_block + (int)window) % 128;
        }
        finish_rollout(ctx, rid, tokens_per_rollout);
    }

    pipeline_release(&ctx->pipeline, Q_DECODE, launched);
    return (int)launched;
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

    printf("GB300 RL Runtime — Traced Pipeline Benchmark\n");
    printf("  Rollouts:         %d\n", n_rollouts);
    printf("  Tokens/rollout:   %d\n", tokens_per_rollout);
    printf("  Total tokens:     %d\n", n_tokens);
    printf("  Device:           %d\n\n", dev_id);

    BenchContext ctx;
    if (bench_init(&ctx, dev_id, n_rollouts) != 0)
        return 1;

    hp_guard_activate(&ctx.hp_guard);

    uint64_t t0 = now_ns();
    pipeline_credits_set(&ctx.pipeline, 256, 128, 128, 1024);

    for (int r = 0; r < n_rollouts; r++) {
        uint32_t rid;
        if (rollout_alloc(&ctx.pipeline.slab, &rid) != 0) {
            fprintf(stderr, "rollout_alloc failed at %d\n", r);
            break;
        }
        trace_push(&ctx.trace, TRACE_ROLLOUT_ALLOC, rid, 0);

        rollout_t *ro = rollout_get(&ctx.pipeline.slab, rid);
        ro->max_tokens = tokens_per_rollout;
        ro->rng_seed   = 42 + r;

        rollout_transition(ro, ROLL_FREE, ROLL_PREFILL_READY);
        pipeline_push(&ctx.pipeline, Q_PREFILL, rid);

        int decode_admitted = 0;
        rollout_transition(ro, ROLL_PREFILL_READY, ROLL_DECODING);
        while (!decode_admitted) {
            if (pipeline_try_push(&ctx.pipeline, Q_DECODE, rid) == 0) {
                decode_admitted = 1;
                break;
            }
            if (schedule_decode_batches(&ctx, DECODE_BATCH_LIMIT, (uint32_t)tokens_per_rollout) == 0) {
                METRIC_INC(&ctx.metrics, pipeline_overflow);
                break;
            }
        }
        if (!decode_admitted) {
            rollout_free(&ctx.pipeline.slab, rid);
            trace_push(&ctx.trace, TRACE_ROLLOUT_FREE, rid, 0);
            continue;
        }
        if ((uint32_t)((r + 1) % DECODE_BATCH_LIMIT) == 0U)
            schedule_decode_batches(&ctx, DECODE_BATCH_LIMIT, (uint32_t)tokens_per_rollout);
    }

    while (pipeline_occupancy(&ctx.pipeline, Q_DECODE) > 0)
        schedule_decode_batches(&ctx, DECODE_BATCH_LIMIT, (uint32_t)tokens_per_rollout);

    uint64_t expected_completions = METRIC_READ(&ctx.metrics, descriptors_posted);
    uint64_t comps = 0;
    while (comps < expected_completions)
        comps += (uint64_t)drain_completions(&ctx);

    hp_guard_deactivate(&ctx.hp_guard);

    uint64_t t1 = now_ns();
    uint64_t wall_ns = t1 - t0;

    ctx.metrics.completion_overflow_attempts = ctx.comp_ring->overflow.value;
    trace_pipeline_window(&ctx.pipeline, &ctx.trace, 32);

    metrics_fprintf(stdout, &ctx.metrics, wall_ns, n_tokens, n_rollouts);

    printf("\n  Completions drained:  %lu\n", (unsigned long)comps);
    printf("  Wrapper-tracked mallocs: %lu\n",
           (unsigned long)ctx.hp_guard.malloc_count);
    printf("  Wrapper-tracked cudaMallocs: %lu\n",
           (unsigned long)ctx.hp_guard.cuda_malloc_count);
    printf("  Wrapper-tracked page faults: %lu\n",
           (unsigned long)ctx.hp_guard.page_fault_count);

    if (ctx.hp_guard.malloc_count == 0 && ctx.hp_guard.cuda_malloc_count == 0)
        printf("  Hot path wrappers:    WRAPPER-CLEAN (not a global guarantee)\n");
    else
        printf("  Hot path wrappers:    WRAPPER VIOLATIONS DETECTED\n");

    trace_report_from(&ctx.trace, wall_ns, n_tokens, n_rollouts, "");

    printf("\nDone.\n");

    bench_shutdown(&ctx);
    return 0;
}
