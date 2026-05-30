# GPU-Resident Rollout Scheduler

## Motivation

In v0.2, the CPU manages rollout state transitions and pipeline
queues. The CPU posts one descriptor per decode step and polls
completions. This is already fast (no syscalls, no per-token kernel
launches), but it still involves the CPU on every token.

The GPU-resident scheduler eliminates the CPU from the per-token
path entirely:

```
v0.2 (CPU per token):
  CPU ──desc──► GPU ──comp──► CPU ──desc──► GPU ──comp──► ...

v0.3 (GPU resident):
  CPU ──request──► GPU ──────────────────────► GPU ──done──► CPU
                    │  decode/sample/KV loop   │
                    │  (all tokens, no CPU)    │
```

## Design

The CPU sends high-level `RolloutRequest` descriptors through a
request ring. GPU persistent workers manage the entire rollout
lifecycle — allocating rollout slots, running decode steps, sampling
tokens, managing KV blocks, and posting `RolloutDone` notifications
when trajectories complete.

```
 RequestRing (CPU→GPU)
 ┌─────────────────┐
 │ RolloutRequest  │  { request_id, max_tokens, rng_seed, ... }
 │ RolloutRequest  │
 │ ...             │
 └──────┬──────────┘
        │
        ▼
 ┌─────────────────────────────────────────────┐
 │         GPU persistent rollout_worker        │
 │                                              │
 │  for each warp:                              │
 │    poll request ring                         │
 │    find free GpuRolloutSlot                  │
 │    init slot with request params             │
 │                                              │
 │  for each active slot:                       │
 │    run one decode step                       │
 │    (mock: increment token counter)           │
 │    if max_tokens reached:                    │
 │      post RolloutDone to done ring           │
 │      free slot                               │
 └──────────────────┬──────────────────────────┘
                    │
                    ▼
 DoneRing (GPU→CPU)
 ┌─────────────────┐
 │ RolloutDone     │  { request_id, tokens_generated, reward, ... }
 │ RolloutDone     │
 │ ...             │
 └─────────────────┘
```

## Key Properties

| Property | v0.2 (CPU-managed) | v0.3 (GPU-resident) |
|----------|-------------------|-------------------|
| CPU per-token work | 2 atomic ops | 0 (batch at init + drain at end) |
| GPU-to-CPU trips | 2 per token (desc + comp) | 2 per rollout (request + done) |
| State machine location | Host memory | Device memory |
| Pipeline queues | Host rings | Device tables |
| Scheduling policy | Host-side pipeline_schedule | GPU-side slot scan |

## GPU-Side Rollout Table

Up to 256 concurrent rollouts stored in device memory:

```c
typedef struct {
    uint32_t active;
    uint32_t request_id;
    uint32_t tokens_generated;
    uint32_t max_tokens;
    uint32_t kv_blocks;
    uint64_t rng_state[4];
    float    temperature;
    uint16_t top_k;
    float    top_p;
} GpuRolloutSlot;
```

Allocation uses a per-warp bitmap (`__ffsll` + atomic OR). Each warp
processes one active slot per iteration in a strided loop.

## CPU Involvement

The CPU is involved only at two points:

1. **Request submission**: batch-write `RolloutRequest` structs to
   the request ring. One `ring_acquire` + write + `ring_commit` per
   request, all in pinned coherent memory.
2. **Done drainage**: poll the done ring for completed trajectory
   notifications. Each `RolloutDone` contains the final token count
   and mock reward score.

No CPU work scales with token count. CPU work scales with rollout
count.

## Benchmark

```
make bench-gpu-scheduler           # default: 10K rollouts, 128 tokens each
make bench-gpu-scheduler ARGS="-r 50000 -t 256"
```

Expected output:

```
── GPU Scheduler Results ──
  Wall time:          0.423 s
  Throughput:         23640 rollouts/s
  Throughput:         3026000 tokens/s
  GPU steps recorded: 1280000
  CPU dispatches:     1  (all rollouts batched)
  CPU polls:          10000  (done notifications)
  CPU per-token work: none  (GPU managed all decode steps)
```

## Comparison

| Benchmark | What it proves |
|-----------|----------------|
| `bench-tax` | Syscall vs polling vs persistent worker overhead |
| `bench-pipeline` | End-to-end RL flow with CPU-managed state machine |
| `bench-gpu-scheduler` | GPU-managed rollout lifecycle — zero CPU per-token work |

## Future

- Real decode kernels (attention, sampling) instead of mock counter
- KV block allocation on GPU (currently pre-assigned)
- Multi-warp scheduling for higher occupancy
- Dynamic priority based on rollout urgency
- Integration with the COW prefix KV table for branch rollouts
