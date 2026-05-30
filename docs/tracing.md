# Tracing & Latency Analysis

The runtime includes a lightweight ring-buffer trace system for
measuring end-to-end latencies across the RL inference pipeline.

## Design

Trace entries are 20-byte records stored in a fixed-size ring buffer
(1M entries, ~20 MB). Each entry records:

- `tsc` — monotonic nanosecond timestamp (via `clock_gettime`)
- `type` — event type from `trace_event_t`
- `rollout_id` — the rollout involved
- `seq` — sequence number (step index, token count, etc.)

The ring is single-producer (the benchmark thread), lock-free, and
uses a relaxed-store on entry write + release-store on head advance.
Overwrite is allowed if the ring wraps (oldest entries discarded).

## Event Types

| Event                  | When recorded |
|------------------------|---------------|
| `TRACE_ROLLOUT_ALLOC`  | Rollout allocated from slab |
| `TRACE_ROLLOUT_FREE`   | Rollout returned to slab |
| `TRACE_DESC_POSTED`    | Descriptor written to command ring slot |
| `TRACE_DESC_COMMITTED` | Ring head released (consumer can read) |
| `TRACE_DESC_CONSUMED`  | Completion polled for a decode step |
| `TRACE_COMPLETION_POLLED` | Completion ring entry read by CPU |
| `TRACE_REWARD_POSTED`  | Reward descriptor pushed to reward ring |
| `TRACE_REWARD_SCORED`  | Reward scored (mock or real) |
| `TRACE_TRAJECTORY_DONE` | Rollout trajectory finalized |

## Latency Pairs

The trace report computes percentiles for paired events:

| Pair | What it measures |
|------|------------------|
| DESC_POSTED → DESC_COMMITTED | Time to release descriptor to consumer |
| DESC_POSTED → DESC_CONSUMED | CPU → GPU descriptor delivery latency |
| DESC_POSTED → COMPLETION_POLLED | End-to-end decode step latency |
| DESC_CONSUMED → COMPLETION_POLLED | GPU decode + completion write time |
| ROLLOUT_ALLOC → REWARD_POSTED | Rollout lifetime before reward |
| ROLLOUT_ALLOC → TRAJECTORY_DONE | Full rollout lifetime |
| REWARD_POSTED → REWARD_SCORED | Reward scoring latency |
| REWARD_POSTED → TRAJECTORY_DONE | Reward → trajectory handoff latency |

## Running

```bash
make bench-trace                      # default: 1000 rollouts, 128 tokens each
make bench-trace ARGS="-r 5000 -t 256" # 5000 rollouts, 256 tokens each
```

## Example Output

```
── Trace Latency Report (256000 events, 0.423 s wall) ──
  descriptor post -> commit               n=128000 avg=42    p50=41    p90=44    p99=52    ns
  descriptor post -> GPU dequeue          n=128000 avg=1842  p50=1210  p90=3400  p99=8200  ns
  descriptor post -> completion polled    n=128000 avg=41500 p50=38900 p90=62000 p99=92000 ns
  GPU dequeue -> completion polled        n=128000 avg=39600 p50=37700 p90=61000 p99=90000 ns
  rollout alloc -> reward posted          n=1000   avg=28400 p50=26000 p90=44000 p99=68000 ns
  rollout alloc -> trajectory done        n=1000   avg=42100 p50=39800 p90=62000 p99=93000 ns
  reward posted -> scored                 n=1000   avg=5     p50=5     p90=5     p99=6     ns
  reward posted -> trajectory done        n=1000   avg=12    p50=11    p90=15    p99=22    ns
```

## Overhead

Each trace push is:
- 1 `clock_gettime` syscall (can be replaced with `__rdtsc` on x86)
- 1 relaxed atomic load + 1 release store
- ~20 bytes of memory bandwidth

For 1M entries this adds ~25 ms of CPU time. The ring is sized to
capture all events in a typical benchmark run without wrap.

## Future

- Replace `clock_gettime` with TSC/CNTPCT for sub-10 ns capture cost
- Add GPU-side trace buffer with post-hoc copyback for true device
  timestamps
- Flamegraph export via folded-format output
