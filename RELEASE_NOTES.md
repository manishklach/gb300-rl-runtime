# Release Notes

## Unreleased

### Decode Optimization Pass

- fixed the fixed128 decode path so the lane-striped prefetch helper is
  now executed by the whole warp rather than lane 0 only
- replaced the old score-buffer plus probability-buffer flow with a
  tiled online softmax accumulation path that rescales prior partial
  output and fuses normalization with V accumulation
- updated decode docs to reflect the more hardware-shaped fixed128 path
- verified the CPU smoke suite still passes after the kernel-side change

### Hardware Fastpath Layer

- added `include/mmio.h` and `src/mmio.c` for host-side MMIO-style
  doorbell helpers and write barriers
- added a 64-byte `hw_desc_t` in `include/hw_desc.h`
- added `include/hw_ring.h` and `src/hw_ring.c` for a cacheline-owned
  SPSC hardware descriptor ring
- added `include/infer_submit.h` and `src/infer_submit.c` for a
  hardware-facing inference submit API
- added `include/hw_worker_sim.h` and `src/hw_worker_sim.c` for a
  CPU-only device-model worker
- added `test/test_hw_ring.c` and `bench/bench_hw_fastpath.c`
- updated the Makefile, README, and metrics docs to expose the new path

## v0.3.2

This release focuses on benchmark readiness, CPU-only verification, and
non-GPU CUDA validation. It does not claim new end-to-end performance
results yet; instead, it strengthens the artifact so measured numbers
can be added with less ambiguity and less breakage risk.

### Highlights

- added a dedicated `bench-prefetch` CUDA microbenchmark to isolate
  global-to-shared KV staging cost from the rest of the decode path
- added `lab/06_memory_ordering`, a standalone experiment that makes
  the runtime's release/acquire publication rules visible and testable
- added CPU-only GitHub Actions coverage via `.github/workflows/ci.yml`
- added `cuda-compile-check` and `cuda-ptx-check` targets for machines
  that have the CUDA toolkit but no attached GPU
- tightened README and benchmark docs so placeholders are clearly
  labeled as hardware-to-fill measurements rather than published claims

### Benchmarks and Validation

- added `bench/bench_prefetch.cu` and `make bench-prefetch` to compare a
  baseline shared-memory staging path against a `cp.async`-based path
- added `make ci-build` and `make ci-run` so the control-plane code,
  smoke tests, and CPU-safe labs can be built and exercised in one
  repeatable path
- added `make cuda-compile-check` to compile the CUDA runtime, CUDA test
  entrypoint, and CUDA benchmark entrypoints with `nvcc` without
  requiring GPU execution
- added `make cuda-ptx-check` to emit PTX for the core device-heavy CUDA
  translation units for low-level inspection

### Lab and Documentation Improvements

- added `lab/06_memory_ordering` to demonstrate the difference between a
  broken relaxed publish and the runtime's correct release/acquire
  publication contract
- updated the lab Makefiles to use a stricter C11 build mode and accept
  `ARGS`, which makes the labs more portable and easier to run in CI
- updated README, `docs/metrics.md`, and release notes so the repo now
  documents:
  - the sixth lab
  - `bench-prefetch`
  - `ci-build` / `ci-run`
  - `cuda-compile-check` / `cuda-ptx-check`

### CPU-Path Fixes Surfaced by CI

The new CPU verification path exposed several issues that were fixed as
part of this release:

- fixed `bench_control_tax.cu` so its polling modes use real helper
  threads instead of hanging in a self-polled loop
- fixed lab 4's `eventfd` consumer to handle coalesced wakeups
  correctly
- fixed lab 4's polling path to use a proper atomic handshake rather
  than a lossy single-bit overwrite pattern
- fixed multiple lab build issues under stricter Linux/C11 compilation,
  including alignment declarations and one thread/timer naming conflict

### Benchmark Honesty

- the benchmark snapshot remains intentionally conservative: it is still
  a run-on-your-hardware template until real measured numbers are
  checked in
- the repo now distinguishes more clearly between:
  - CPU-only correctness and control-plane verification
  - `nvcc` compile/PTX validation without a GPU
  - actual CUDA runtime benchmarking on real hardware

## v0.3.1

This release consolidates the post-`v0.3.0` correctness, decode, and
pipeline work into a more honest and more hardware-shaped runtime
baseline.

### Highlights

- corrected and hardened the SPSC queue semantics used by the runtime
  control path
- made shutdown safer by synchronizing persistent-worker exit before
  cleanup
- added overflow detection and backpressure for completion and done
  rings
- upgraded the fixed128 decode path from scaffold to real QK / softmax /
  V math
- split the pre-decode path into explicit `model_state` and
  `query_producer` stages
- made the decode kernel warp-cooperative instead of mostly lane-0
  driven
- added decode queue snapshots, batch-window helpers, batch-driven
  decode admission, and grouped descriptor window submission
- added a GitHub Pages site under `docs/index.html`

### Correctness and Stability

- fixed command-ring accounting so producer free-space checks observe the
  consumer-owned head and consumer availability checks observe the
  producer-owned tail
- fixed warp broadcast handling in the worker path to use
  `__shfl_sync`-style value broadcast rather than treating
  `__sync_warp` as a value-producing primitive
- ensured runtime and benchmark shutdown paths wait for persistent CUDA
  workers to exit before freeing host or device resources
- added overflow counting to completion and done rings so slow
  consumers are visible instead of silently overwritten
- expanded CPU smoke coverage to include:
  - empty/full ring behavior
  - wraparound
  - producer observing consumer progress
  - no permanent-full condition
  - completion and done ring overflow
  - pipeline snapshot and batch-window helpers
  - grouped descriptor window submission
- corrected the stale `rollout_t` size assertion when the rollout and
  pipeline code was pulled into the smoke target

### Decode Path and Hardware-Close Runtime Work

- added a fixed-shape KV layout and the first real decode microkernel
  path
- upgraded the decode microkernel from metadata scaffold to real
  single-token attention math for one fixed configuration
- added explicit device-side query and output buffer plumbing to the
  runtime-style path
- introduced `model_state` and `query_producer` modules so the path is
  now:
  `descriptor -> model_state -> query_producer -> decode`
- upgraded the synthetic model-state stage from a simple scalar mix to a
  tiny residual-style weighted block
- made the fixed128 decode kernel warp-cooperative for query load,
  score generation, softmax normalization, output accumulation, and
  argmax selection
- deepened the QK score pass further by using 8-lane groups per token
  row
- added explicit prefetch pipeline helper state and stage helpers so
  worker-side prefetch code is less ad hoc

### Pipeline and Scheduling Work

- added `PipelineSnapshot` so queue occupancy and stage credit headroom
  can be inspected explicitly
- added decode batch-window helpers to estimate how much decode work can
  be admitted safely
- updated pipeline benchmarks so decode admission is credit-gated and
  drained in batches rather than immediately dispatching each rollout
- added grouped host-side descriptor window submission so multiple
  decode steps can be published with one ring commit
- added metrics and tracing for decode batches, scheduled rollout count,
  and batch peak size

### Documentation and Repo UX

- tightened README and release-note wording to distinguish real,
  partial, and scaffolded components more honestly
- documented decode-path status in `docs/decode_microkernel.md`
- added a detailed static project page in `docs/index.html`
- enabled the repository for GitHub Pages publishing from `master/docs`

### Current Limitations

- the decode path is still intentionally narrow:
  - one fixed head dimension
  - one warp
  - one staged KV block
  - one correctness-first kernel shape
- model-state and query production are still deterministic scaffolding,
  not trained model activations
- the new batch-window and descriptor-window helpers are host-side
  scheduling structure, not yet a full GPU-visible batch contract
- broader reward/model semantics remain partial or stubbed

### Follow-On Low-Level Work

- refined the `cp.async` prefetch path so it now uses lane-striped
  16-byte copy helpers with explicit commit/wait flow instead of the old
  oversized placeholder inline assembly
- added a device-visible decode batch contract by stamping
  `batch_size` and `batch_index` onto grouped descriptors at submission
  time and consuming that metadata inside the worker

### Verification Notes

- `make smoke` passes on the supported CPU-only path
- CUDA compile/runtime verification is still environment-dependent and
  requires `nvcc` plus a CUDA-capable device
- benchmark claims remain focused on control-path behavior and the
  specific fixed decode path rather than broad model-serving
  performance

## v0.2.2-f

This checkpoint adds another layer of pipelining work around the decode
path, both inside the kernel and in the host control plane.

- the fixed128 score pass now uses 8-lane groups so each token row is
  computed cooperatively instead of by a single lane
- added prefetch-pipeline helper structs/functions in
  `include/prefetch.h` and `cu/prefetch.cu` so worker-side staging code
  has explicit pipeline state instead of raw shared-memory pointers
- added pipeline snapshot and batch-window helpers in
  `include/pipeline.h` and `src/pipeline.c`
- expanded the CPU smoke target to compile pipeline/rollout code too,
  which also surfaced and fixed the stale `rollout_t` size assertion
- pipeline benchmarks now use decode credits plus batch-window helpers
  to admit and drain decode work in small batches instead of dispatching
  every rollout immediately after enqueue
- pipeline benchmarks now report or trace decode queue occupancy,
  credit headroom, and a suggested decode batch window
- added host-side descriptor window helpers in `include/decode_batch.h`
  and `src/decode_batch.c`, and wired runtime-style paths to submit
  grouped descriptor windows with one ring commit per window

Current limitation:

- the decode kernel is still a single-warp path for one fixed shape and
  one staged KV block
- the new host-side pipeline helpers expose queue/batch state, but they
  do not yet drive a fully batched decode scheduler

## v0.2.2-e

This checkpoint makes the fixed128 decode kernel less reference-style
and more hardware-shaped by turning the inner attention math into a
warp-cooperative path.

- `cu/attention_decode.cu` now spreads work across the warp instead of
  leaving almost all math on lane 0
- score generation, softmax normalization, and output-vector
  accumulation now execute cooperatively across lanes
- `bench/bench_decode_microkernel.cu` and the CUDA correctness test were
  updated to exercise the warp-cooperative path directly

Current limitation:

- the decode path is still one warp, one KV block, one fixed head
  dimension, and one correctness-first layout
- the score pass still assigns one lane per token row, so this is a
  meaningful step toward hardware-shaped execution, not a fully tuned
  kernel

## v0.2.2-d

This checkpoint separates synthetic activation preparation from query
projection so the decode path more closely resembles a real inference
pipeline.

- added `include/model_state.h` and `cu/model_state.cu`
- runtime, pipeline benchmarks, and CUDA pipeline tests now execute
  `model_state -> query_producer -> decode` instead of generating hidden
  state inside the projection layer
- `model_state` now owns explicit fixed-shape projection weights and a
  bias term for a tiny residual update block instead of a single
  hand-written scalar mix
- kept the stage explicitly documented as synthetic scaffolding rather
  than a true transformer prefill/update path

Current limitation:

- model-state preparation is still a deterministic synthetic update rule
  rather than a real residual stream / MLP / attention stack from a
  trained model
- the real decode path remains intentionally narrow: one fixed head
  dimension, one staged KV block, one correctness-first execution path

## v0.2.2-c

This checkpoint removes the last direct host-filled query staging from
the runtime-shaped path and replaces it with a tiny explicit query
producer stage.

- added `include/query_producer.h` and `cu/query_producer.cu`
- runtime, pipeline benchmarks, and CUDA pipeline tests now prepare
  decode query slots through resident hidden-state and projection buffers
  on the device side instead of host-side direct query fills
- kept the new stage documented as synthetic scaffolding rather than a
  full model forward pass

Current limitation:

- query production is now an explicit runtime stage, but it still sits
  on top of synthetic hidden-state scaffolding rather than real
  transformer activations
- the real decode path is still intentionally narrow: one fixed head
  dimension, one staged KV block, one correctness-first execution path

## v0.2.2-b

This checkpoint upgrades the fixed128 decode path from a pure scaffold to
one mathematically real kernel path for a narrow configuration.

- `cu/attention_decode.cu` now performs real QK / softmax / V
  accumulation math for one fixed-shape decode step
- `bench/bench_decode_microkernel.cu` now exercises explicit query and
  output buffers and reports real math-path measurements
- `test/test_bench.cu` now includes a CUDA-vs-reference correctness test
  for the fixed128 decode path

Current limitation:

- the runtime path now carries explicit query/output buffers, but it
  still depends on synthetic query values rather than real model
  activations
- the real path is still intentionally narrow: one fixed head dimension,
  one staged KV block, one lane-0 style correctness-first implementation

## v0.2.2-a scaffold

This checkpoint adds the first repository scaffolding for a real,
hardware-close decode path without claiming that the attention math is
implemented yet.

- added `include/kv_layout.h` for one fixed-shape KV block layout
- added `include/attention_decode.h` and `cu/attention_decode.cu`
- routed `cu/worker.cu` through the fixed128 decode helper scaffold
- added `make bench-decode` and `make bench-kv-layout`
- added `docs/decode_microkernel.md` to document what is real versus
  still stubbed

The decode helper currently stages data through shared memory and reports
deterministic metadata, but it does not yet implement QK/softmax/V
attention math.

## Up Next: v0.2.2

Planned next step:

- add one real, fixed-shape decode math path
- keep control-plane and math-path benchmarks separate
- add hardware-shaped KV layout docs and benchmarks
- add CUDA correctness tests against a tiny reference implementation

See `docs/v0.2.2-roadmap.md` for the file-by-file breakdown.

## v0.2.1

This release is a correctness and stabilization pass.  The goal is to
make the queue semantics, persistent-worker lifecycle, and benchmark
claims match what the code actually guarantees today.

### Correctness fixes

- Fixed SPSC command-ring accounting in `include/ring.h` so producer
  free-space checks read the consumer-owned head and publish the
  producer-owned tail, while consumer availability checks read the
  producer-owned tail and advance the consumer-owned head.
- Fixed `cu/worker.cu` warp broadcast logic to use `__shfl_sync` for
  value broadcast instead of incorrectly treating `__sync_warp` as a
  value-returning primitive.
- Added shutdown synchronization for persistent kernels before freeing
  host/device resources in runtime and benchmark paths.
- Added full-condition checks and overflow counters to completion/done
  rings.  GPU producers now apply backpressure instead of silently
  overwriting unread entries.

### Testing

- Added CPU smoke tests for:
  - empty/full command-ring behavior
  - wraparound
  - producer observes consumer progress
  - no permanent-full condition across repeated fill/drain cycles
  - slow-consumer overflow behavior for completion and done rings
- Added a CUDA warp-broadcast self-test and updated the CUDA test suite
  to skip GPU-only checks when no CUDA device is available.
- Added `make smoke` for CPU-only verification on supported Linux hosts.

### Benchmark honesty

- Hot-path guard output now reports `wrapper-clean` rather than `CLEAN`
  when only the explicit wrapper hooks stayed unused.
- Added optional `getrusage` page-fault snapshots around the pipeline
  benchmark on supported platforms.
- Tightened benchmark and README wording so stubbed attention/reward
  paths are not presented as full model-performance results.

### Portability notes

- CUDA kernels still target compute capability 8.0+.
- The host runtime and many tests/benchmarks currently assume Linux/POSIX
  APIs such as `mmap`, `MAP_HUGETLB`, `mbind`, `clock_gettime`, and
  `getopt`.
