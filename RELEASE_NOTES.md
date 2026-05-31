# Release Notes

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
