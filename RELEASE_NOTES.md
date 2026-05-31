# Release Notes

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
