# RTL Descriptor Engine

This directory contains a SystemVerilog model of the hardware-facing
control plane for RL inference descriptors.

## What This Is

- a descriptor package with fixed packed structs
- a descriptor ring and completion ring
- an MMIO-style doorbell register model
- a rollout worker FSM that simulates decode progress
- a top-level control-plane pipeline and a simple testbench

## What This Is Not

- not a GPU
- not transformer math
- not GB300 internals
- not NVIDIA doorbells

This models the queue/control-plane protocol: descriptors, rings,
doorbells, worker state progression, completions, and backpressure.

## Architecture

```text
host descriptor
    -> desc_ring
    -> rollout_worker_fsm
    -> completion_ring
    -> host completion
```

## Modules

- `desc_pkg.sv`: packed descriptor and completion definitions
- `mmio_regs.sv`: one-register MMIO-style doorbell model
- `desc_ring.sv`: descriptor FIFO with ready/valid behavior
- `completion_ring.sv`: completion FIFO with backpressure
- `rollout_worker_fsm.sv`: fake decode / control-plane worker
- `rl_runtime_top.sv`: top-level composition of ring -> worker -> completion
- `tb_rl_runtime_top.sv`: basic testbench

## Ready/Valid Backpressure

The rings and worker use ready/valid-style flow control:

- producer asserts `*_valid` when data is available
- consumer asserts `*_ready` when it can accept data
- transfer occurs only when both are high in the same cycle

This is the hardware-side analogue of the software ring invariants:
never overwrite unread descriptors and never drop unread completions.

## How To Run

```bash
make rtl-test
```

Expected output:

```text
PASS
```

If `iverilog` is not installed, the Makefile prints a helpful message.

## Future Work

The next logical step is C + Verilator co-simulation so
`infer_submit_decode()` can drive the RTL descriptor engine directly
from software tests and benchmarks.
