# RTL Control-Plane Model

This document describes the SystemVerilog control-plane model included
in `rtl/`.

## Scope Boundary

This RTL models the queue/control-plane protocol for RL inference. It
does not model transformer compute, tensor cores, GB300 internals, or
any NVIDIA-private doorbell mechanism.

The intended mapping is:

```text
C infer_submit_decode()
    -> hw_desc_t
    -> command ring
    -> MMIO doorbell
    -> worker/device consumes descriptor
    -> completion ring
    -> host observes completion

RTL control-plane model
    -> host descriptor input
    -> desc_ring
    -> rollout_worker_fsm
    -> completion_ring
    -> host completion output
```

## Module Map

- `rtl/desc_pkg.sv`
- `rtl/mmio_regs.sv`
- `rtl/desc_ring.sv`
- `rtl/completion_ring.sv`
- `rtl/rollout_worker_fsm.sv`
- `rtl/rl_runtime_top.sv`
- `rtl/tb_rl_runtime_top.sv`

## Descriptor Format

The RTL descriptor is intentionally fixed and packed. The goal is to
make queue traffic explicit and deterministic, not configurable.

The important current limitation is that the RTL descriptor is not yet
the same byte-level wire format as the 64-byte software-side
`hw_desc_t`. The co-simulation bridge validates a shared logical
control-plane contract today, while the repo still carries separate C
and RTL physical encodings. See `docs/descriptor_contract.md`.

Key fields:

- opcode
- flags
- rollout ID
- KV arena and prefix IDs
- KV and delta offsets
- sequence length and max tokens
- reward model ID

## Doorbell Semantics

`mmio_regs.sv` models a tiny MMIO-style doorbell register block.

- write to `DOORBELL_ADDR`
- latch the new value
- pulse `doorbell_pulse` for one cycle

This is a protocol model for:

```c
*(volatile uint32_t *)doorbell = tail;
```

It is not claiming access to NVIDIA hardware doorbells.

## Ring Full/Empty Logic

Both rings use head/tail pointers with a wrap bit.

- empty: `head == tail`
- full: same low bits, different wrap bit

That keeps the RTL semantics aligned with the software ring invariants:

- do not overwrite unread descriptors
- do not lose unread completions

## Worker FSM Behavior

`rollout_worker_fsm.sv` is deliberately fake decode:

- accept a descriptor in `S_IDLE`
- for `DESC_OP_DECODE`, increment token count once per cycle
- emit `DONE` when `max_tokens` is reached
- emit `REWARD_NEEDED` at a reward boundary
- stall safely in `S_COMPLETE` until downstream is ready

The point is to model control-plane progression, not inference math.

## Completion Backpressure

Completion backpressure matters because a real host or downstream queue
may not be ready every cycle.

The model therefore guarantees:

- a completion can stall in the worker
- the completion ring does not silently overwrite data
- the host can delay `host_comp_ready` without losing a result

## Testbench Strategy

`tb_rl_runtime_top.sv` covers:

- decode completion
- reward-needed boundary
- completion backpressure retention

That gives one fast protocol-level regression loop before bringing in
co-simulation.

## Verilator Bridge

The repo now also includes a C++ Verilator bridge around
`rl_runtime_top`.

Bridge API:

- `reset()`
- `tick()`
- `submit_decode(...)`
- `submit_stop()`
- `poll_completion(...)`
- `set_completion_ready(...)`

Expected co-sim output:

```text
RTL co-sim basic decode: PASS
RTL co-sim reward boundary: PASS
RTL co-sim completion backpressure: PASS
RTL bridge tests: PASS
```

If enabled with tracing, the bridge writes:

```text
build/rtl_bridge.vcd
```

## Roadmap

- `v0.1`: basic descriptor engine
- `v0.2`: doorbell-visible tail
- `v0.3`: multi-worker dispatch
- `v0.4`: reward/done split
- `v0.5`: C + Verilator co-simulation
