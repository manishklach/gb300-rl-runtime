# Verilator Bridge

## Motivation

The Verilator bridge turns the RTL descriptor engine into a host-driven
co-simulation target. It gives the repo a vertical path from software
descriptor submission to RTL completion handling.

## Build Requirements

- Verilator
- C++ compiler
- Make

## How To Run

```bash
make verilate
make sim-rtl
make test-rtl-bridge
```

## Bridge API

`RtlRuntimeBridge` exposes:

- `reset()`
- `tick()`
- `submit_decode(...)`
- `submit_stop()`
- `poll_completion(...)`
- `set_completion_ready(...)`
- `cycles()`

## Signal Mapping

The bridge maps host-side fields into the packed RTL descriptor port and
unpacks the completion port back into `RtlCompletion`.

The bridge currently maps between the software-side hardware descriptor
intent and the RTL's smaller packed control-plane format. That means the
bridge is validating semantic agreement, not a byte-identical shared
wire format. The field-by-field contract and remaining gaps are tracked
in `docs/descriptor_contract.md`.

Descriptor mapping:

- opcode -> `DESC_OP_DECODE`
- rollout ID
- KV arena / prefix IDs
- offsets
- `seq_len`
- `max_tokens`
- `reward_model_id`

Completion mapping:

- rollout ID
- status
- final sequence length
- reward ID

## Test Scenarios

- basic decode completion
- reward boundary completion
- completion backpressure
- multiple descriptors in sequence
- submit readiness gate
- stop descriptor

## Limitations

- validates the control-plane contract only
- does not validate real transformer compute
- does not validate GB300 hardware behavior

## Future Work

- C wrapper on top of the C++ bridge
- stronger signal-level tracing
- C + Verilator integration with the runtime submit path
