# v0.3.0 — RTL Descriptor Engine for RL Inference Control Plane

- Added SystemVerilog descriptor package.
- Added descriptor ring and completion ring.
- Added MMIO doorbell register model.
- Added rollout worker FSM with fake decode and reward boundary behavior.
- Added top-level RTL runtime module.
- Added RTL testbench covering decode completion, reward-needed boundary, and completion backpressure.
- Added `make rtl-test` target.
- Updated README and architecture docs.
- Clarified this is a control-plane model, not transformer compute or GB300 internals.
