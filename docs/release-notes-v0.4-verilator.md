# v0.4.0 — C/RTL Verilator Co-Simulation Bridge

- Added Verilator bridge for RTL descriptor engine.
- Added C++ `RtlRuntimeBridge` wrapper.
- Host simulation can submit decode descriptors into Verilated `rl_runtime_top`.
- RTL worker FSM produces completions observed by host bridge.
- Added tests for basic decode, reward boundary, completion backpressure, and multiple descriptors.
- Added Makefile targets: `verilate`, `sim-rtl`, `test-rtl-bridge`, `clean-verilator`.
- Added docs for signal mapping and co-simulation limitations.
- Added CI job for Verilator co-simulation.
- Clarified that co-sim validates the control-plane contract, not transformer compute or GB300 internals.
