# Verilator Co-Simulation Bridge

This directory contains the C++ bridge that drives the RTL descriptor
engine through Verilator.

## Motivation

The software runtime and the RTL descriptor engine now share a common
descriptor/control-plane contract. The Verilator bridge is the layer
that lets host-side code submit descriptors into the RTL model and
observe completions back out.

## What It Proves

- descriptor submission from host code
- descriptor consumption by the RTL ring + worker
- completion production and host polling
- backpressure behavior at the completion boundary

## What It Does Not Prove

- transformer math
- GPU execution
- GB300 hardware behavior
- NVIDIA internal doorbells

## Main Files

- `rtl_bridge.h`
- `rtl_bridge.cpp`
- `sim_main.cpp`
- `test_rtl_bridge.cpp`

## Build and Run

```bash
make verilate
make sim-rtl
make test-rtl-bridge
```

If Verilator is missing, the Makefile prints:

```text
Verilator not found. Install verilator to run C/RTL co-simulation.
```

## Optional VCD

```bash
make sim-rtl TRACE=1
```

Trace output:

```text
build/rtl_bridge.vcd
```
