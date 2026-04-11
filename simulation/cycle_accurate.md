# Cycle-Accurate Simulators

← [Simulation Index](index.md)

Cycle-accurate simulators model the GPU/NPU microarchitecture at the level of individual clock cycles. They are the highest-fidelity tools available without real hardware, at the cost of simulation speed (hours to days for large workloads).

---

## Accel-Sim {#accel-sim}

**What it is:** A trace-driven, cycle-accurate simulator for NVIDIA GPU architectures. It is the successor to GPGPU-sim, designed to model modern CUDA GPUs.

**How it works:**
1. Run the CUDA application on real hardware using the Accel-Sim tracer (NVBit-based)
2. Capture a hardware execution trace (warp instructions, memory accesses)
3. Replay the trace in the Accel-Sim timing model — no GPU hardware needed for simulation

**What it models:**
- SM pipeline (warp scheduling, instruction issue, execution units)
- L1/L2 cache hierarchy, MSHR, memory partition controllers
- DRAM timing (GDDR / HBM)
- Tensor core functional behavior (not full timing model for TCs)

**Accuracy:** validated against A100 and RTX 3090 hardware; typically within 10–15% on memory-bound workloads.

**Recent updates:** v1.3.0 released Feb 2025. MAccel-sim extension supports multi-GPU simulation of PyTorch / TensorFlow workloads end-to-end.

**Links:** https://accel-sim.github.io/ | [Georgia Tech / UBC]

**Reference:** Khairy et al. (2020). "Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling." *ISCA 2020.*

---

## MGPUSim {#mgpusim}

**What it is:** A cycle-accurate, multi-GPU simulator targeting AMD GCN3 architecture. Written in Go; highly modular.

**Key features:**
- Models AMD GCN3 ISA (Vega-era GPUs)
- Full memory hierarchy: cache hierarchy, HBM, Infinity Fabric interconnect
- Multi-GPU simulation with peer-to-peer transfers
- 5.5% average error vs. real AMD hardware

**Best for:** multi-GPU memory system studies, cache/interconnect research on AMD-like architectures.

**Limitation:** targets GCN3; does not model CDNA3 / MI300X features (tensor cores, HBM3).

**Links:** https://github.com/sarchlab/mgpusim | [Boston University]

**Reference:** Sun et al. (2019). "MGPUSim: Enabling Multi-GPU Performance Modeling and Optimization." *ISCA 2019.*

---

## SCALE-sim v3 {#scale-sim}

**What it is:** A modular, cycle-accurate simulator for systolic-array-based AI accelerators (NPU/TPU-like designs). v3 (2025) adds support for recent innovations including multi-level memory hierarchies and end-to-end system analysis.

**What it models:**
- Systolic array compute (weight-stationary, output-stationary, row-stationary dataflows)
- On-chip SRAM buffer hierarchy
- DMA / memory controller timing
- Full layer execution: tiling → buffer fill → systolic compute → writeback

**Integration:** SCALE-sim is used as the compute backend inside **ASTRA-sim**.

**v3 improvements over v2:**
- Modular architecture: pluggable memory, compute, and interconnect models
- Models structural hazards and DMA/compute overlap
- End-to-end system-level analysis beyond single-layer

**Links:** https://arxiv.org/abs/2504.15377 | [ARM / academia]

**Reference:** SCALE-sim v3 (2025). "A modular cycle-accurate systolic accelerator simulator for end-to-end system analysis." arXiv:2504.15377.

---

## Simulation Speed vs. Fidelity

| Simulator | Relative speed | Modeling target |
|-----------|---------------|-----------------|
| Roofline / analytical | ~1s | Throughput ceiling |
| MAESTRO / Timeloop | ~1 min | Dataflow + energy |
| ASTRA-sim (analytical backend) | ~10 min | Distributed training |
| Accel-Sim | hours–days | NVIDIA GPU microarch |
| MGPUSim | hours–days | AMD GPU microarch |
| SCALE-sim v3 | minutes–hours | Systolic NPU |

---

## Choosing Between Them

- **Studying NVIDIA GPU behavior:** Accel-Sim
- **Studying AMD GPU behavior / multi-GPU:** MGPUSim
- **Designing a custom NPU / TPU-like chip:** SCALE-sim + Timeloop
- **Distributed training scalability:** ASTRA-sim (uses one of the above as compute backend)

---

## See Also

- [ASTRA-sim](distributed.md) — wraps cycle-accurate simulators in a distributed system
- [Dataflow mappers](dataflow_mappers.md) — faster alternative for design space exploration
- [Characterization](../characterization/index.md) — profiling real hardware for validation
