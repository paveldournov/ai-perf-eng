---
type: Method
title: Distributed System Simulators
description: Model compute, memory, and collective communication across multi-GPU/multi-node systems.
tags: [simulation, distributed, collectives, astra-sim, scaling]
timestamp: 2026-05-30T23:45:33-07:00
---

# Distributed System Simulators

← [Simulation Index](index.md)

Distributed simulators model the interplay of compute, memory, and collective communication across multi-GPU / multi-node training systems. They answer questions that single-chip models cannot: how does all-reduce latency affect MFU? Where does pipeline parallelism introduce bubbles? How does NVLink topology affect scaling efficiency?

---

## ASTRA-sim 2.0 {#astra-sim}

**What it is:** A modular, plug-and-play distributed training simulator. It co-simulates the *compute*, *memory*, and *network* layers of a distributed ML system.

**Architecture — three pluggable layers:**

```
┌─────────────────────────────────────────────────┐
│   Workload layer  (training loop, parallelism)   │
├──────────────────┬──────────────────────────────┤
│  Compute backend │  Network backend              │
│  (Roofline /     │  (Analytical / ns-3 /         │
│   SCALE-sim)     │   Garnet)                     │
├──────────────────┴──────────────────────────────┤
│  Memory backend  (Analytical)                    │
└─────────────────────────────────────────────────┘
```

**What it simulates:**
- Data parallel, tensor parallel, pipeline parallel, and hybrid parallelism
- Hierarchical network topologies (NVLink intra-node, InfiniBand / Ethernet inter-node)
- Collective operations: all-reduce, all-gather, reduce-scatter, all-to-all
- Disaggregated memory systems (ASTRA-sim 2.0 addition)

**Key use cases:**
- Compare topology choices (fat-tree vs. torus vs. NVLink rail)
- Predict scaling efficiency at 1K+ GPU scale without running actual hardware
- SW/HW co-design: evaluate new collective algorithms or parallelism strategies

**Input:** workload description (ops + dependencies), hardware config (compute + network), parallelism strategy.

**Output:** training step time, utilization breakdown (compute vs. communication), bottleneck identification.

**Maintained by:** Georgia Tech, Meta, AMD, NVIDIA.

**Links:** https://astra-sim.github.io/ | https://github.com/astra-sim/astra-sim

**References:**
- Rashidi et al. (2020). "ASTRA-sim: Enabling SW/HW Co-Design Exploration for Distributed DL Training Platforms." *ISPASS 2020.*
- Won et al. (2023). "ASTRA-sim 2.0: Modeling Hierarchical Networks and Disaggregated Systems." *ISPASS 2023.*

---

## Chakra — Standardized Execution Traces {#chakra}

**What it is:** A community standard (MLCommons) for representing distributed ML workloads as execution traces — enabling any simulator or replay tool to consume the same workload description.

**Format:** Directed acyclic graph (DAG) where:
- **Nodes** = ML operations (compute, memory, communication)
- **Edges** = data and control dependencies
- Nodes carry: op type, timing, resource constraints

**Why it matters:** Without a standard trace format, every simulator speaks a different workload language. Chakra decouples *workload capture* from *simulation*, enabling:
- Sharing traces across research groups without exposing model weights
- Using the same trace in ASTRA-sim, proprietary simulators, or replay tools
- Synthesizing traces via ML models when real traces are unavailable

**Ecosystem tools:**
- Trace collection from PyTorch / TF-XLA
- Trace synthesis (ML-based, for obfuscation / projection)
- Trace replay for network performance testing

**Maintained by:** MLCommons (joint: Georgia Tech, Meta, and others).

**Links:** https://mlcommons.org/working-groups/research/chakra/ | https://github.com/mlcommons/chakra

**Reference:** Won et al. (2023). "Chakra: Advancing Performance Benchmarking and Co-design using Standardized Execution Traces." arXiv:2305.14516.

---

## Heterogeneity-Aware Distributed LLM Simulator

Recent work (2024–2025) extends distributed simulation to *heterogeneous* clusters — mixed GPU generations, shared cloud resources with variable interconnect quality.

- Models device groups with per-device compute and intra/inter-chip BW
- Predicts training time under heterogeneous parallelism assignments
- Relevant for cloud training where hardware homogeneity cannot be assumed

**Reference:** arXiv:2508.05370 — "Simulating LLM Training Workloads for Heterogeneous Compute and Network Infrastructure."

---

## See Also

- [ASTRA-sim tutorials at MICRO 2024](https://astra-sim.github.io/tutorials/micro-2024)
- [Cycle-accurate simulators](cycle_accurate.md) — use as compute backend inside ASTRA-sim
- [Collective operations](../workloads/collective_ops.md) — all-reduce / all-gather latency formulas
- [Collective ops BW numbers](../workloads/collective_ops.md#bandwidth-numbers-by-topology) — NVLink, InfiniBand, ICI reference values
