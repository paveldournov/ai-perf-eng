---
type: Method
title: Dataflow Mapper Simulators
description: Model how a DNN layer is tiled and scheduled onto an accelerator's buffer hierarchy.
tags: [simulation, dataflow, mapping, tiling, timeloop]
timestamp: 2026-04-11T16:59:22-07:00
---

# Dataflow Mapper Simulators

← [Simulation Index](index.md)

Dataflow mappers model how a DNN layer is *scheduled* onto an accelerator — how loops are tiled, which data sits in which buffer level, and what the resulting latency and energy are. They sit between pure roofline models and full cycle-accurate simulators.

---

## Timeloop + Accelergy {#timeloop}

**What it is:** Timeloop is a systematic framework for DNN accelerator evaluation. It analytically models the data movement and compute latency for any mapping (tiling + loop order) of a DNN layer onto a specified hardware architecture. Accelergy is its companion energy estimator.

**Architecture:**
- **Mapper:** searches the mapping space (loop tiling × loop ordering × data staging) to find the optimal schedule
- **Model:** given a mapping, analytically computes latency, energy, and area
- **Accelergy:** plug-in energy tables for any PE/buffer/interconnect component

**Inputs:**
- Hardware description (PE count, buffer sizes, NoC bandwidth)
- Workload description (layer shapes: M, N, K, etc.)
- Mapper constraints (which dataflows to explore)

**Outputs:**
- Latency (cycles), energy (pJ), arithmetic intensity per buffer level
- Optimal mapping for a given optimization objective

**Accuracy:** matches RTL simulation closely for systolic designs.

**Limitations:**
- Models idealized systolic arrays, not GPU SM microarchitecture
- Does not model collective communication (see ASTRA-sim)
- Best for accelerator design space exploration, not GPU performance prediction

**Links:** https://timeloop.csail.mit.edu/ | [MIT / NVIDIA Research]

**Reference:** Parashar et al. (2019). "Timeloop: A Systematic Approach to DNN Accelerator Evaluation." *ISPASS 2019.*

---

## MAESTRO {#maestro}

**What it is:** MAESTRO (Modeling Accelerator Efficiency via Spatio-Temporal Reuse and Occupancy) is a data-centric analytical cost model for DNN dataflows. It characterizes reuse patterns, performance, and hardware cost from a dataflow description.

**Key idea:** Expresses dataflows using *directives* that describe how data is spatially and temporally mapped to PEs. From these, it derives:
- Working set sizes at each buffer level
- Reuse factors (spatial + temporal)
- Latency, energy, and NoC traffic

**Speed vs. accuracy:** 90–95% accuracy vs. RTL, 1000–4000× faster.

**Inputs:** DNN model layers, dataflow directives, hardware config (PE count, buffer, bandwidth).

**Outputs:** Latency, energy (compute + buffer + NoC), throughput, utilization — over 20 statistics.

**Links:** https://maestro.ece.gatech.edu/ | https://github.com/maestro-project/maestro | [Georgia Tech Synergy Lab]

**Reference:** Kwon et al. (2020). "MAESTRO: A Data-Centric Approach to Understand Reuse, Performance, and Hardware Cost of DNN Mappings." *IEEE Micro 2020.*

---

## Timeloop vs. MAESTRO Comparison

| | Timeloop | MAESTRO |
|-|----------|---------|
| Primary focus | Optimal mapping search | Dataflow cost analysis |
| HW description | Flexible hierarchy | Abstract PE + buffer |
| Energy model | Accelergy (component library) | Analytical (built-in) |
| Mapper | Yes (exhaustive / heuristic) | No (user specifies dataflow) |
| Best for | Accelerator DSE, optimal tiling | Fast dataflow tradeoff analysis |

---

## See Also

- [Cycle-accurate simulators](cycle_accurate.md) — for GPU microarch modeling
- [ASTRA-sim](distributed.md) — add collective communication on top
- [Roofline model](../modeling/roofline.md) — even simpler analytical bound
