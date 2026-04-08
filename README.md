# AI Performance Modeling & Simulation — Knowledge Base

Entry point for the AI performance modeling knowledge base. Topics span hardware characterization, workload analysis, analytical modeling, and simulation of modern AI accelerators.

---

## Sections

| Section | Description |
|---------|-------------|
| [Hardware](hardware/index.md) | Architecture and specs of modern AI accelerators (GPU, TPU, NPU, etc.) |
| [Workloads](workloads/index.md) | AI workload taxonomy: training, inference, operators, kernels |
| [Performance Modeling](modeling/index.md) | Analytical models, roofline, latency/throughput prediction |
| [Characterization](characterization/index.md) | Benchmarking methodology, microbenchmarks, profiling |
| [References](references/index.md) | Papers, tools, datasets, and external resources |

---

## Key Concepts at a Glance

- **Roofline model** — bounds performance by compute (FLOPS) or memory bandwidth; see [modeling/roofline.md](modeling/roofline.md)
- **Arithmetic intensity** — ratio of FLOPs to bytes moved; determines roofline operating point
- **Memory hierarchy** — HBM → L2 → SRAM → registers; critical for latency modeling
- **Tensor parallelism / pipeline parallelism** — affects how workloads map to multi-accelerator systems

---

## How to Navigate

Each section has an `index.md` that lists and links its sub-topics. Cross-references between sections use relative paths. All hardware specs are in `hardware/`, all model predictions in `modeling/`.

---

## Status

> This knowledge base is actively being built. Stubs are marked with `[stub]`.
