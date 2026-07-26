---
type: Index
title: AI Performance Modeling & Simulation — Knowledge Base
description: Root entry point spanning hardware, workloads, modeling, simulation, scheduling, and characterization.
tags: [overview, root, knowledge-base]
timestamp: 2026-05-30T23:45:33-07:00
---

# AI Performance Modeling & Simulation — Knowledge Base

### 🌐 Read it on the web: **<https://paveldournov.github.io/ai-perf-eng/>**

> The knowledge base is published as a live [docsify](https://docsify.js.org) site at
> **[paveldournov.github.io/ai-perf-eng](https://paveldournov.github.io/ai-perf-eng/)** —
> browse it there for rendered pages, search, and navigation.

Entry point for the AI performance modeling knowledge base. Topics span hardware characterization, workload analysis, analytical modeling, and simulation of modern AI accelerators.

---

## Sections

| Section | Description |
|---------|-------------|
| [Hardware](hardware/index.md) | Architecture and specs of modern AI accelerators (GPU, TPU, NPU, etc.) |
| [Workloads](workloads/index.md) | AI workload lifecycle (training → post-training → inference) and the shared operators/kernels underneath |
| [Performance Modeling](modeling/index.md) | Analytical models, roofline, latency/throughput prediction |
| [Simulation](simulation/index.md) | Simulator taxonomy, tools (ASTRA-sim, Timeloop, Accel-Sim, LLM-Viewer, …) |
| [Scheduling](scheduling/index.md) | Job admission, distributed runtimes, inference routing (Pathways, Ray, Kueue, llm-d) |
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

New here? Start with the **[Reading Guide](GUIDE.md)** — it gives structured learning paths with a recommended reading order and worked examples.

Each section has an `index.md` that lists and links its sub-topics. Cross-references between sections use relative paths. All hardware specs are in `hardware/`, all model predictions in `modeling/`.

---

## Status

> This knowledge base is actively being built. Stubs are marked with `[stub]`.
