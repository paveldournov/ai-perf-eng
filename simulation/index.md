---
type: Index
title: Performance Simulation for AI Workloads
description: The simulation spectrum from fast analytical bounds to slow cycle-accurate models.
tags: [simulation, fidelity, taxonomy]
timestamp: 2026-05-31T23:00:31-07:00
---

# Performance Simulation for AI Workloads

← [Back to README](../README.md)

Performance simulation spans a wide spectrum — from fast analytical bounds to slow cycle-accurate RTL-level models. Choosing the right tool depends on what question you are answering and at what fidelity.

---

## Taxonomy

```
Fast / low fidelity                              Slow / high fidelity
─────────────────────────────────────────────────────────────────────────►
Analytical     Analytical      Trace-driven      Cycle-accurate    RTL /
roofline       mapper          simulator         full simulator    FPGA
(Roofline,     (Timeloop,      (Accelsim,        (MGPUSim,         emulation
 LLM math)      MAESTRO)        ASTRA-sim)        SCALE-sim v3)
```

| Category | Speed | Fidelity | Best for |
|----------|-------|----------|----------|
| [Analytical / roofline](analytical.md) | Seconds | Low–medium | Quick bottleneck ID, design space sweep |
| [Dataflow mappers](dataflow_mappers.md) | Minutes | Medium | Tiling/dataflow optimization, energy-efficiency |
| [Distributed system simulators](distributed.md) | Minutes–hours | Medium–high | Collective comm, parallelism strategy, multi-GPU |
| [Cycle-accurate GPU simulators](cycle_accurate.md) | Hours–days | High | Microarch changes, cache/memory system study |
| [LLM-specific analysis tools](llm_tools.md) | Seconds | Medium | Per-layer roofline, batch/seq sweep for LLMs |
| [Fault tolerance](fault_tolerance.md) | — | Production | Effective MFU, MTTF cost models, recovery strategies |

---

## Key Simulators at a Glance

| Tool | Category | HW Target | Maintained by |
|------|----------|-----------|---------------|
| [Timeloop + Accelergy](dataflow_mappers.md#timeloop) | Dataflow mapper | Systolic / custom ASIC | MIT / NVIDIA |
| [MAESTRO](dataflow_mappers.md#maestro) | Dataflow mapper | Systolic / custom ASIC | Georgia Tech |
| [ASTRA-sim 2.0](distributed.md#astra-sim) | Distributed simulator | GPU clusters | Georgia Tech / Meta / AMD / NVIDIA |
| [Chakra](distributed.md#chakra) | Workload trace format | Any | MLCommons |
| [Accel-Sim](cycle_accurate.md#accel-sim) | Cycle-accurate trace-driven | NVIDIA GPU | Georgia Tech / UBC |
| [MGPUSim](cycle_accurate.md#mgpusim) | Cycle-accurate | AMD GCN GPU | Boston U. |
| [SCALE-sim v3](cycle_accurate.md#scale-sim) | Cycle-accurate systolic | Systolic array NPU | ARM / academia |
| [LLM-Viewer](llm_tools.md#llm-viewer) | LLM analysis | Any (parameterized) | Community |
| [LLMRoofline](llm_tools.md#llmroofline) | Roofline for LLMs | Any | Community |

---

## Simulation vs. Analytical Modeling

See [modeling/index.md](../modeling/index.md) for pure analytical models (no simulator needed). Simulators add fidelity by modeling microarchitectural effects that analytical models ignore:
- Structural hazards, pipeline stalls
- Cache miss penalties, bank conflicts
- Collective communication contention
- Kernel launch overhead

---

## Workflow Recommendation

1. **Roofline first** — classify ops as compute or BW bound ([modeling/roofline.md](../modeling/roofline.md))
2. **LLM-Viewer** — sweep batch size / seq length per layer for LLM workloads
3. **Timeloop / MAESTRO** — optimize tiling and dataflow for a specific op on a target HW
4. **ASTRA-sim** — model distributed training across multi-GPU topology
5. **Accel-Sim / MGPUSim** — only when microarch-level fidelity is required

---

## See Also

- [Modeling](../modeling/index.md)
- [Characterization](../characterization/index.md)
- [References — simulation papers](../references/index.md#simulation--distributed-systems) — distributed, dataflow-mapper, cycle-accurate GPU, and LLM-inference-analysis subsections
