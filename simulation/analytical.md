# Analytical Performance Models

← [Simulation Index](index.md)

Analytical models derive performance bounds from hardware specs and workload math — no simulation loop required. They are the fastest and most portable tools in the toolbox.

---

## Roofline Model

The foundational bound. Classifies any operation as compute- or memory-bandwidth-bound given arithmetic intensity.

- Full writeup: [modeling/roofline.md](../modeling/roofline.md)
- Per-chip ridge points: [hardware/roofline_params.md](../hardware/roofline_params.md)

---

## LLM Inference Arithmetic

First-principles latency model for prefill and decode steps:

- Full writeup: [modeling/llm_inference.md](../modeling/llm_inference.md)
- Covers: token latency, KV cache size, batch size crossover point, tensor parallel scaling

**Key insight:** decode at batch=1 achieves <1% of peak FLOPS on H100 BF16 — it is entirely HBM-BW-bound.

---

## Roofline-Driven ML Method (IBM, NeurIPS 2024)

Combines an analytical roofline model with regression models trained on historical profiling data. The hybrid approach calibrates the theoretical ceiling with empirically measured runtime overhead.

- Achieved 17-point R² improvement for OPT on vLLM, 87% MSE reduction
- Useful when you have some historical profiling data and want to generalize to new configs

**Reference:** Imai (2024). "Predicting LLM Inference Latency: A Roofline-Driven ML Method." *NeurIPS 2024 MLforSystems Workshop.*

---

## Forecasting LLM Inference via Hardware-Agnostic Modeling

Extends beyond the simple roofline to account for compute efficiency, memory efficiency, and their interaction — demonstrating that a two-factor model explains more variance than roofline alone.

**Reference:** arXiv:2508.00904

---

## LLM Inference Unveiled — Survey + Roofline Framework (2024)

Systematic survey of LLM inference optimizations analyzed through the roofline lens. Covers quantization, distillation, MoE, and system-level optimizations.

- Companion tool: **LLM-Viewer** (see [llm_tools.md](llm_tools.md))
- Explains *why* LLMs are memory-bound and quantifies the impact of each optimization

**Reference:** Yuan et al. (2024). "LLM Inference Unveiled: Survey and Roofline Model Insights." arXiv:2402.16363.

---

## Limitations of Analytical Models

- Ignore structural hazards, pipeline stalls, partial wavefront occupancy
- Assume perfect memory access (no bank conflicts, no TLB misses)
- Cannot model kernel launch latency or synchronization overhead
- Irregular ops (sparse attention, MoE routing) require careful FLOPs accounting

For cases where these matter: use [cycle-accurate simulators](cycle_accurate.md).
