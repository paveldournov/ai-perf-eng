---
type: Index
title: Performance Modeling
description: Analytical models that predict throughput, latency, and utilization from first principles.
tags: [modeling, analytical, roofline, latency, throughput]
timestamp: 2026-05-30T23:45:33-07:00
---

# Performance Modeling

← [Back to README](../README.md)

Analytical performance models predict throughput, latency, and utilization from first principles — without requiring a full simulation or hardware access.

---

## Models

| Model | What it predicts | Inputs |
|-------|-----------------|--------|
| [Roofline model](roofline.md) | Compute vs. BW bound, peak achievable FLOPS | AI, peak FLOPS, peak BW |
| [LLM inference model](llm_inference.md) | Prefill/decode latency, throughput, MFU | Model config, hardware specs, batch size |
| Training throughput model | Step time, MFU, scaling efficiency | Model size, parallelism config, hardware |
| [Memory capacity model](memory_capacity.md) | Whether model fits; weights + KV cache + activations | Model config, dtype, batch, seq length |
| [Parallelism strategies](parallelism.md) | Memory per GPU, communication overhead per strategy | TP/PP/DP degrees, model config, topology |
| [Inference routing](inference_routing.md) | KV-aware routing, disaggregated prefill/decode, session affinity | Serving topology, cache state, hardware specialization |
| [Speculative decoding](speculative_decoding.md) | Decode speedup from draft-and-verify; acceptance rate, tree drafting | Drafter type, acceptance rate, draft length, batch size |

---

## Foundational Concepts

- [Arithmetic intensity](../workloads/gemm.md) — FLOP/byte ratio, determines roofline region
- [Achieved vs. peak FLOPS (MFU)](mfu.md) — model FLOP utilization metric
- [Parallelism strategies](parallelism.md) — tensor, pipeline, data, sequence; effect on memory and communication

---

## How to Use These Models

1. Start with [roofline](roofline.md) to classify each operation as compute- or BW-bound
2. Use [LLM inference model](llm_inference.md) for end-to-end latency estimates
3. Cross-check against [hardware roofline params](../hardware/roofline_params.md)
4. Validate with profiler data — see [characterization](../characterization/index.md)

---

## Simulation Frameworks

- [Simulation overview](../simulation/index.md) — analytical simulators vs. cycle-accurate vs. ML-based
