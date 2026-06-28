---
type: Hardware
title: Google TPU v7x
description: Seventh-generation-class TPU; specs sourced from third-party serving benchmarks (Ling-2.6-1T on SGLang-JAX). [stub]
tags: [google, tpu, v7x, fp8, ici]
timestamp: 2026-06-27T00:00:00-07:00
---

# Google TPU v7x

← [TPU Index](index.md) | [Roofline params](../roofline_params.md)

> **[stub]** — Google has not published a full architecture whitepaper for this part.
> The figures below are **as reported by a third-party serving benchmark** (LMSYS,
> *Optimizing Ling-2.6-1T on TPU with SGLang-JAX*, June 2026), not official specs.
> Treat them as indicative and verify against Google Cloud docs when available.

TPU v7x is the generation sitting between the publicly-documented [v6e (Trillium)](tpu_v6e.md)
and [v8t/v8i](tpu_v8.md) in this knowledge base. It first appears here as the hardware
target of an external MoE-serving deep-dive rather than from a primary Google source.

---

## Key Specifications (as reported)

| Parameter | Value | Source |
|-----------|-------|--------|
| Peak FP8 compute | ~4.6 PFLOP/s per chip | LMSYS Ling-2.6 post |
| HBM bandwidth | 7.38 TB/s per chip | LMSYS Ling-2.6 post |
| ICI bandwidth | 1.2 TB/s bidirectional per chip | LMSYS Ling-2.6 post |
| Benchmark topology | 16 chips (32 devices), 2×2×4 ICI torus | LMSYS Ling-2.6 post |

---

## Roofline (FP8)

| Precision | Peak FLOPS | HBM BW | Ridge Point |
|-----------|-----------|--------|-------------|
| FP8 | ~4,600 TFLOPS | 7.38 TB/s | ~623 FLOP/B |

The high HBM bandwidth (≈4.5× v6e) materially shifts the decode operating point:
LLM decode remains memory-bandwidth-bound, but per-chip token throughput scales with
the bandwidth bump. See [Roofline parameters](../roofline_params.md).

---

## Reported Serving Result

In the LMSYS benchmark, a **TPU v7x-16** slice serving the 1T-parameter
[Ling-2.6 MoE](../../workloads/moe.md#serving-case-study-fused-moe-v2-on-tpu-ling-26-1t)
delivered **1.29×–1.77× decode output throughput vs. an H200×16** node (batch-size
dependent), using a fused MoE kernel that overlaps all-to-all communication with routed compute.

---

## See Also

- [TPU v6e (Trillium)](tpu_v6e.md) — documented predecessor generation
- [TPU v8t / v8i](tpu_v8.md) — documented successor generation
- [TPU family overview](index.md)
- [MoE efficiency — Fused MoE V2 case study](../../workloads/moe.md#serving-case-study-fused-moe-v2-on-tpu-ling-26-1t)
- [References — TPU papers & docs](../../references/index.md)
