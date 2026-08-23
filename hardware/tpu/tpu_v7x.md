---
type: Hardware
title: Google TPU v7x
description: Seventh-generation-class TPU, probably Ironwood (v7); specs from third-party serving benchmarks (Ling-2.6-1T on SGLang-JAX) corroborated by a second independent source.
tags: [google, tpu, v7x, ironwood, v7, fp8, ici, sparsecore]
timestamp: 2026-08-23T00:00:00-07:00
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

> **Probable identification: this is Ironwood (TPU v7).** A second independent source
> ([Peake, *AI Chip Architectures*](../architectures.md), 2026) describes Ironwood/v7
> with figures that match every number below — 4.6 PFLOPS FP8, ~7.4 TB/s HBM, and 1.2
> TB/s bidirectional ICI per chip — alongside 192 GB of HBM3e. Two independent sources
> converging on the same three figures is strong evidence they describe one part; the
> `v7x` label remains unexplained (it may be a SKU suffix in the style of v5e/v5p).
> Treat the identification as probable, not confirmed.

---

## Key Specifications (as reported)

| Parameter | Value | Source |
|-----------|-------|--------|
| Peak FP8 compute | ~4.6 PFLOP/s per chip | LMSYS Ling-2.6 post |
| HBM bandwidth | 7.38 TB/s per chip | LMSYS Ling-2.6 post |
| ICI bandwidth | 1.2 TB/s bidirectional per chip | LMSYS Ling-2.6 post |
| Benchmark topology | 16 chips (32 devices), 2×2×4 ICI torus | LMSYS Ling-2.6 post |
| HBM capacity | 192 GB HBM3e | Peake (as Ironwood/v7) |
| Numerics | native FP8 (E4M3 + E5M2), ~2× BF16 throughput | Peake (as Ironwood/v7) |
| SparseCore | 4 per chip (2 per chiplet, dual-die layout) | Peake (as Ironwood/v7) |
| ICI topology | 3D torus (6 ICI ports, flagship configuration) | Peake (as Ironwood/v7) |
| Superpod | 9,216 chips = 144 cubes of 64; 1.77 PB HBM (~68 PB/s); 42.5 ExaFLOPS FP8 | Peake (as Ironwood/v7) |
| Scale-out (DCN) | ~100 Gbps per chip | Peake (as Ironwood/v7) |

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

## Positioning

Ironwood is the first TPU built for **inference of reasoning models**, and the
generation that introduced native FP8. Its 9,216-chip superpod is the scale-up unit
equivalent in role to NVIDIA's 72-GPU NVL72 — two orders of magnitude more chips, at
much lower per-chip bandwidth, tied together by message-passing ICI and optical circuit
switches rather than a coherent crossbar. See
[AI chip architectures](../architectures.md#google-tpu--the-compiler-is-the-system).

---

## See Also

- [AI chip architectures](../architectures.md) — cross-vendor comparison; source of the Ironwood identification
- [TPU v6e (Trillium)](tpu_v6e.md) — documented predecessor generation
- [TPU v8t / v8i](tpu_v8.md) — documented successor generation
- [TPU family overview](index.md)
- [MoE efficiency — Fused MoE V2 case study](../../workloads/moe.md#serving-case-study-fused-moe-v2-on-tpu-ling-26-1t)
- [References — TPU papers & docs](../../references/index.md)
