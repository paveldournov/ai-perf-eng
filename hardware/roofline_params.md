---
type: Dataset
title: Roofline Parameters by Chip
description: Per-device peak FLOPS, peak memory bandwidth, and ridge point lookup table.
tags: [roofline, specs, flops, bandwidth, ridge-point]
timestamp: 2026-08-23T00:00:00-07:00
---

# Roofline Parameters by Chip

← [Hardware Index](index.md) | [Roofline model](../modeling/roofline.md)

Roofline ridge point = Peak FLOPS / Peak Memory Bandwidth.
Operations with arithmetic intensity **above** the ridge point are compute-bound; below it, memory-bandwidth-bound.

---

## Data Table

| Chip | Precision | Peak FLOPS (dense) | Peak HBM BW | Ridge Point |
|------|-----------|-------------------|-------------|-------------|
| H100 SXM5 | BF16 tensor | 989 TFLOPS | 3.35 TB/s | ~295 FLOP/B |
| H100 SXM5 | FP8 tensor | 1979 TFLOPS | 3.35 TB/s | ~591 FLOP/B |
| H200 SXM5 | BF16 tensor | 989 TFLOPS | 4.8 TB/s | ~206 FLOP/B |
| H200 SXM5 | FP8 tensor | 1979 TFLOPS | 4.8 TB/s | ~412 FLOP/B |
| B200 SXM | FP8 tensor | ~4,500 TFLOPS | 8.0 TB/s | ~563 FLOP/B |
| B200 SXM | FP4 tensor | ~9,000 TFLOPS | 8.0 TB/s | ~1,125 FLOP/B |
| B300 SXM | FP8 tensor | ~7,500 TFLOPS | 8.0 TB/s | ~938 FLOP/B |
| B300 SXM | FP4 tensor | ~15,000 TFLOPS | 8.0 TB/s | ~1,875 FLOP/B |
| RTX 4090 | BF16 tensor | 165 TFLOPS | 1.0 TB/s | ~165 FLOP/B |
| MI300X | BF16 tensor | 1307 TFLOPS | 5.3 TB/s | ~247 FLOP/B |
| MI355X | FP8 tensor | ~10,000 TFLOPS | 8.0 TB/s | ~1,250 FLOP/B |
| MI355X | FP4 tensor | ~20,000 TFLOPS | 8.0 TB/s | ~2,500 FLOP/B |
| TPUv5p | BF16 | 459 TFLOPS | 2.76 TB/s | ~166 FLOP/B |
| TPU v6e (Trillium) | BF16 | 918 TFLOPS | 1.64 TB/s | ~560 FLOP/B |
| TPU v6e (Trillium) | Int8 | 1,836 TOPs | 1.64 TB/s | ~1,120 FLOP/B |
| TPU v7x | FP8 | ~4,600 TFLOPS | 7.38 TB/s | ~623 FLOP/B |
| TPU v8t | FP4 | 12,600 TFLOPS | 6.53 TB/s | ~1,930 FLOP/B |
| TPU v8i | FP4 | 10,100 TFLOPS | 8.60 TB/s | ~1,174 FLOP/B |
| Gaudi3 | BF16 | 1835 TFLOPS | 3.7 TB/s | ~496 FLOP/B |
| Apple ANE (M1) | fp16 | ~12 TFLOPS (roof) | 0.085 TB/s | ~141 FLOP/B |
| Etched Sohu | FP8 | not disclosed | ~4.8 TB/s (est.) | — |
| Trainium2 | FP8 | 1,300 TFLOPS | 2.9 TB/s | ~448 FLOP/B |
| Trainium3 | FP8 | 2,500 TFLOPS | 4.9 TB/s | ~510 FLOP/B |
| Cerebras WSE-3 | FP16 | ~15,800 TFLOPS (dense, derived) | **21 PB/s (on-wafer SRAM aggregate)** | **~0.75 FLOP/B** |
| Groq LPU (GroqChip) | FP16 | 188 TFLOPS | **80 TB/s (on-chip SRAM aggregate)** | **~2.4 FLOP/B** |

> Numbers are for peak dense (non-sparse) unless noted. Sparse (2:4) doubles FLOPS on NVIDIA chips.
> RTX 4090 is a consumer card (GDDR6X, not HBM); listed as a low-ridge reference point — see the [roofline model](../modeling/roofline.md#intuition-a-gpu-has-two-speeds).
> Apple ANE is an edge NPU: the "roof" is the overhead-isolated slope (a single large matmul saturates near 4.8 fp16 TFLOP/s), and its real limit is a 2 MB on-chip working set, not the ridge point. See [ANE](apple/ane.md).
> Etched Sohu is a transformer-only inference ASIC that does not publish a peak-FLOPS figure, so no ridge point is given; the ~4.8 TB/s HBM bandwidth is an estimate. See [Sohu](etched/sohu.md).
>
> **Cerebras and Groq rows are not comparable to the HBM rows above.** Their bandwidth
> figure is an *aggregate of hundreds of thousands of local SRAM ports*, not a
> point-to-point link to a shared memory, and there is no HBM tier at all. Their
> ridge points land two to three orders of magnitude lower than every HBM part —
> which is the entire architectural claim: at ~1.3 bytes per dense FP16 FLOP (WSE-3)
> versus ~0.002 for a B200, these are the only machines whose compute and bandwidth
> are in balance, and the reason both are sold on batch-1 decode latency. WSE-3 dense
> FLOPs are derived (900,000 cores × 8-wide FMAC × 1.1 GHz); Cerebras publishes only a
> 125 PFLOPS *sparse* figure. See [AI chip architectures](architectures.md).

---

## Worked Example: LLM Decode on H100

A single attention head query during decode:
- FLOPs ≈ 2 × seq_len × d_model (matmul)
- Bytes moved ≈ KV cache size loaded from HBM

Typical arithmetic intensity for decode: **< 10 FLOP/B** → deeply memory-bandwidth-bound on all chips above.

---

## See Also

- [AI chip architectures](architectures.md) — what each vendor's numbers mean and why they are not apples-to-apples
- [Roofline model](../modeling/roofline.md)
- [H100 specs](nvidia/h100.md)
- [Apple ANE](apple/ane.md) — edge-NPU roofline contrast
