---
type: Index
title: Hardware — AI Accelerator Overview
description: Architecture and specs of modern AI accelerators (GPU, TPU, NPU) and their common dimensions.
tags: [hardware, accelerators, gpu, tpu]
timestamp: 2026-05-30T23:45:33-07:00
---

# Hardware — AI Accelerator Overview

← [Back to README](../README.md)

Modern AI accelerators share a common architectural pattern: a sea of parallel compute units backed by high-bandwidth memory, connected by high-speed interconnects. The differences lie in the specifics.

---

## Accelerator Families

| Family | Representative Chips | Primary Use |
|--------|---------------------|-------------|
| [NVIDIA GPU](nvidia/index.md) | H100, H200, B200, GB200 | Training + inference |
| AMD GPU | MI300X, MI325X | Training + inference |
| [Google TPU](tpu/index.md) | TPUv4, TPUv5p, TPUv6e (Trillium), TPUv8t/v8i | Training + inference (LLM at scale) |
| Custom NPU / ASIC | Trainium2, Gaudi3, Groq LPU | Inference-optimized |
| [Apple Silicon](apple/index.md) | [Neural Engine (ANE)](apple/ane.md) | On-device / edge inference |
| Cerebras / Wafer-Scale | WSE-3 | Extreme memory-BW workloads |

---

## Common Architectural Dimensions

- Compute units — SM/CU structure, tensor cores, systolic arrays
- [Memory hierarchy](memory_hierarchy.md) — HBM capacity/BW, on-chip SRAM, L2 cache
- Interconnect — NVLink, Infinity Fabric, ICI; topology and BW (see [Collective Ops](../workloads/collective_ops.md) for BW numbers)
- Numerical formats — FP32, BF16, FP8, INT8, INT4 and their throughput multipliers (see [GEMM](../workloads/gemm.md#precision-modes-and-hardware-paths))

---

## Hardware Characterization Data

- [Roofline parameters by chip](roofline_params.md) — peak FLOPS, peak BW, ridge point per device
- Spec comparison table — side-by-side of key chips (see [Roofline parameters](roofline_params.md) for current numbers)

---

## Notable Existing Artifacts

- [`h100_hw_model.svg`](../h100_hw_model.svg) — H100 hardware model diagram
