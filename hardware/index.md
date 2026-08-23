---
type: Index
title: Hardware — AI Accelerator Overview
description: Architecture and specs of modern AI accelerators (GPU, TPU, NPU) and their common dimensions.
tags: [hardware, accelerators, gpu, tpu]
timestamp: 2026-08-23T00:00:00-07:00
---

# Hardware — AI Accelerator Overview

← [Back to README](../README.md)

Modern AI accelerators share a common architectural pattern: a sea of parallel compute units backed by high-bandwidth memory, connected by high-speed interconnects. The differences lie in the specifics.

---

## Accelerator Families

| Family | Representative Chips | Core bet |
|--------|---------------------|----------|
| [NVIDIA GPU](nvidia/index.md) | H100, H200, B200, B300, GB200/GB300 NVL72 | Programmability; hardware-scheduled SIMT with an ever-larger async Tensor Core |
| [AMD GPU](architectures.md#amd-instinct--the-bet-lives-between-the-cus) | MI300X, MI325X, MI355X, MI455X (Helios) | Conservative CU, aggressive packaging: 3D-stacked chiplets, HBM capacity, 256 MB Infinity Cache |
| [Google TPU](tpu/index.md) | TPUv4, v5p, v6e (Trillium), v7 (Ironwood), v8t/v8i | Systolic array + software scratchpads; **the compiler is the scheduler** |
| [AWS Trainium](architectures.md#aws-trainium--the-tpu-thesis-inside-a-different-cloud) | Trainium2, Trainium3 | The TPU thesis inside AWS, plus collectives in dedicated silicon |
| [Cerebras](architectures.md#cerebras-wse--the-memory-wall-as-a-packaging-choice) | WSE-2, WSE-3 | Don't cut the wafer; SRAM is the only memory |
| [Groq](architectures.md#groq-lpu--determinism) | GroqChip / LPU | Delete every reactive component; compile every cycle |
| [Etched](etched/index.md) | [Sohu](etched/sohu.md) | Hardwire the transformer graph into silicon |
| [Apple Silicon](apple/index.md) | [Neural Engine (ANE)](apple/ane.md) | Fixed-function fp16 NPU for on-device inference |

**Start here:** [AI chip architectures](architectures.md) reads all six datacenter
families through one frame — where data lives, how it moves, what the compute units look
like, and how chips talk at scale — with per-chip and per-rack comparison tables.

---

## Common Architectural Dimensions

- Compute units — SM/CU structure, tensor cores, systolic arrays
- [Memory hierarchy](memory_hierarchy.md) — HBM capacity/BW, on-chip SRAM, L2 cache
- Interconnect — NVLink, Infinity Fabric, ICI; topology and BW (see [Collective Ops](../workloads/collective_ops.md) for BW numbers)
- Numerical formats — FP32, BF16, FP8, INT8, INT4 and their throughput multipliers (see [GEMM](../workloads/gemm.md#precision-modes-and-hardware-paths))

---

## Cross-Cutting Analysis

- [AI chip architectures](architectures.md) — comparative survey: philosophy, architecture, scale-up/scale-out, and software stack per family; the recurring patterns (off-core engines, the migrating matmul instruction, precision halving, topology following the collective, copper setting the rack boundary)

---

## Hardware Characterization Data

- [Roofline parameters by chip](roofline_params.md) — peak FLOPS, peak BW, ridge point per device
- Spec comparison table — side-by-side of key chips (see [Roofline parameters](roofline_params.md) for current numbers)

---

## Notable Existing Artifacts

- [`h100_hw_model.svg`](../h100_hw_model.svg) — H100 hardware model diagram
