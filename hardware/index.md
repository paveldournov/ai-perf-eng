# Hardware — AI Accelerator Overview

← [Back to README](../README.md)

Modern AI accelerators share a common architectural pattern: a sea of parallel compute units backed by high-bandwidth memory, connected by high-speed interconnects. The differences lie in the specifics.

---

## Accelerator Families

| Family | Representative Chips | Primary Use |
|--------|---------------------|-------------|
| [NVIDIA GPU](nvidia/index.md) | H100, H200, B200, GB200 | Training + inference |
| [AMD GPU](amd/index.md) | MI300X, MI325X | Training + inference |
| [Google TPU](tpu/index.md) | TPUv4, TPUv5p, TPUv6 | Training (LLM at scale) |
| [Custom NPU / ASIC](npu/index.md) | Trainium2, Gaudi3, Groq LPU | Inference-optimized |
| [Cerebras / Wafer-Scale](wafer_scale.md) | WSE-3 | Extreme memory-BW workloads |

---

## Common Architectural Dimensions

- [Compute units](compute_units.md) — SM/CU structure, tensor cores, systolic arrays
- [Memory hierarchy](memory_hierarchy.md) — HBM capacity/BW, on-chip SRAM, L2 cache
- [Interconnect](interconnect.md) — NVLink, Infinity Fabric, ICI; topology and BW
- [Numerical formats](numerical_formats.md) — FP32, BF16, FP8, INT8, INT4 and their throughput multipliers

---

## Hardware Characterization Data

- [Roofline parameters by chip](roofline_params.md) — peak FLOPS, peak BW, ridge point per device
- [Spec comparison table](spec_table.md) — side-by-side of key chips

---

## Notable Existing Artifacts

- [`h100_hw_model.svg`](../h100_hw_model.svg) — H100 hardware model diagram
