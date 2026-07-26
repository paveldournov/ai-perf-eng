---
type: Hardware
title: Apple Neural Engine (ANE)
description: Apple's fixed-function fp16 matrix accelerator in A11–A18 / M1–M5 SoCs; an edge-NPU roofline contrast to datacenter GPUs and TPUs.
tags: [apple, ane, npu, edge, on-device, fp16, fixed-function, unified-memory]
resource: https://arxiv.org/abs/2606.22283
timestamp: 2026-07-03T00:00:00-07:00
---

# Apple Neural Engine (ANE)

← [Apple Index](index.md) | [Hardware Index](../index.md) | [Roofline params](../roofline_params.md)

The Apple Neural Engine is the **fixed-function matrix accelerator** built into every recent
Apple SoC (A11-class iPhone/iPad and M1-class Mac and later), normally reachable only through
the **Core ML** framework. It is the cleanest real-world example of a *fixed-function* AI
accelerator, and a useful foil to the programmable, HBM-backed, scale-out datacenter parts that
dominate the rest of this knowledge base.

> Specs and behavior below are drawn from Bryngelson (2026), a **reverse-engineered** account
> based on direct measurement (M1 and M5) plus static analysis of Apple's private runtime,
> compiler, driver, and firmware — not official Apple documentation. Figures are for the **M1**
> unless noted; the roofline has the same shape across the family, scaling with core count and clock.

---

## Key Specifications (M1, measured)

| Parameter | Value |
|-----------|-------|
| Type | Fixed-function fp16 matrix accelerator (NPU) |
| Compute datapath | fp16 in/out with a **wide, fp32-class accumulator** |
| Peak fp16 compute | ~4.8 TFLOP/s saturating (~12 TFLOP/s overhead-isolated slope); ~5.5 TFLOP/s convention from marketed ~11 int8 TOPS |
| int8 mode | Double-int8 lanes → ~2× fp16 rate |
| Memory | **Unified SoC DRAM, shared** with CPU/GPU (no dedicated HBM/VRAM) |
| DRAM bandwidth | ~85 GB/s (at memory controller); ~50 GB/s achieved weight-streaming |
| On-chip working set | **2 MB** (the primary design limit) |
| Dispatch floor | ~0.23 ms per dispatch; ~25 µs user-space bind overhead |
| Concurrency | **One firmware command in flight** — submissions serialize |
| Power envelope | ~1–2 W; ~0.37–0.5 pJ/FLOP (peak ~2.68 Tops/W) |
| Chip families covered | A11–A18, M1–M5 (28 compiler targets) |
| Programming access | Core ML (supported); undocumented direct path below Core ML |

---

## Roofline Parameters

| Precision | Peak FLOPS | DRAM BW | Ridge Point |
|-----------|-----------|---------|-------------|
| fp16 | ~12 TFLOP/s (roof) | 85 GB/s | ~141 FLOP/B |

- **Ridge point ~141 FLOP/byte** (using the 12 TFLOP/s compute roof over 85 GB/s). A layer above
  it is compute-bound; below it, DRAM-bandwidth-bound. Convolutions sit far to the right
  (a 3×3 conv at 256 channels reaches ~466 FLOP/B), so the engine rarely touches its BW ceiling
  on conv work; autoregressive **decode** sits far left and is bandwidth-bound.
- **The real limit is the 2 MB on-chip working set, not the ridge.** While a layer's activation
  fits in 2 MB its intermediates stay on-chip; exceed 2 MB and every layer streams its full
  activation through DRAM and arithmetic intensity collapses (a matmul at 12 TFLOP/s with a 1 MB
  activation drops to ~4.8 TFLOP/s once it crosses 2 MB).

See [Roofline model](../../modeling/roofline.md) and [Roofline parameters by chip](../roofline_params.md).

---

## What Makes It Different from GPUs and TPUs

| Property | Apple ANE | NVIDIA GPU | Google TPU |
|---|---|---|---|
| **Execution model** | Fixed-function **static "walked graph"** — no program counter, **no data-dependent control flow** (branches resolved on host) | Programmable SIMT (CUDA) | Programmable via XLA; systolic MXU |
| **Compute** | fp16 only + wide fp32-class accumulator; int8 double-lane 2× | Tensor Cores fp8/fp4/bf16/tf32/int8 | bf16 / int8 MXU |
| **Scale** | ~4.8 fp16 TFLOP/s | ~990 BF16 / ~2000 FP8 TFLOP/s | 918 BF16 TFLOP/s |
| **Memory** | Unified SoC DRAM, 85 GB/s, shared | 3.35 TB/s dedicated HBM3 | 1.64 TB/s HBM |
| **On-chip** | 2 MB working set (hard design limit) | 50 MB L2 | 128 MB VMEM |
| **Weight compression** | int4 / palettized **dequantized to fp16 before multiply** → saves *bytes*, not compute | Native low-precision *compute* | int8 compute |
| **Interconnect** | **None** — single engine, one command in flight | NVLink, multi-GPU pods | ICI torus, 256-chip pods |
| **Power** | ~1–2 W | 700 W | ~200 W class |
| **Access** | Core ML / undocumented direct path | CUDA / PTX | XLA / [Pallas](../../workloads/pallas_kernels.md) |

**One-liner:** the ANE trades all programmability and scale for energy efficiency on compute-bound
inference. Its scarce resource is a **2 MB on-chip buffer**, not FLOPs or interconnect.

---

## Execution & Numerics Notes

- **Static walked graph.** The compiled program is a static graph of work segments the controller
  advances through; there is no instruction stream to decode. Control flow must be static — a
  data-dependent branch must be resolved on the host or turned into a mask over both sides. Fixed
  trip-count loops unroll; data-dependent loops do not run on the engine.
- **Wide accumulator.** Products are fp16 but the running sum is fp32-class: a reduction of 16,384
  ones is bit-exact, where a naive fp16 running sum would stall near 2048. Precision is lost on
  **cancellation-heavy** steps (e.g. a transformer down-projection), from per-product fp16 input
  rounding rather than the accumulator.
- **Resident state across dispatches.** A KV-cache or optimizer state can be kept resident on the
  engine by aliasing one dispatch's output buffer to the next dispatch's input, so an
  autoregressive decoder appends in place instead of restreaming the cache through the host each
  step. (On the M1 this is done via output→input buffer aliasing; the native persistent-state
  data-movement engine is stubbed out.)

---

## Performance vs. the On-SoC GPU

- **Convolution / compute-bound matmul:** ~3.8× faster and ~9× more energy-efficient than the
  Apple GPU on a 256-channel 3×3 conv. Throughput-per-watt advantage of **~13–14.5×** on the
  convolution stack (2063 GFLOP/s/W on M1 vs 142 on the GPU; 2289 vs 175 on M5).
- **Where the GPU wins:** bandwidth-bound **autoregressive decode** (low arithmetic intensity),
  where raw DRAM bandwidth matters more than the engine's compute efficiency.
- **Compression on the direct path:** int4 lookup-table weights run ~2.37× faster than fp16
  (fewer bytes moved); structured sparsity 1.55–1.64× faster.

---

## References

- Bryngelson, S. H. (2026). "Apple Neural Engine: Architecture, Programming, and Performance." *arXiv:2606.22283.* https://arxiv.org/abs/2606.22283
- Apple — Core ML / Apple Neural Engine (marketed specs, e.g. ~11 int8 TOPS on M1).

---

## See Also

- [Roofline model](../../modeling/roofline.md) — the analytical lens this chip is a clean example of
- [Roofline parameters by chip](../roofline_params.md) — ANE row alongside GPU/TPU
- [Memory hierarchy](../memory_hierarchy.md) — unified-memory vs HBM contrast
- [Etched Sohu](../etched/sohu.md) — the datacenter mirror: also fixed-function, but trades programmability for throughput instead of edge energy efficiency
- [Hardware index](../index.md) — accelerator families
