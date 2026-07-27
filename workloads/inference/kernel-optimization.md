---
type: Concept
title: Kernel Optimization for LLM Inference
description: The GPU-code layer beneath the serving stack — execution model, memory hierarchy, tensor cores, occupancy, FlashAttention (v1–v4), and the kernel tooling ladder (cuBLAS/cuDNN, TVM/XLA, Triton, custom, Mojo/MAX).
tags: [kernel-optimization, gpu-architecture, flashattention, tensor-cores, triton, cuda, mojo, max]
resource: https://handbook.modular.com/kernel-optimization/
timestamp: 2026-07-26T00:00:00-07:00
---

# Kernel Optimization for LLM Inference

← [Inference Workloads Index](index.md)

Digest of the handbook's [Kernel optimization](https://handbook.modular.com/kernel-optimization/)
chapter — the GPU code that does the actual math, one level below
[serving-layer optimization](optimization.md). Kernels set the performance
ceiling everything above them lives under: system optimization is *how
efficiently requests flow*; kernel optimization is *how fast a single operation
runs*. See also the KB's [GPU kernels](../gpu_kernels.md) and
[Pallas kernels](../pallas_kernels.md).

---

## GPU execution model

GPUs keep thousands of lightweight threads in flight to hide latency.

- **Thread** — smallest unit; SIMT (Single Instruction, Multiple Threads). Each
  derives its data index from `blockIdx * blockDim + threadIdx`.
- **Warp** — 32 threads executing in lockstep; the actual scheduling unit. Branch
  divergence within a warp serializes both paths.
- **Thread block** — up to 1024 threads (32 warps) that cooperate via shared
  memory and barriers; scheduled on one SM.
- **SM (Streaming Multiprocessor)** — the core compute unit: CUDA cores, tensor
  cores, warp scheduler, register file, shared memory/L1. H100 SXM has 132 SMs,
  A100 has 108.
- **Occupancy** — active warps ÷ max warps per SM. Higher occupancy hides memory
  latency, but it is **not** a metric to maximize blindly: a kernel using more
  registers/shared memory has lower occupancy yet can run faster.
  FlashAttention deliberately trades occupancy for less HBM traffic. Profile
  first.

## Memory hierarchy

Most kernel bottlenecks are memory, not compute. Every optimization (tiling,
fusion, layout) moves data *up* this pyramid (H100 figures):

| Level | Size | Bandwidth | Latency | Managed by |
|-------|------|-----------|---------|------------|
| Registers | 256 KB/SM | highest | ~1 cycle | compiler |
| Shared memory / L1 | up to 228 KB/SM | ~20 TB/s | ~20–30 cyc | programmer (SMEM) / hardware (L1) |
| L2 cache | 50 MB | ~12 TB/s | ~200 cyc | hardware |
| HBM | 80 GB | 3.35 TB/s | ~400+ cyc | programmer |

- **Registers** — fastest, private per thread, limited (drives the occupancy
  tradeoff).
- **Shared memory** — programmer-managed on-chip scratchpad shared within a
  block; the main tool to cut HBM traffic (load once, reuse). Watch **bank
  conflicts** (typically 32 banks).
- **L1 / L2** — hardware-managed caches; L2 is shared across SMs. In LLM
  inference, large working sets often blow past L2, so **HBM bandwidth becomes
  the binding constraint** — the root cause of the memory-bound decode phase.

The KB's [memory hierarchy](../../hardware/memory_hierarchy.md) and
[H100](../../hardware/nvidia/h100.md) pages give the hardware detail.

## Tensor cores

Specialized matrix-multiply-accumulate units (since Volta, 2017) operating on
small tiles, delivering far higher throughput than CUDA cores for the GEMMs that
dominate transformers. They support mixed precision (FP16/BF16/FP8/INT8) — which
both raises throughput and cuts memory traffic — but require correct data
layout/dtype to engage, which is why libraries like cuBLAS and DSLs like Triton
exist. See [GEMM](../gemm.md).

## FlashAttention

Standard attention is **memory-bound**: it materializes the N×N score matrix in
HBM and reads it back (a 4K-token matrix is ~16M elements; 16K tokens need 256×
the memory of 1K). FlashAttention never materializes it — using **tiling +
recomputation** (compute each tile entirely in SRAM) and **kernel fusion**
(matmul → softmax → matmul in one kernel, no HBM round-trips).

| Version | Year | Key change | Result |
|---------|------|-----------|--------|
| FA-1 | 2022 | IO-aware tiled, fused, exact attention | 2–4× faster, up to 10× less memory |
| FA-2 | 2023 | Better warp parallelism / work partitioning | ~2× over FA-1 |
| FA-3 | 2024 | Hopper async + FP8/BF16 tensor cores | up to 2× over FA-2; ~740 TFLOPS on H100 |
| FA-4 | 2026 | Fully async MMA, larger tiles, SW-emulated exp, 2-CTA MMA; CuTeDSL | up to ~1613 TFLOPS on B200; tuned for Blackwell |

FA is a *kernel* optimization frameworks *adopted* — the canonical example of the
kernel layer setting the ceiling. The KB covers the algorithm from the workload
side in [attention](../attention.md); the references list the FA
papers.

## Approaches & the tooling ladder

Two broad approaches: **hand-written kernels** (CUDA, or the friendlier Triton
DSL) for maximum control, and **compiler-driven** (TVM, XLA) that generate/fuse
kernels from a high-level graph. Moving *down* the ladder buys control but costs
GPU expertise and portability.

| Layer | Examples | Tradeoff |
|-------|----------|----------|
| Vendor libraries | cuBLAS, cuBLASLt, cuDNN | Fastest start for standard ops; NVIDIA-locked, lags new patterns |
| AI compilers | Apache TVM, XLA/OpenXLA | Auto fusion/layout across the graph; hard to hit peak, abstraction limits control |
| Kernel DSLs | Triton | Tile-oriented Python, no manual warp/SMEM mgmt; still needs GPU intuition |
| Custom kernels | hand-written CUDA | Full control (where FlashAttention comes from); high cost, hard to maintain, CUDA locks to NVIDIA |
| Full-stack | **Mojo + MAX** | One vertically-integrated language/compiler/runtime aiming for kernel-level control *and* cross-vendor portability (NVIDIA/AMD/…) via MLIR |

**Kernel fusion** — combining ops into one kernel so intermediates stay in
on-chip memory instead of round-tripping HBM — is one of the highest-impact
optimizations. **[Mojo and MAX](https://www.modular.com/open-source/max)** are
Modular's attempt to give the same kernel code hardware-portable performance
(the MLIR compiler lowers warp-sync/tensor-core/memory primitives per target).

**Choosing:** most teams should *not* start at the bottom — begin with vendor
libraries via your framework, reach for a compiler when graph-level fusion helps,
Triton when you need a custom kernel with a friendlier model, Mojo/MAX for
control-plus-portability, and hand-written CUDA only for a critical bottleneck on
one target. Custom kernels also matter because new model architectures need new
kernels before frameworks support them efficiently.

---

## See Also

- [Attention](../attention.md) — the algorithm FlashAttention accelerates
- [GEMM](../gemm.md) — the matmuls tensor cores target
- [GPU kernels](../gpu_kernels.md) · [Pallas kernels](../pallas_kernels.md) · [CUDA/PTX](../cuda_ptx.md)
- [Memory hierarchy](../../hardware/memory_hierarchy.md) · [H100](../../hardware/nvidia/h100.md)
- [Inference optimization](optimization.md) — the serving layer above the kernels

---

*Adapted & summarized from Modular's [LLM Inference Handbook](https://handbook.modular.com/) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — changes were made. See [Attribution](index.md#attribution).*
