---
type: Index
title: Workloads — AI Workload Taxonomy
description: The lifecycle (training → post-training → inference) and the shared operators/kernels underneath it.
tags: [workloads, taxonomy, training, post-training, inference, operators, kernels]
timestamp: 2026-08-22T00:00:00-07:00
---

# Workloads — AI Workload Taxonomy

← [Back to README](../README.md)

AI workloads span a **lifecycle** — training → post-training → inference — that
all runs on the same underlying **operators and kernels**. Performance
characteristics differ sharply across lifecycle stages, but the GEMMs, attention,
and collectives beneath them are shared.

---

## Lifecycle Subsections

| Stage | What it is | Bottleneck |
|-------|------------|------------|
| [Training](training/index.md) | Build the model: forward + backward over large datasets | Compute (backward ≈ 2× forward) + interconnect BW |
| [Post-training](post-training/index.md) | Shape behavior (SFT, preference optimization, RL/RLVR) and adapt/compress for serving (fine-tuning, distillation, quantization) | Varies (training-like for SFT/FT; rollout-throughput-bound for online RL; offline for quant) |
| [Inference](inference/index.md) | Serve the model in production: prefill + decode, optimization, ops | Prefill = compute; decode = HBM bandwidth |

The inference and post-training subsections are digested from Modular's
[LLM Inference Handbook](https://handbook.modular.com/).

---

## Regimes at a glance

| Regime | Compute Pattern | Memory Pattern | Bottleneck |
|--------|----------------|----------------|------------|
| Training | High FLOPs/batch, backward pass | Activations + weights in HBM | Compute or NVLink BW |
| Inference — prefill | Long prompt, high parallelism | KV cache write, weight load | Compute (high AI) |
| Inference — decode | One token/step, low batch | KV cache read, weight load | HBM bandwidth |

## Model Families

- Transformer / LLM — attention, MLP blocks; dominant modern workload
- CNN — convolutions; image/video; historically dominant
- [MoE (Mixture of Experts)](moe.md) — sparse activation; all-to-all, routing collapse, S-MFU, skinny GEMMs
- Diffusion models — iterative denoising; mixed attention + CNN

---

## Shared Operators & Kernels

These pages describe operators and kernel-development topics used across *all*
lifecycle stages (training, post-training, and inference alike).

### Operator-level view

- Operators overview — GEMM, attention, convolution, elementwise, reduction
- [GEMM](gemm.md) — general matrix multiply; workhorse of all DNN workloads
- [Attention](attention.md) — FlashAttention, paged attention, MLA
- [All-reduce / All-gather](collective_ops.md) — distributed training communication
- [Collectives on TPU & GPU clusters](collective_algorithms.md) — ring/tree algorithms on torus & fat-tree topologies
- [Mixture-of-Experts efficiency](moe.md) — sparse routing, expert parallelism, S-MFU

### Kernel development

- [What is a GPU kernel?](gpu_kernels.md) — launches, threads/warps, memory traffic, fusion, torch.compile & profiling; the on-ramp concept
- [Pallas](pallas_kernels.md) — JAX extension for custom GPU/TPU kernels; grids, Refs, memory spaces, TPU VMEM tiling
- [CUDA PTX](cuda_ptx.md) — NVIDIA's virtual ISA between CUDA C++ and SASS; compilation flow, registers, forward compatibility, inline PTX
- [XLA compiler](xla_compiler.md) — the compiler beneath JAX/TF/PyTorch; StableHLO → classic HLO optimizer → LLVM/Triton/Mosaic backends; PJRT, Pallas escape hatch

### Mechanistic understanding

- [Transformer as a programmable computer](transformer_programs.md) — manually-set weights implementing Hello World, Lookup Table, Search, Sort, Decimal Addition; attention hardening, linearly shiftable encodings, LayerNorm bypass

---

## Arithmetic Intensity by Operation

| Operation | Typical AI (FLOP/B) | Bound |
|-----------|---------------------|-------|
| Large GEMM (training) | 100–1000+ | Compute |
| Small GEMM (decode, batch=1) | < 1 | Memory BW |
| Attention (prefill, long seq) | ~seq_len/2 | Compute for long seq |
| Attention (decode) | << 1 | Memory BW |
| LayerNorm / Softmax | ~1–5 | Memory BW |
| All-reduce | — | NVLink / network BW |

---

## See Also

- [Roofline model](../modeling/roofline.md)
- [Analytical LLM inference model](../modeling/llm_inference.md)
