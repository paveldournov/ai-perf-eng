---
type: Index
title: LLM Inference Serving
description: Production LLM inference — serving, optimization, kernels, infrastructure/ops, model interaction, and model preparation, digested from Modular's LLM Inference Handbook.
tags: [inference, serving, production, llm, kv-cache, optimization]
resource: https://handbook.modular.com/
timestamp: 2026-07-26T00:00:00-07:00
---

# LLM Inference Serving

← [Back to README](../README.md)

This section covers the **production side** of LLM inference: how to serve,
optimize, scale, and operate LLMs once the analytical performance is understood.
It complements [Performance Modeling](../modeling/index.md) (which reasons
*about* inference analytically) by focusing on the concrete serving stack:
frameworks, KV-cache management, batching schedulers, GPU kernels, deployment
patterns, and inference operations.

The material is digested from Modular's **[LLM Inference Handbook](https://handbook.modular.com/)**
— "a practical guide for understanding, optimizing, scaling, and operating LLM
inference systems." The handbook's `docs/` content is licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); these pages summarize
and cross-link it, with links back to the canonical chapters for full depth and
the handbook's interactive calculators/visualizers.

---

## Chapters

Each page here mirrors one chapter of the handbook and links out to its
individual topic pages.

| Page | Covers | Handbook chapter |
|------|--------|------------------|
| [Inference basics](basics.md) | Prefill/decode, KV cache, context window, sampling, latency/throughput/goodput metrics, CPU vs GPU vs TPU, training vs inference, diffusion LLMs | [llm-inference-basics](https://handbook.modular.com/llm-inference-basics/) |
| [Planning a deployment](getting-started.md) | Serverless vs self-hosted, choosing GPU/model/framework, GPU memory sizing, BYOC/on-prem/edge | [getting-started](https://handbook.modular.com/getting-started/) |
| [Inference optimization](optimization.md) | Batching (static/dynamic/continuous, chunked prefill), PagedAttention, prefix caching, KV offloading, prefill-decode disaggregation, speculative decoding, parallelism, routing | [inference-optimization](https://handbook.modular.com/inference-optimization/) |
| [Kernel optimization](kernel-optimization.md) | GPU execution & memory hierarchy, tensor cores, occupancy, FlashAttention (v1–v4), kernel tooling (cuBLAS/cuDNN, TVM/XLA, Triton, custom, Mojo/MAX) | [kernel-optimization](https://handbook.modular.com/kernel-optimization/) |
| [Infrastructure & operations](infrastructure-ops.md) | Inference infrastructure, distributed inference, fast scaling & cold starts, observability, InferenceOps, cost, multi-cloud, multi-model pipelines | [infrastructure-and-operations](https://handbook.modular.com/infrastructure-and-operations/) |
| [Model interaction](model-interaction.md) | Inference parameters, structured outputs / constrained decoding, function calling, MCP, OpenAI/Anthropic-compatible APIs, prompt engineering | [model-interaction](https://handbook.modular.com/model-interaction/) |
| [Model preparation](model-preparation.md) | Quantization, fine-tuning (LoRA/QLoRA), distillation | [model-preparation](https://handbook.modular.com/model-preparation/) |

---

## Where serving meets the rest of the knowledge base

- **[Performance Modeling](../modeling/index.md)** — the analytical two-phase
  (prefill/decode) latency/throughput model behind these serving techniques
  lives in [LLM inference model](../modeling/llm_inference.md);
  [speculative decoding](../modeling/speculative_decoding.md) and
  [inference routing](../modeling/inference_routing.md) have dedicated modeling
  pages that these serving digests link back to.
- **[Workloads](../workloads/index.md)** — the kernel/operator side:
  [attention](../workloads/attention.md), [GEMM](../workloads/gemm.md),
  [GPU kernels](../workloads/gpu_kernels.md).
- **[Scheduling](../scheduling/index.md)** — distributed runtimes and inference
  routers ([llm-d](../scheduling/llm_d.md), [Ray](../scheduling/ray.md)).
- **[Characterization](../characterization/index.md)** — benchmarking
  methodology for the metrics defined in [basics](basics.md).
- **[Hardware](../hardware/index.md)** — the accelerators these workloads run on.

---

## See Also

- [Reading Guide](../GUIDE.md)
- [References](../references/index.md) — the handbook and its cited papers
