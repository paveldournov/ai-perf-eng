---
type: Index
title: Inference Workloads
description: The inference stage of the LLM lifecycle — serving, optimization, kernels, infrastructure/ops, and model interaction, digested from Modular's LLM Inference Handbook.
tags: [inference, serving, production, llm, kv-cache, optimization]
resource: https://handbook.modular.com/
timestamp: 2026-07-26T00:00:00-07:00
---

# Inference Workloads

← [Workloads Index](../index.md)

The **inference** stage of the workload lifecycle (see also
[training](../training/index.md) and [post-training](../post-training/index.md)):
how to serve, optimize, scale, and operate LLMs in production. It builds on the
[shared operators & kernels](../index.md#shared-operators--kernels)
(attention, GEMM, collectives) and on the analytical
[LLM inference model](../../modeling/llm_inference.md) in Performance Modeling.

Digested from Modular's **[LLM Inference Handbook](https://handbook.modular.com/)**
(source `docs/` licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/));
these pages summarize and cross-link it, with links back to the canonical
chapters for full depth and the handbook's interactive tools.

---

## Pages

| Page | Covers | Handbook chapter |
|------|--------|------------------|
| [Inference basics](basics.md) | Prefill/decode, KV cache, context window, sampling, latency/throughput/goodput metrics, CPU vs GPU vs TPU | [llm-inference-basics](https://handbook.modular.com/llm-inference-basics/) |
| [Planning a deployment](getting-started.md) | Serverless vs self-hosted, choosing GPU/model/framework, GPU memory sizing, BYOC/on-prem/edge | [getting-started](https://handbook.modular.com/getting-started/) |
| [Inference optimization](optimization.md) | Batching, PagedAttention, prefix caching, KV offloading, prefill-decode disaggregation, speculative decoding, parallelism, routing | [inference-optimization](https://handbook.modular.com/inference-optimization/) |
| [Kernel optimization](kernel-optimization.md) | GPU execution & memory hierarchy, tensor cores, occupancy, FlashAttention (v1–v4), kernel tooling | [kernel-optimization](https://handbook.modular.com/kernel-optimization/) |
| [Infrastructure & operations](infrastructure-ops.md) | Distributed inference, fast scaling & cold starts, observability, InferenceOps, cost, multi-cloud, multi-model pipelines | [infrastructure-and-operations](https://handbook.modular.com/infrastructure-and-operations/) |
| [Model interaction](model-interaction.md) | Inference parameters, structured outputs / constrained decoding, function calling, MCP, OpenAI/Anthropic APIs, prompt engineering | [model-interaction](https://handbook.modular.com/model-interaction/) |

> Model **preparation** for inference — quantization, fine-tuning, distillation —
> lives in [post-training](../post-training/model-preparation.md).

---

## See Also

- [LLM inference model](../../modeling/llm_inference.md) — the analytical prefill/decode latency & throughput equations
- [Attention](../attention.md) · [GEMM](../gemm.md) · [GPU kernels](../gpu_kernels.md) — the operators/kernels inference runs on
- [Scheduling](../../scheduling/index.md) — distributed runtimes & inference routers ([llm-d](../../scheduling/llm_d.md))
- [Characterization](../../characterization/index.md) — measuring TTFT/TPOT/throughput
- [References](../../references/index.md) — the handbook and its cited papers
