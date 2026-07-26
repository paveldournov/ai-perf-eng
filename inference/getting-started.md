---
type: Method
title: Planning an LLM Deployment
description: Decision guide for deploying LLMs — serverless vs self-hosted, choosing GPU/model/framework, GPU-memory sizing, and BYOC/on-prem/edge patterns.
tags: [deployment, gpu-selection, vram, inference-framework, serverless, self-hosted]
resource: https://handbook.modular.com/getting-started/
timestamp: 2026-07-26T00:00:00-07:00
---

# Planning an LLM Deployment

← [LLM Inference Serving Index](index.md)

Digest of the handbook's [Planning your deployment](https://handbook.modular.com/getting-started/)
chapter — the early decisions that shape infrastructure, cost, and performance.

---

## Serverless vs. self-hosted

| | Serverless APIs (OpenAI, Anthropic, …) | Self-hosted (cloud GPUs, VPC, on-prem) |
|--|--|--|
| Ease of use | High (API key + a few lines) | Lower (deploy & maintain) |
| Data privacy / compliance | Limited | Full control |
| Customization | Limited | Full (batching, disaggregation, fine-tuning) |
| Cost at scale | Higher, scales linearly with usage | Potentially much lower per token |
| Hardware mgmt | Abstracted | You own GPU setup & ops |

Start serverless for prototyping; take control as needs for **performance,
privacy, cost, and differentiation** grow. Self-hosting adds responsibilities:
DevOps, monitoring, data-transfer/storage cost, redundancy, and
[cold starts](infrastructure-ops.md#fast-scaling--cold-starts).

## Choosing the right GPU

Raw benchmark numbers rarely capture real LLM workloads. The four hardware axes:

| Axis | Why it matters |
|------|----------------|
| **VRAM (HBM capacity)** | Ceiling on model size + context. Weights must fit first; the KV cache then consumes what's left. DeepSeek V3/R1 (671B) needs 8×H200 (141 GB); Phi-3 fits in 16–24 GB quantized. |
| **Memory bandwidth** | The binding constraint for the **memory-bound decode phase**. Single-stream ceiling ≈ `memory_bandwidth / bytes_read_per_token`. A 70B FP16 model (~140 GB) on H100 SXM (~3.35 TB/s) ⇒ ~24 tok/s per sequence before overhead. |
| **Compute throughput (FLOPS)** | Limits the **compute-bound prefill phase** and large batches. Watch the asterisks: vendors quote peak FLOPS with sparsity / lowest precision. FP8 needs Hopper+; lower precision ~doubles the rate. |
| **Interconnect** | Governs multi-GPU throughput. Intra-node NVLink (H100: 900 GB/s bidirectional, ~7× a PCIe 5.0 link) vs inter-node InfiniBand/RoCEv2. Keep the most communication-intensive [parallelism](../modeling/parallelism.md) within one node. |

**GPU CAP theorem** — a GPU supply cannot guarantee **Control**, on-demand
**Availability**, and low **Price** simultaneously:

| | Hyperscaler | NeoCloud (serverless) | NeoCloud (committed) | On-prem |
|--|--|--|--|--|
| Control | High | Low | Medium | High |
| Availability | Medium | High | Low | Low |
| Price | High | Medium | Low | Medium |

Also mind **ecosystem support** (NVIDIA CUDA/TensorRT-LLM is mature; AMD ROCm is
catching up) and **driver/CUDA version alignment** (driver's CUDA version must be
≥ the toolkit the framework was built with, or you lose FP8/FlashAttention).

## Sizing GPU memory

Weights are the fixed baseline; the [KV cache](basics.md#kv-cache) is the largest
runtime overhead. A rough sizing shortcut:

```
Memory (GB) = P × (Q / 8) × (1 + Overhead)
```

- **P** = parameters (billions), **Q** = bit precision (16, 8, …; ÷8 → bytes),
  **Overhead** = KV cache + activations + workspace + framework reservations.

The flat 10–30% overhead is only a shortcut — real KV-cache memory depends on
sequence length, batch size, concurrency, layers, and hidden size, so
long-context workloads need far more headroom. Use the KB's
[memory capacity model](../modeling/memory_capacity.md) for the exact KV formula,
and [quantization](model-preparation.md#quantization) to shrink the weight
footprint (freeing room for KV cache and larger batches).

## Choosing the right model

- **Base vs instruct vs chat** — base (foundation) models predict next tokens;
  *instruct* models are fine-tuned to follow prompts; *chat* models are further
  tuned (RLHF/DPO) for multi-turn dialogue.
- **Dense vs MoE** — dense uses every parameter per token; Mixture-of-Experts
  activates only a subset of experts per token for efficient scaling (naming like
  `Qwen3.5-35B-A3B` = total-B / active-B). See [MoE workloads](../workloads/moe.md).
- **Compose with other models** — SLMs, embedding models, VLMs, image/TTS models
  in [multi-model pipelines](infrastructure-ops.md#multi-model-pipelines).
- **Where to get them** — Hugging Face (default; mind *gated* models needing a
  token), ModelScope (strong Chinese/multilingual), OpenRouter (access layer).
- **Weight formats** — PyTorch checkpoints (`.bin`/`.pt`, can execute arbitrary
  code), **Safetensors** (safe, fast mmap; now standard, often sharded), **GGUF**
  (quantized, for local/`llama.cpp`).

## Choosing the right inference framework

A framework loads the model and handles serving work raw PyTorch does not:
batching/scheduling, KV-cache management, streaming, memory control, production
APIs, multi-GPU. (vLLM reported up to 24× the throughput of HF Transformers on
the same hardware.)

| Framework | Best for |
|-----------|----------|
| **vLLM**, **SGLang**, **MAX**, **LMDeploy**, **TensorRT-LLM** | High-throughput, batched, data-center serving |
| **llama.cpp**, **MLC-LLM** | Edge / desktop / low-resource, broad hardware |
| **Ollama** | Local single-user experimentation (no concurrency) |
| SGLang Diffusion, vLLM-Omni, MAX | Diffusion / multimodal generation |

> Hugging Face **TGI** is now in maintenance mode — plan an upgrade path.

- **Library mode** (embed the engine in-process) suits offline batch jobs and
  eval pipelines; **server mode** (standalone process, usually
  [OpenAI-compatible](model-interaction.md#api-compatibility) HTTP) suits shared
  models, multi-language clients, and independent scaling.
- No single runtime wins everywhere — keep infrastructure **runtime-agnostic**.

A common scaling path: Ollama (laptop) → vLLM/SGLang/MAX (data-center runtime) →
[distributed inference platform](infrastructure-ops.md#distributed-inference)
(multi-cluster/region routing, autoscaling, observability).

## Deployment patterns

- **Cloud** — on-demand GPUs, managed autoscaling/monitoring; the default.
- **Multi-cloud / cross-region** — lower latency for global users, better GPU
  availability, avoids lock-in, meets data residency. See
  [multi-cloud inference](infrastructure-ops.md).
- **BYOC (Bring Your Own Cloud)** — vendor software runs inside your cloud
  account: managed orchestration + your data/network/cost control.
- **On-prem** — full control over data/performance/compliance; more ops
  overhead.
- **Edge** — model runs on-device/near-data for latency and privacy; smaller,
  optimized models.

---

## See Also

- [Inference basics](basics.md) — prefill/decode & the metrics you're optimizing
- [Memory capacity model](../modeling/memory_capacity.md) — exact weight + KV sizing
- [Model preparation](model-preparation.md) — quantization / fine-tuning / distillation
- [Infrastructure & operations](infrastructure-ops.md) — scaling and running it
- [Hardware](../hardware/index.md) — accelerator specs (H100, TPU, …)
