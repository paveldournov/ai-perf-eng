---
type: Concept
title: LLM Inference Basics
description: Foundations of LLM inference — prefill/decode, KV cache, context window, sampling, and the latency/throughput/goodput metrics used to measure serving.
tags: [inference, prefill, decode, kv-cache, metrics, ttft, tpot, goodput]
resource: https://handbook.modular.com/llm-inference-basics/
timestamp: 2026-07-26T00:00:00-07:00
---

# LLM Inference Basics

← [LLM Inference Serving Index](index.md)

Digest of the handbook's [Foundations](https://handbook.modular.com/llm-inference-basics/)
chapter. For the analytical latency/throughput equations, see
[LLM inference model](../modeling/llm_inference.md).

---

## Inference vs. serving

- **Inference** is the computation: tokens in → next-token distribution out,
  repeated through prefill and the decode loop. You can run it in a notebook.
- **Serving** is everything needed to make that a production service: an
  HTTP/gRPC API, request queuing/scheduling, batching, load balancing,
  autoscaling, model version management, health checks, and observability.

Modern frameworks (vLLM, SGLang, TensorRT-LLM, MAX) do both, so the terms are
often used interchangeably — but the distinction matters the moment you design
infrastructure. See [Planning a deployment](getting-started.md).

## The two phases

Decoder-only Transformers run inference in two phases with different bottlenecks
(the core insight the whole serving stack is built around):

| Phase | Work | Bottleneck | Key metric |
|-------|------|------------|------------|
| **Prefill** | Process the whole prompt in parallel; build the KV cache | Compute-bound (saturates the GPU) | [TTFT](#metrics) |
| **Decode** | Autoregressive generation, one token per step, reusing the KV cache | Memory-bandwidth-bound (streams weights + KV each step) | [ITL / TPOT](#metrics) |

Because decode reloads the weights and the growing KV cache from HBM for every
single token, it is memory-bound and cannot be parallelized across the sequence
— which is why LLM inference is "slow" and why so much optimization targets the
decode regime. The quantitative model of this split is in
[LLM inference model](../modeling/llm_inference.md).

### Collocation problem

Running prefill and decode on the same GPU means only one phase runs at a time:
a compute-heavy prefill stalls in-flight decodes (raising ITL and tail latency),
and vice versa. Separating them is
[prefill-decode disaggregation](optimization.md#prefill-decode-disaggregation).

## KV cache

The **KV cache** stores the key (K) and value (V) vectors already computed for
every token at every layer, so each decode step only computes K/V for the *new*
token and reads the rest back from memory instead of recomputing all prior
tokens. It is the structure most serving optimizations revolve around.

- It grows **linearly with sequence length** and with the number of concurrent
  requests, and it is usually what limits how many users a server can handle —
  not the model weights.
- Example (the handbook's): Llama 3 8B in FP16 — weights ≈ 16 GB, and one
  8K-token sequence ≈ 1 GB of KV cache. On an 80 GB GPU that leaves ~64 GB,
  bounding concurrency to ~60 such sequences.
- Techniques built on it: [PagedAttention](optimization.md#pagedattention),
  [prefix caching](optimization.md#prefix-caching),
  [KV cache offloading](optimization.md#kv-cache-offloading),
  [PD disaggregation](optimization.md#prefill-decode-disaggregation), and KV-cache
  [quantization](model-preparation.md#quantization).

See the KV-cache sizing formula in [memory capacity](../modeling/memory_capacity.md)
and [LLM inference model](../modeling/llm_inference.md#kv-cache-size).

## Context window & sampling

- **Context window** — the max tokens processed in one pass (e.g. 8K/32K/128K).
  LLMs have no real memory; the *entire* conversation is re-sent each turn, which
  is what makes [prefix caching](optimization.md#prefix-caching) so valuable.
- **Sampling** — the model outputs a probability distribution over the vocab at
  each step; `temperature` reshapes it, then a strategy (greedy, top-k, top-p)
  picks the token. Details in
  [inference parameters](model-interaction.md#sampling-parameters).

## Metrics

The vocabulary the rest of the serving stack optimizes toward. See
[the handbook's metrics page](https://handbook.modular.com/llm-inference-basics/llm-inference-metrics/)
and the [Characterization](../characterization/index.md) section for measurement
methodology.

### Latency

| Metric | Definition |
|--------|------------|
| **TTFT** (Time to First Token) | Prompt submission → first token. Dominated by prefill. |
| **TPOT** (Time per Output Token) | `(E2EL − TTFT) / (output_tokens − 1)`; request-weighted. |
| **ITL** (Inter-Token Latency) | Pause between two consecutive tokens; token-weighted. Mean ITL = TPOT for a single request, but they differ when averaged across requests. |
| **E2EL** (Total Latency) | Request sent → final token received. |

Always report the distribution, not one number: **mean** (trend, skewed by
outliers), **median** (typical user), **P99** (tail — often what SLAs guarantee).
Always check *how* a benchmark defines TPOT vs. ITL before comparing systems.

### Throughput

- **RPS** (Requests/sec) — coarse; ignores work per request.
- **TPS** (Tokens/sec) — finer; always check whether it means *input*, *output*,
  or combined. TPS is easy to game (short prompts inflate it).

### Goodput & SLOs

**Goodput** = requests/sec completed *while meeting the SLO* (e.g. "95% of chats
under 200 ms TTFT"). High throughput with missed latency targets produces
unusable requests, so goodput is the metric that reflects real service quality.
The whole latency-vs-throughput tradeoff (big batches raise throughput but hurt
per-user latency) is tuned against the SLO of the specific workload.

| Use case | Primary metric |
|----------|----------------|
| Interactive chat | TTFT, then ITL/TPOT |
| Long-form streaming | ITL/TPOT and E2EL |
| Agentic / multi-step | E2EL |
| High-volume offline | TPS and cost/token |
| Latency-constrained online | Goodput |

## Where inference runs: CPU vs GPU vs TPU

- **CPU** — general-purpose; fine for small models or infrequent requests, but
  lacks parallelism for production LLMs.
- **GPU** — the default for training and inference; matrix/tensor-op optimized,
  broadest ecosystem (CUDA/ROCm). See
  [GPU architecture fundamentals](kernel-optimization.md#gpu-execution-model).
- **TPU** — Google ASICs built ground-up for dense tensor ops; very high memory
  bandwidth and power efficiency, accessed mainly via cloud (XLA/JAX). See the
  KB's [TPU family](../hardware/tpu/index.md).

**Deployment patterns:** cloud (default), multi-cloud/cross-region, BYOC,
on-prem, and edge — see [Planning a deployment](getting-started.md).

## Training vs. inference

Training builds the model (one-time, batch, long, expensive); inference uses it
(continuous, real-time, scales with traffic). Because inference runs on *every*
request, its cumulative cost often exceeds training over a model's life.
Fine-tuning is a (small) form of training — see
[model preparation](model-preparation.md).

## Diffusion LLMs (dLLMs)

An emerging alternative to autoregressive decoding: dLLMs generate the whole
response in parallel via iterative denoising (à la image diffusion), removing the
token-by-token bottleneck and enabling self-correction. Early examples: Inception
AI's Mercury, Google's Gemini Diffusion. Autoregressive models remain mainstream.

---

## See Also

- [LLM inference model](../modeling/llm_inference.md) — the analytical prefill/decode latency & throughput equations
- [Memory capacity model](../modeling/memory_capacity.md) — weights + KV-cache sizing
- [Inference optimization](optimization.md) — techniques targeting these metrics
- [Attention](../workloads/attention.md) — the mechanism behind the KV cache
- [Characterization](../characterization/index.md) — measuring TTFT/TPOT/throughput
