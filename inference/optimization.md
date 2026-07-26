---
type: Method
title: Inference Optimization (Serving Layer)
description: Serving-layer techniques to make LLM inference faster and cheaper — batching, PagedAttention, prefix caching, KV offloading, prefill-decode disaggregation, speculative decoding, parallelism, and routing.
tags: [inference-optimization, batching, paged-attention, prefix-caching, disaggregation, kv-cache]
resource: https://handbook.modular.com/inference-optimization/
timestamp: 2026-07-26T00:00:00-07:00
---

# Inference Optimization (Serving Layer)

← [LLM Inference Serving Index](index.md)

Digest of the handbook's [Inference optimization](https://handbook.modular.com/inference-optimization/)
chapter — the **system/runtime-layer** levers (distinct from
[kernel optimization](kernel-optimization.md), which is one level deeper in the
GPU code). These techniques target the [metrics](basics.md#metrics) TTFT, ITL,
throughput, and goodput.

---

## Batching

GPUs waste bandwidth reloading weights per request; batching reuses one weight
load across many requests, trading memory-bandwidth pressure for compute.

| Strategy | How it works | Limitation |
|----------|--------------|------------|
| **Static** | Wait for a fixed batch, run together | First request waits for the last; slowest sequence dictates completion |
| **Dynamic** | Time-window or size-limit, whichever first | Batches may launch under-full; longest request still gates the batch |
| **Continuous** (in-flight / persistent) | Iteration-level scheduling: as each sequence emits EOS, insert a new one in its slot | The standard for LLM serving (vLLM, SGLang, TensorRT-LLM, LMDeploy) |

**Continuous batching** maximizes GPU occupancy because output lengths vary
widely and no sequence waits for the batch's slowest member.

### Chunked prefill

A long prefill for a new request stalls the next token of every active decode.
**Chunked prefill** splits the prompt into token ranges scheduled across
iterations (mathematically equivalent to one pass, since every token still
attends through the KV cache). SARATHI's **decode-maximal batching** piggybacks
decode tokens onto a prefill chunk that saturates compute — smoothing ITL and
reducing pipeline bubbles. Tradeoff: smaller chunks lower ITL spikes but can
raise TTFT and re-read KV entries; chunk size is a workload-specific knob.

> Padding aligns ragged sequences into dense tensors but wastes compute;
> **ragged tensors** + paged KV layouts avoid it.

## PagedAttention

The serving win is a better **KV-cache allocator**, not a faster kernel.
Contiguous KV allocation sized for `max_seq_len` per request wastes memory
(the original paper: only 20.4–38.2% of allocated KV memory held real tokens).
PagedAttention stores KV in **fixed-size non-contiguous blocks** tracked by a
lookup table, cutting waste to near zero, enabling higher effective batch size,
and letting blocks be **shared** across outputs. First shipped in vLLM; adopted
by TGI and TensorRT-LLM. It's the substrate that makes
[continuous batching](#batching), [prefix caching](#prefix-caching), and
[KV offloading](#kv-cache-offloading) compose. Kernel-level attention efficiency
is separate — see [FlashAttention](kernel-optimization.md#flashattention).

## Prefix caching

Reuse the KV cache of a shared prompt **prefix** across requests (a.k.a. prompt
/ context caching). On a new request the engine finds the longest cached prefix
and runs prefill only for the remaining tokens.

- Requires an **exact** token-for-token match (whitespace/formatting included) —
  different from *semantic* caching.
- Big wins for repeated **system prompts**, growing **multi-turn chat**, agents,
  and RAG. Anthropic's prompt caching reports up to 90% cost / 85% latency
  reduction on long prompts; agent workflows with 100:1 input:output ratios
  benefit most.
- Best practices: front-load static content, avoid per-request variables
  (timestamps/IDs) in the prefix, use deterministic serialization, monitor
  hit-rate **per workload**.
- Limits: KV memory fills up (needs eviction/tiering), and feature composition is
  tricky (draft/target caches, VLM encoder states, sliding-window attention).
  A hit also requires [routing](#inference-routing) to a worker holding the cache.

## KV cache offloading

Move inactive/less-used KV blocks from GPU memory to CPU RAM, local SSD, or
remote storage, and reload on demand — freeing GPU memory for active requests
without recomputation.

- Useful for long contexts, idle/intermittent sessions, shared content across
  users/agents, and memory-constrained or distributed deployments.
- **The offload tier's speed is critical**: if transfer costs more than
  recompute, it's a loss (usually a win for long multi-turn contexts). NVIDIA
  GH200 reports up to 14× faster TTFT vs. recompute.
- **Quality risk** with *selective* offloading — dropping important context
  tokens degrades multi-doc QA / legal / codebase reasoning. Compare against a
  full-attention baseline on your workload; track answer quality alongside TTFT.
- **LMCache** is the common engine extension (tiers across GPU/CPU/disk; reuses
  KV for repeated content, not just prefixes); integrated by llm-d, KServe, vLLM
  (3×–10× latency reductions reported).

## Prefill-decode disaggregation

Run the compute-bound prefill and the memory-bandwidth-bound decode on
**separate** hardware so they stop interfering. Benefits: independent scaling &
tuning per phase, parallel execution, and better tail latency (a long prefill no
longer stalls in-flight decodes). Supported by SGLang, vLLM, Dynamo, and llm-d.

Not a silver bullet: below a workload threshold it can *lose* 20–30%; short or
cache-heavy prompts are faster prefilled locally; and the **KV-cache transfer**
(via NIXL, CXL, NVMe-oF) must be cheaper than the prefill it saves, with prefill
and decode workers agreeing on KV layout/dtype/attention variant.

**Cross-cluster PD** pushes this across datacenters to use the right chip per
phase (e.g. NVIDIA Rubin CPX for prefill, Groq LPU for decode). It becomes a
routing problem — *Prefill-as-a-Service* keeps short/cached prompts local and
sends only long uncached prefills remote (reported 54% higher throughput, 64%
lower P90 TTFT). See [inference routing](../modeling/inference_routing.md).

## Speculative decoding

A small **draft** model proposes several tokens that the **target** model
verifies in parallel, exploiting the memory-bound decode regime to emit multiple
tokens per target pass — losslessly (accepted tokens match target sampling). The
KB has a full modeling treatment: **[speculative decoding](../modeling/speculative_decoding.md)**
(draft-and-verify acceptance, EAGLE/Medusa, tree drafting).

## Parallelism

Split a model that doesn't fit (or to hit a latency SLO) across GPUs: data (DP),
tensor (TP), pipeline (PP), and expert (EP) parallelism, plus hybrids. Keep the
most communication-intensive dimension within a node (NVLink) where possible.
Full modeling in **[parallelism](../modeling/parallelism.md)**; the MoE all-to-all
angle is in [MoE workloads](../workloads/moe.md).

## Inference routing

At scale a naive round-robin balancer breaks down. State-aware routing sends each
request to the worker with useful **KV-cache locality** or spare capacity, honors
session affinity, and coordinates PD disaggregation. Modeled in the KB under
**[inference routing](../modeling/inference_routing.md)**; runtimes include
[llm-d](../scheduling/llm_d.md).

## Related knobs

- **Offline / batch inference** — throughput-first, latency-relaxed processing of
  large request sets (library-mode engines; explicit `max_tokens` so one bad
  input doesn't dominate).
- **Benchmarks** — compare frameworks/GPUs on *your* model with TTFT/ITL/goodput,
  not vendor peak numbers; see [Characterization](../characterization/index.md).

---

## See Also

- [Inference basics](basics.md) — prefill/decode, KV cache, the target metrics
- [LLM inference model](../modeling/llm_inference.md) — why decode is memory-bound
- [Speculative decoding](../modeling/speculative_decoding.md) · [Inference routing](../modeling/inference_routing.md) · [Parallelism](../modeling/parallelism.md)
- [Kernel optimization](kernel-optimization.md) — FlashAttention & the layer below
- [llm-d](../scheduling/llm_d.md) — distributed inference / KV-aware routing runtime
