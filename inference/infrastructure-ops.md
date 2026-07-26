---
type: Method
title: Inference Infrastructure & Operations
description: Running LLM inference in production — infrastructure components, distributed inference, fast scaling and cold starts, observability, InferenceOps, cost, and multi-cloud/multi-model deployment.
tags: [infrastructure, distributed-inference, autoscaling, cold-start, observability, inferenceops, cost]
resource: https://handbook.modular.com/infrastructure-and-operations/
timestamp: 2026-07-26T00:00:00-07:00
---

# Inference Infrastructure & Operations

← [LLM Inference Serving Index](index.md)

Digest of the handbook's [Infrastructure and operations](https://handbook.modular.com/infrastructure-and-operations/)
chapter — the layer that determines how far you can scale and how reliably you
can grow, once the model and [optimization](optimization.md) are in place. This
is the operational complement to the KB's [Scheduling](../scheduling/index.md)
section.

---

## What inference infrastructure includes

Hardware provisioning, model deployment (packaging weights/config/adapters for
safe release & rollback), the serving runtime, orchestration (resource
allocation, scaling, versioning), observability, security/access control,
cost-to-serve management, and repeatable operational procedures.

## Distributed inference

Spreading inference across machines so no single device is a bottleneck. Two
levels:

- **Macro** — *where* inference runs: multiple regions, heterogeneous GPU
  clusters, [multi-cloud](#multi-cloud--cross-region) and on-prem, ideally
  presented as one logical serving layer with multi-region routing, failover, and
  elastic scaling.
- **Micro** — *how* a request is split for efficiency:
  [PD disaggregation](optimization.md#prefill-decode-disaggregation),
  [KV offloading](optimization.md#kv-cache-offloading),
  [routing](optimization.md#inference-routing), and
  [parallelism](../modeling/parallelism.md).

**Why:** scale beyond one GPU, serve models too large for one device, fault
tolerance, and cost via smarter resource use. **Challenges:** network overhead
(often the bottleneck, not compute), build/operational complexity, unified
observability & cost attribution, and stateful KV-cache/session management. Teams
typically run vLLM/SGLang/[llm-d](../scheduling/llm_d.md) on Kubernetes; those
solve micro-level concerns but not macro-level routing/autoscaling/visibility on
their own.

## Fast scaling & cold starts

Inference demand is bursty and unforgiving, so systems must scale up fast and
scale **to zero** when idle. Serverless (Lambda-style) doesn't map cleanly to AI:
most platforms lack GPU support, GPUs don't slice easily, and idle GPUs are
expensive.

The **cold start** problem has three stages:

1. **Cloud provisioning** — allocating a GPU node (30 s to minutes, or hours for
   scarce H100/A100).
2. **Container image pull** — LLM images are large; pulls take 3–5 min.
3. **Model loading** — billions of parameters flow `remote storage → local disk →
   memory → GPU` with little parallelism; hubs like Hugging Face aren't tuned for
   fast multipart downloads, and weights must fully land before serving starts.

**Scaling metric choice** matters: CPU util is misleading (Python GIL), GPU util
is inaccurate (`nvml` marks a GPU "used" for any kernel in the sample), QPS varies
too much per request. **Concurrency** (active queued/processing requests) is the
best signal — it correlates with load and maps to batch size, but needs framework
instrumentation.

## Observability

End-to-end visibility via metrics + logs + events across layers:

| Layer | Example metrics |
|-------|-----------------|
| Container/deployment | pod status, replica count |
| App performance | RPS, request latency, in-progress requests, error rate, queue wait |
| Cluster resources | quotas & limits |
| LLM-specific | tokens/sec, TTFT, total generation time |
| GPU | utilization, memory usage |

Metrics say *what*; events (restarts, scaling, scheduling delays) and aggregated
logs say *why*. These are the [basics metrics](basics.md#metrics) measured in
production.

## InferenceOps

The practices for ongoing deployment/updates/management at scale:

- **Standardized deployment** — CI/CD for models (regression/latency/token
  checks, reviewed infra changes), plus **canary** and **blue-green** releases.
- **Safe updates & fault tolerance** — rolling updates, automatic rollback +
  alerting, and fault isolation (retries, timeouts, circuit breakers, load
  shedding).
- **Centralized management** — model registry & lifecycle tracking, a unified
  control plane, multi-region/multi-cloud coordination to avoid drift.
- **Cost & resource hygiene** — idle-GPU cleanup, access control, and audit logs.

## Cost

Inference cost is ongoing and scales with traffic, so cost-to-serve is a
first-class concern: track per-workload cost and tune model choice, hardware,
batching, routing, and scaling policies. Self-hosting has higher upfront cost but
lower per-token cost at scale (especially with
[KV offloading](optimization.md#kv-cache-offloading) and
[quantization](model-preparation.md#quantization)); serverless is cheaper to
start but scales linearly. Both are getting cheaper over time (API price cuts,
more efficient GPUs, better open models and runtimes).

## Multi-cloud & cross-region

Distribute workloads across providers/regions to cut latency for global users,
improve GPU availability, optimize cost, avoid vendor lock-in, and satisfy data
residency — routed as one logical layer with seamless failover.

## Multi-model pipelines

Production apps rarely use one model. Compose LLMs with embedding models (for
**RAG**), SLMs, VLMs, and image/TTS models into pipelines where each stage
consumes the previous stage's output. RAG in particular is often the better
answer than [fine-tuning](model-preparation.md#fine-tuning) when the problem is
missing or fast-changing information, since it fetches fresh data at inference
time. See [choosing a model](getting-started.md#choosing-the-right-model).

---

## See Also

- [Scheduling](../scheduling/index.md) — job admission & distributed runtimes ([Kueue](../scheduling/kueue.md), [Ray](../scheduling/ray.md), [llm-d](../scheduling/llm_d.md), [Pathways](../scheduling/pathways.md))
- [Inference optimization](optimization.md) — the micro-level techniques
- [Fault tolerance](../simulation/fault_tolerance.md) — resilience modeling
- [Planning a deployment](getting-started.md) — serverless vs self-hosted, GPU choice
