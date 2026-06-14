---
type: Index
title: Scheduling & Orchestration
description: How jobs are admitted, placed, and share accelerators across users and clusters.
tags: [scheduling, orchestration, queuing]
timestamp: 2026-05-30T10:50:39-07:00
---

# Scheduling & Orchestration

← [Back to README](../README.md)

Scheduling sits between the hardware and the workload: it decides when jobs run, where they run, and how resources are shared across users and clusters. For large-scale AI, scheduling has become a first-class performance concern — poor scheduling wastes expensive accelerator time, causes head-of-line blocking, and inflates queuing latency.

---

## Scope

| Layer | What it decides | Examples |
|-------|----------------|---------|
| Cluster-level job admission | Which jobs get resources and when | [Kueue](kueue.md), SLURM |
| Distributed runtime | How a single job spans many devices | [Pathways](pathways.md), [Ray](ray.md) |
| Inference serving | Which request goes to which replica, KV routing | [llm-d](llm_d.md), [Ray Serve](ray.md) |

---

## Pages

- [Pathways on Cloud](pathways.md) — Google's single-controller runtime for multi-slice JAX/TPU workloads; PathwaysJob CRD, resilient training, multihost inference
- [Ray / KubeRay](ray.md) — distributed Python runtime; Ray Train for multi-node training, Ray Serve for inference, KubeRay for Kubernetes integration
- [Kueue](kueue.md) — Kubernetes-native job queueing and admission control; ClusterQueues, fair-sharing, topology-aware scheduling
- [llm-d](llm_d.md) — CNCF inference serving stack; disaggregated prefill/decode, KV-cache-aware routing, SLO-aware autoscaling

---

## Key Concepts

**Admission control** — the decision to allow a job to start. Kueue holds jobs in a queue until sufficient quota is available, preventing over-subscription.

**Resource flavors** — named resource variants (e.g. `h100-80gb`, `tpu-v6e-8`) that allow schedulers to express heterogeneous hardware and enforce placement.

**Fair-sharing / preemption** — multi-tenant clusters need policies to reclaim resources from lower-priority jobs without starvation.

**Disaggregated prefill/decode** — separating the compute-heavy prefill phase from the memory-bandwidth-heavy decode phase onto different hardware, then routing requests accordingly. Key for inference throughput (see [llm-d](llm_d.md)).

**KV-cache routing** — directing a request to the replica that already holds the relevant KV cache prefix, avoiding recomputation. Can yield 2–3× throughput improvement on repeated or shared prefixes.

**Single-controller vs. multi-controller** — whether one process orchestrates all devices (Pathways) or each process manages its own local devices (standard JAX/PyTorch).

---

## See Also

- [LLM inference model](../modeling/llm_inference.md) — prefill/decode latency model that scheduling decisions directly affect
- [Inference routing](../modeling/inference_routing.md) — analytical treatment of KV-aware routing
- [TPU hardware](../hardware/tpu/index.md) — Pathways targets
- [Attention](../workloads/attention.md) — the operator whose KV cache scheduling optimizes around
