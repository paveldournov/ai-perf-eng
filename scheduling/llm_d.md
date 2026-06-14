---
type: Tool
title: llm-d
description: Kubernetes-native distributed LLM inference serving stack (CNCF Sandbox).
tags: [scheduling, inference, kubernetes, cncf, serving]
resource: https://github.com/llm-d/llm-d
timestamp: 2026-05-30T10:50:39-07:00
---

# llm-d

← [Back to Scheduling](index.md)

---

**Source:** [github.com/llm-d/llm-d](https://github.com/llm-d/llm-d) · CNCF Sandbox (March 2026)
**Founding members:** Red Hat, Google Cloud, IBM Research, CoreWeave, NVIDIA

llm-d is a high-performance distributed LLM inference serving stack for Kubernetes. It layers scheduling intelligence — KV-cache-aware routing, disaggregated prefill/decode, SLO-aware autoscaling — on top of standard model servers (primarily vLLM) rather than replacing them.

---

## Why Standard Serving Falls Short

A naive serving setup routes requests round-robin across replicas. This ignores two key properties of LLM inference:

1. **KV cache is expensive to recompute.** If a replica already cached the tokens for a system prompt or shared prefix, routing a new request there avoids recomputation entirely.
2. **Prefill and decode have different hardware needs.** Prefill is compute-bound (processes the full prompt in parallel); decode is memory-bandwidth-bound (one token at a time, reading all weights). Running both on the same hardware forces a compromise.

llm-d addresses both with intelligent routing and disaggregation.

---

## Architecture

```
                    ┌─────────────────────┐
  Client requests → │   llm-d Inference   │
                    │   Gateway / Router  │
                    └──────┬──────┬───────┘
                           │      │
              ┌────────────▼──┐ ┌─▼─────────────────┐
              │  Prefill Pool │ │    Decode Pool     │
              │  (compute-    │ │  (BW-optimized     │
              │   optimized)  │ │   replicas)        │
              └───────────────┘ └────────────────────┘
                                        │
                    ┌───────────────────▼──────────────┐
                    │   Global KV Cache State Index    │
                    │   (tracks which replica holds    │
                    │    which prefix hashes)          │
                    └──────────────────────────────────┘
```

**Inference Gateway / Router** — receives incoming requests and applies routing policies. Plugs into Kubernetes via the Gateway API; works with any Gateway API-compatible ingress.

**Prefill Pool** — replicas dedicated to prompt processing. Compute-bound; benefits from high FLOP/s hardware.

**Decode Pool** — replicas dedicated to token generation. Memory-bandwidth-bound; benefits from high HBM bandwidth.

**Global KV Cache State Index** — distributed index tracking which replicas hold cached KV state for which token prefix hashes. The router queries this to make cache-aware placement decisions.

---

## Routing Policies

### Prefix-Cache Routing

Hashes the request prefix and routes to the replica that already holds matching KV cache state. Avoids recomputing the system prompt or shared context on every request.

**Measured gain:** 3× higher output throughput, 2× lower TTFT vs. round-robin on Llama 3.1 70B / AMD MI300X.

### Predicted-Latency Scheduling

Estimates per-request latency based on prompt length, current queue depth, and replica load. Routes to minimize expected end-to-end latency rather than queue length alone.

**Measured gain:** 40% TTFT reduction on NVIDIA GPUs.

### Load-Aware Balancing

Tracks per-replica utilization in real time. Steers away from overloaded replicas before they cause queuing.

---

## Disaggregated Prefill/Decode (P/D Disaggregation)

Prefill and decode are separated onto different hardware pools:

| Phase | Compute pattern | Bottleneck | Ideal hardware |
|-------|----------------|------------|----------------|
| Prefill | Processes all prompt tokens in parallel | Compute (high AI) | High TFLOPS |
| Decode | One token per step, full weight read | HBM bandwidth | High BW |

When co-located, decode's sequential nature wastes the compute capacity needed for fast prefill, and prefill's memory pressure disrupts decode's steady-state throughput. Disaggregation eliminates this interference.

**Measured gain:** 70% higher tokens/sec with P/D disaggregation on GPT-OSS / NVIDIA B200.

KV cache state is transferred from the prefill replica to the assigned decode replica after the prefill phase completes.

---

## KV Cache Management

Beyond routing, llm-d extends the KV cache working set:

**Hierarchical KV offloading** — tiers the KV cache across GPU HBM → CPU DRAM → NVMe. Evicted cache blocks are written to CPU memory or disk and reloaded on demand.

**Measured gain:** 13.9× throughput improvement with hierarchical KV offloading at scale on NVIDIA H100.

---

## Wide Expert Parallelism (MoE)

For MoE models, llm-d can spread expert shards across a large number of accelerators connected via high-bandwidth interconnects, reducing expert routing bottlenecks.

**Measured gain:** 50,000 tokens/sec cluster throughput with wide expert parallelism on 16×16 B200 configuration.

---

## Performance Summary

| Optimization | Benchmark | Gain |
|---|---|---|
| Prefix-cache routing | Llama 3.1 70B, AMD MI300X | 3× throughput, 2× TTFT |
| Predicted-latency scheduling | NVIDIA GPUs | 40% TTFT reduction |
| P/D disaggregation | GPT-OSS, NVIDIA B200 | 70% higher tokens/sec |
| Wide expert parallelism | 16×16 B200 MoE | 50k tokens/sec cluster |
| Hierarchical KV offloading | NVIDIA H100, scale | 13.9× throughput |

---

## Hardware Support

Tested on: NVIDIA H100, B200 · AMD MI300X · Intel XPU · Google TPU

---

## Kubernetes Integration

llm-d is deployed on Kubernetes and integrates with:
- **Gateway API** — standard Kubernetes ingress for the router
- **vLLM** — primary backend model server
- **Kueue** — for admission control and quota management of serving deployments
- **Prometheus / OpenTelemetry** — metrics for SLO-aware autoscaling

---

## See Also

- [Inference routing model](../modeling/inference_routing.md) — analytical model of the KV-cache-aware routing llm-d implements
- [Attention](../workloads/attention.md) — KV cache is the attention mechanism's memory; llm-d optimizes around it
- [LLM inference model](../modeling/llm_inference.md) — prefill/decode latency model
- [Kueue](kueue.md) — cluster-level admission control for llm-d deployments
- [Ray Serve](ray.md) — alternative serving framework without native KV-cache routing
