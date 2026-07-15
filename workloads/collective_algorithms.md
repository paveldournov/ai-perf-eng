---
type: Concept
title: Collectives on TPU & GPU Clusters — Topology & Algorithms
description: How ring/tree collective algorithms map onto TPU torus and GPU fat-tree topologies, with the latency-vs-bandwidth threshold and SHARP in-network reduction.
tags: [collectives, nccl, topology, torus, fat-tree, ring, tree, sharp, ici, nvlink, tpu, gpu]
resource: https://www.aleksagordic.com/blog/collective-operations
timestamp: 2026-07-14T00:00:00-07:00
---

# Collectives on TPU & GPU Clusters — Topology & Algorithms

← [Workloads Index](index.md)

Where [Collective Operations](collective_ops.md) gives the alpha-beta cost formulas,
this page covers how those collectives actually run on real cluster **topologies** —
the TPU torus and the GPU fat-tree — and which **algorithm** (ring vs tree) wins in
which regime.

Source: [Inside TPU & GPU Clusters: Collective Communication](https://www.aleksagordic.com/blog/collective-operations)
(Aleksa Gordic, 2026)

---

## The Four Primitives

The article builds everything from four collectives:

| Primitive | Effect | Drives |
|-----------|--------|--------|
| **All-gather** | shards → full copy on every rank | TP / FSDP forward, ZeRO-3 |
| **Reduce-scatter** | reduce, each rank keeps one shard | FSDP backward, ZeRO |
| **All-reduce** | reduce-scatter + all-gather (**2×** cost) | DP gradient sync |
| **All-to-all** | every rank sends a distinct message to every other | MoE / expert-parallel routing |

All-reduce is not primitive — it is a reduce-scatter followed by an all-gather, which
is why it costs **twice** either half.

---

## Ring vs Tree

- **Ring** — the natural choice for **large messages**. Ranks form a logical ring; each
  step forwards a chunk to the neighbor while receiving another, so links stay saturated.
  Ring pipelines well, giving high *effective* bandwidth despite `N-1` steps.
  - **Bidirectional ring** uses full-duplex links in both directions for ~2× speedup.
  - **Unidirectional ring** is the half-duplex fallback.
  - **Chain / path** variant is used when there is no wraparound link to close the ring.
- **Tree** — better for **latency-bound, small messages**: only `log₂ N` steps instead
  of `N-1`. The step count, not bandwidth, is what matters when messages are tiny.

Rule of thumb: **rings for big tensors, trees for small ones.**

---

## TPU Cluster: the Torus

TPU chips connect directly to nearest neighbors via **ICI (Inter-Chip Interconnect)**,
forming a torus:

| Generation | Torus | Neighbors |
|------------|-------|-----------|
| v2 / v3 / v5e / v6e | 2D | 4 |
| v4p / v5p / v7x / v8t | 3D | 6 |

Bandwidth tiers, fastest to slowest: **ICI** (e.g. 45 GB/s unidirectional on v5e) →
**PCIe** → **DCN** (data-center network, cross-pod).

**Wraparound matters.** A full torus has wraparound links that close each ring; a plain
mesh (no wraparound) does not. Losing wraparound along an axis can **roughly double** the
time of ring collectives along that axis. Practical consequence: for
communication-heavy jobs, **request slice shapes that preserve the torus** rather than
carving out a mesh.

### The latency ↔ bandwidth threshold

> Saturating 45 GB/s unidirectional ICI, how much data flows through one link in 1 μs?
> **~45 KB.**

Below ~45 KB per message you are in the **latency-bound** regime (α dominates, favor
trees); above it you are **bandwidth-bound** (β·m dominates, favor rings). This single
number tells you which side of the alpha-beta model you're on for a given ICI.

---

## GPU Cluster: the Fat-Tree

A DGX H100 SuperPod is a hierarchical, **full-bisection** fat-tree:

- **Node (intra-node):** 8 GPUs fully connected via **NVSwitch** (all-to-all).
- **Scalable Unit (SU):** 32 nodes joined by InfiniBand **leaf** switches.
- **Cluster:** multiple SUs joined by **spine** switches.

Full bisection (non-oversubscribed) means any partition of N nodes gets full injection
bandwidth across the partition boundary — the fabric doesn't become the bottleneck when
you split the job.

### Hierarchical cost

Collectives run in stages (intra-node NVLink, then inter-node IB), so the wall-clock is
gated by the slower stage:

```
T_total ≈ max( D / BW_gpu , D / BW_node )
```

### SHARP: in-network reduction

InfiniBand **SHARP** performs the reduction inside the switch fabric. Theoretically ~2×,
but in practice only **~1.3× (~30%)** on H100 — pipelining inefficiency, multicast
overhead, and SM/HBM bottlenecks eat the rest. A concrete case of theory-vs-practice
gaps that make **microbenchmarking your actual deployment** essential.

---

## Ring Cost Models (bandwidth-bound)

For message size `M` across `N` chips on a bidirectional ring:

| Collective | Time |
|------------|------|
| All-gather | `M(N-1)/BW` |
| Reduce-scatter | `M(N-1)/BW` |
| All-reduce | `2M(N-1)/BW` |
| All-to-all | `M(N-1)/BW` |

(These are the `(N-1)/N`-per-rank formulas of [Collective Operations](collective_ops.md)
scaled to total message movement.)

---

## Mapping to Parallelism

| Parallelism | Collective |
|-------------|------------|
| **Data parallel** | all-reduce (gradient sync during backprop) |
| **Tensor / model parallel, FSDP** | all-gather + reduce-scatter (fwd/bwd) |
| **Expert parallel (MoE)** | all-to-all (token routing) |

---

## Takeaways

- All-reduce = reduce-scatter + all-gather → costs 2× either half.
- **Rings** dominate large-message collectives on both TPU and GPU; **trees** win when
  latency dominates (small messages, `log₂ N` steps).
- **Topology is destiny:** preserve the TPU torus (wraparound ≈ 2× on ring axes); GPU
  fat-trees give full bisection but stage across NVLink → IB.
- The **~45 KB / μs** ICI figure is a quick test for latency- vs bandwidth-bound.
- **SHARP** and other theoretical wins under-deliver in practice — always microbenchmark.

---

## See Also

- [Collective Operations](collective_ops.md) — alpha-beta model and per-primitive formulas
- [MoE (Mixture of Experts)](moe.md) — the all-to-all-heavy workload
- [Parallelism strategies](../modeling/parallelism.md) — which collective each strategy uses
- [Boardfly interconnect](../hardware/tpu/boardfly.md) — TPU ICI topology detail
