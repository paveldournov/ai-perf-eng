# Reading Guide

This guide gives you a structured path through the knowledge base. Each path builds intuition sequentially — later pages assume the earlier ones.

---

## Path 1 — From Zero: "Why is my GPU slow?"

The foundational mental model. Start here if you're new to AI hardware performance.

1. **[Memory Hierarchy](hardware/memory_hierarchy.md)** — the physical reason why data movement, not arithmetic, usually limits performance
2. **[GEMM](workloads/gemm.md)** — the operation that dominates all DNN compute; understand arithmetic intensity here
3. **[Roofline Model](modeling/roofline.md)** — the single most useful tool: are you compute-bound or memory-bound?
4. **[Roofline Parameters by Chip](hardware/roofline_params.md)** — look up the ridge point for your hardware
5. **[MFU — Model FLOP Utilization](modeling/mfu.md)** — how to measure whether you're actually using the hardware efficiently

**Key insight you should have after this path:** For a large matrix multiply in training, the GPU is compute-bound and you can approach peak TFLOPS. For a single token of LLM decode, you are reading all model weights from HBM to produce one output — that's memory-bound by 100–300×.

---

## Path 2 — LLM Inference: "How fast can it go?"

From hardware limits to predicting tokens/second.

1. **[Attention](workloads/attention.md)** — the key operator; Flash Attention, KV cache, MHA variants
2. **[LLM Inference Model](modeling/llm_inference.md)** — prefill vs. decode, latency formulas, batch size crossover
3. **[Memory Capacity Model](modeling/memory_capacity.md)** — does the model fit? weights + KV cache + activations
4. **[Inference Routing](modeling/inference_routing.md)** — KV-aware routing, disaggregated prefill/decode

**Worked question:** Can you serve LLaMA-3 70B at 50 tokens/sec per user on a single H100?
→ 70B × 2 bytes = 140 GB > 80 GB HBM. Doesn't fit. Need 2 GPUs minimum.
→ With 2× H100: 6.7 TB/s aggregate BW. Throughput ≈ 6.7e12 / 140e9 ≈ **48 tokens/sec at batch=1** (barely).

---

## Path 3 — Scale: "How do I run a model that doesn't fit?"

Distributing training and inference across many GPUs.

1. **[Parallelism Strategies](modeling/parallelism.md)** — DP, TP, PP, SP, EP; which axis to parallelize and when
2. **[Collective Operations](workloads/collective_ops.md)** — all-reduce, all-gather, reduce-scatter; the communication cost of parallelism
3. **[Memory Capacity Model](modeling/memory_capacity.md)** — how parallelism changes what fits on each GPU

**Key insight:** Tensor parallelism splits computation but adds an all-reduce on every layer. At TP=8 within an NVLink domain, the communication is fast enough to be worth it. At TP=16 across nodes, you almost certainly lose more to inter-node latency than you gain from extra parallelism.

---

## Path 4 — Production: "How do I run this at scale on Kubernetes?"

From a working model to a reliable cluster deployment.

1. **[Kueue](scheduling/kueue.md)** — job admission and quota; how jobs queue for GPU resources
2. **[Ray / KubeRay](scheduling/ray.md)** — distributed runtime for training jobs and serving
3. **[llm-d](scheduling/llm_d.md)** — KV-cache-aware inference routing; disaggregated prefill/decode in production
4. **[Pathways](scheduling/pathways.md)** — Google's alternative: single-controller model for TPU workloads

---

## Path 5 — TPU Deep Dive

For those working on Google infrastructure specifically.

1. **[TPU Family Overview](hardware/tpu/index.md)**
2. **[TPU v6e (Trillium)](hardware/tpu/tpu_v6e.md)**
3. **[Boardfly Interconnect](hardware/tpu/boardfly.md)**
4. **[Pallas Kernels](workloads/pallas_kernels.md)** — writing custom GPU/TPU kernels in JAX
5. **[Pathways on Cloud](scheduling/pathways.md)**

---

## Concept Dependency Map

```
Memory Hierarchy
    └── Roofline Model ──────────────────────────────┐
            └── GEMM (arithmetic intensity)           │
                    └── LLM Inference Model ──────────┤
                            └── MFU                   │
                            └── Memory Capacity       │
                            └── Attention / KV Cache  │
                                                      │
Parallelism Strategies ───────────────────────────────┤
    └── Collective Ops (communication cost)           │
                                                      │
Hardware Specs ────────────────────────────────────── Roofline params
    └── TPU / GPU pages
```

---

## Quick Reference: Common Calculations

**Will a model fit on N GPUs?**
→ See [Memory Capacity Model](modeling/memory_capacity.md)

**Is this kernel compute- or bandwidth-bound?**
→ Compute AI, look up ridge point in [Roofline Params](hardware/roofline_params.md), compare

**What's my decode throughput ceiling?**
→ `Throughput ≈ N_gpus × BW_per_gpu / (2 × P × dtype_bytes)` — see [LLM Inference](modeling/llm_inference.md)

**How much does adding a TP dimension cost?**
→ One all-reduce per transformer layer — see [Collective Ops](workloads/collective_ops.md)

**How long does an all-reduce take?**
→ `t ≈ 2 × (N-1)/N × message_bytes / link_BW` — see [Collective Ops](workloads/collective_ops.md)
