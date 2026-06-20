---
type: Concept
title: Mixture-of-Experts (MoE) Efficiency
description: Why sparse MoE models break traditional MFU, and the communication/load-balance/GEMM bottlenecks that bound their hardware utilization.
tags: [moe, sparse, expert-parallelism, mfu, all-to-all, routing, load-balancing]
timestamp: 2026-06-19T00:00:00-07:00
---

# Mixture-of-Experts (MoE) Efficiency

← [Workloads Index](index.md)

Mixture-of-Experts (MoE) scales LLMs to trillion-parameter capacity without scaling compute proportionally: a sparse gating network routes each token to a small subset of feed-forward "experts" instead of the full parameter space. This **decouples total parameter capacity from per-token compute** — the source of MoE's efficiency *and* of nearly every system headache below. The sparse, dynamic computational graph maps poorly onto dense, statically-partitioned hardware.

---

## The MoE Layer: Four Phases

Every MoE layer executes four phases, two of which are communication:

1. **Route** — a gating network projects each token's hidden state to per-expert logits, normalizes (softmax or sigmoid), and picks the top-`k` experts.
2. **Dispatch** — tokens are permuted and sent to the devices hosting their assigned experts via **all-to-all** collectives.
3. **Expert compute** — grouped matrix multiplies (Grouped GEMMs) run the local experts.
4. **Combine** — the inverse all-to-all returns outputs to their origin device and unpermutes to restore sequence order.

Phases 2 and 4 are the dominant cost at scale. See [Collective Operations](collective_ops.md).

---

## Why Traditional MFU Lies About MoE

[MFU](../modeling/mfu.md) uses the global parameter count `N` in `FLOPs ≈ 6·N·B·S`. That assumes **every** parameter is active for every token — true for dense models, false for MoE, where only **1–25%** of parameters fire per token. Plugging global `N` into the MFU numerator inflates "observed FLOPs," **overestimating real hardware utilization by up to ~260%** (MoE-CAP). The inflated number hides the actual bottlenecks — communication stalls and bandwidth saturation — and leads to bad provisioning decisions.

### Sparsity-Aware Metrics (S-MFU / S-MBU)

The **MoE-CAP** framework replaces global-`N` accounting with the *active routed path*:

```
S-MFU = (T_token × S-F_token) / F_peak

S-F_token = F_attn + 2 × (N_router + k_expert × N_expert)
```

- `T_token` — measured throughput (tokens/sec)
- `F_peak` — peak theoretical hardware FLOPS
- `F_attn` — FLOPs of (dense) attention + non-sparse layers
- `N_router` — gating-network cost
- `k_expert` — routing factor (experts per token)
- `N_expert` — parameters in a single expert

**S-MBU** is the bandwidth analog of [MBU](../modeling/mfu.md#mbu--memory-bandwidth-utilization): traditional MBU assumes *all* expert weights are loaded during decode; S-MBU counts only the experts actually activated and credits the overlap when tokens in a batch share an expert (avoiding double-counting redundant weight loads). MoE-CAP reports the profiler overhead as minimal — single-digit-percent TTFT/TPOT penalty — making real-time sparsity-aware tracking practical in production.

> The CAP framing: **Cost, Accuracy, Performance** — you can optimize any two, but the third degrades. Max accuracy + performance demands many parameters on high-cost, high-bandwidth nodes, sacrificing cost.

---

## The Three Walls

System tuning for MoE is iterative — fixing one wall shifts the bottleneck to the next.

| Wall | Physical root cause | Effect on MFU / S-MFU | Key mitigations |
|---|---|---|---|
| **Memory** | Large parameter footprint; inflated transient activations; fragmentation from dynamic batching | OOM at startup; small batches keep compute ops in low-efficiency regimes | Selective recompute; weight/activation offload to CPU DRAM; ZeRO-3 / sharded optimizers |
| **Communication** | All-to-all over slow inter-node links; gradient sync volume | GPUs idle waiting for remote tokens; scales worse as cluster grows | Topology-aware hierarchical all-to-all; compute/comm overlap; FP8/BF16 communication compression |
| **Compute / host overhead** | Routing load imbalance; low-intensity skinny GEMMs | Fast nodes wait on slow; poor SM/tensor-core occupancy | Auxiliary-loss-free routing; Grouped GEMM kernels; GPU-initiated (device-side) communication |

---

## Expert Parallelism and the All-to-All Wall

**Expert Parallelism (EP)** shards experts across GPUs. It lets a model exceed single-node memory, but converts the intra-layer pattern from regular data-parallel collectives (all-reduce / reduce-scatter) into dynamic, high-frequency **all-to-all**. Per-GPU dispatch volume for EP size `P_EP`:

```
V = (P_EP − 1)/P_EP × N_tokens × k_expert × H_dim
```

and it is **bidirectional** (dispatch + combine). The problem is interconnect non-uniformity: intra-node NVLink (~900 GB/s on Hopper) vs. inter-node InfiniBand / Ethernet (~50–100 GB/s). As EP spans more nodes, more peers fall on the slow path. Reported profiling on H800 clusters: for fine-grained MoE (`k=4`), all-to-all grows from ~22% of layer latency on one node to ~78% across 8 nodes — GPUs stall waiting for tokens to arrive. See [Parallelism](../modeling/parallelism.md).

**Mitigation pattern:** confine the high-frequency collectives (TP, EP) to the intra-node NVLink domain; scale CP/PP across nodes first (their traffic is regular and easy to overlap). Couple **device-initiated communication** (DeepEP, Hybrid-EP, NCCL EP — kernel-level RDMA via IBGDA/TMA that bypasses the host-CPU control loop) with asynchronous pipeline schedules (DualPipe, AF-Pipe) so token transfers overlap independent attention/GEMM compute.

---

## Load Imbalance and Routing Collapse

Routing is content-dependent and dynamic, which risks **routing collapse**: the gate concentrates on a few "popular" experts. Consequences:

- **Compute hotspots** — GPUs with popular experts are overloaded while others idle; synchronous training is bounded by the slowest GPU.
- **OOM during warmup** — in the first ~100–300 steps the router is untrained and unstable, spiking tokens onto random experts and saturating their activation memory before the router self-corrects.

### Auxiliary-Loss-Free Load Balancing (DeepSeek-V3)

The classic fix — an auxiliary balancing loss with coefficient `α` — is a double-edged sword: too small and routing collapses; too large and the balancing gradient interferes with the language-modeling objective and hurts perplexity. DeepSeek-V3 instead adds a **non-differentiable per-expert bias** `b_i` to the top-k selection only:

```
s_{i,t} = Sigmoid(u_t · e_i)                          # affinity score
Selected = TopK({ s_{j,t} + b_j }, K_r)               # bias steers selection
b_i ← b_i + u · sign(deviation_i)                     # nudged after each step
```

The bias steers *which* experts are picked (decreasing `b_i` for over-loaded experts, increasing it for idle ones) but the **gating weights use the unbiased `s_{i,t}`**, so it never distorts the gradient. DeepSeek-V3 reports near-uniform allocation with no accuracy penalty. It pairs this with **Multi-Token Prediction (MTP)** — auxiliary prediction heads sharing the embedding/output, discardable at inference or reusable for speculative decoding.

---

## Skinny GEMMs: the Compute-Side Tax

Tensor cores hit peak only on large, structurally balanced matrices. Fine-grained MoE (DeepSeek-MoE, Qwen-MoE) splits the FFN into many narrow experts to improve representational capacity — but this forces **"tall-and-skinny" GEMMs** (small token batch `M`, narrow `K`/`N`) that fail to saturate warp schedulers and run **memory-bound at <10% of peak FLOPS**. [Grouped GEMM](gemm.md) kernels pack multiple local experts into one launch to recover intensity. There is a real architecture↔systems tension here: finer experts help convergence but hurt kernel efficiency.

---

## Systems Landscape (Selected)

| System | Lead | Core idea |
|---|---|---|
| **Megatron-Core MoE** | NVIDIA + collaborators | *Parallel Folding* — decouple attention sharding (`G_A = TP_a·CP·DP·PP`) from MoE sharding (`G_M = TP_m·EP·DP·PP`); set `TP_m=1` to maximize local GEMM size |
| **DeepSeek-V3 / DeepEP** | DeepSeek-AI | Aux-loss-free balancing, DualPipe near-zero-bubble overlap, GPU-initiated RDMA |
| **MegaScale-MoE** | ByteDance | Operator-level compute/comm overlap; selective rematerialization; FP8/BF16 comm compression |
| **Piper** | ORNL / Frontier | Automated 5-D parallel mapping; Dragonfly-topology-aware hierarchical all-to-all; live expert migration |
| **DisagMoE** | UC Berkeley + MSR | Spatially disaggregate attention vs. FFN onto separate device pools; AF-Pipe schedule |
| **NCCL EP** | NVIDIA | Unified `ncclEpDispatch`/`ncclEpCombine` API; low-latency (decode) and high-throughput (prefill/train) modes |

Inference-side: **Klotski** (expert-aware multi-batch prefetch overlapping PCIe), **TD-Pipe** (prefill/decode temporal disaggregation), **ReMoE** (router fine-tuning for cache locality), **RTP-LLM** (production KV-cache + adaptive quant).

---

## Empirical MFU (reported)

| Model | Total / Active | Hardware | Stack | MFU |
|---|---|---|---|---|
| Mixtral-8x22B | 141B / 39B | 128× H100 | FSDP baseline | 4.3% |
| Mixtral-8x22B | 141B / 39B | 128× H100 | Megatron-Core + Parallel Folding | **49.3%** (2.1× over FSDP+EP) |
| Qwen2-57B-A14B | 57B / 14B | 64× H100 | Megatron-Core + Parallel Folding | **39.0%** |
| Internal-352B | 352B | 1,440× H800 | Megatron-LM baseline | 19.3% (747k tok/s) |
| Internal-352B | 352B | 1,440× H800 | MegaScale-MoE | **36.3%** (1.41M tok/s, 1.88×) |
| DeepSeek-style | 545B | 1,024× MI250X | X-MoE → Piper | 5.0% → **15–17.5%** (3–3.5×) |
| DeepSeek-V3 | 671B / 37B | 2,048× H800 | Megatron-Core + DeepEP + DualPipe | **~52%** sustained |

> These figures are as reported by the respective papers/report; the 352B run is an unattributed "internal" model. Treat the FSDP baselines as worst-case (no EP) rather than tuned references.

---

## See Also

- [MFU](../modeling/mfu.md) — the metric MoE breaks; S-MFU corrects it
- [Collective Operations](collective_ops.md) — all-to-all cost model
- [Parallelism](../modeling/parallelism.md) — where EP sits among TP/PP/DP/CP
- [GEMM](gemm.md) — why skinny/grouped GEMMs matter
- [References](../references/index.md) — MoE-CAP, DeepSeek-V3, MegaScale-MoE citations
