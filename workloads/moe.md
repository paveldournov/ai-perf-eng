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

The bias steers *which* experts are picked (decreasing `b_i` for over-loaded experts, increasing it for idle ones) but the **gating weights use the unbiased `s_{i,t}`**, so it never distorts the gradient. DeepSeek-V3 reports near-uniform allocation with no accuracy penalty. It pairs this with **Multi-Token Prediction (MTP)** — auxiliary prediction heads sharing the embedding/output, discardable at inference or reusable as drafters for [speculative decoding](../modeling/speculative_decoding.md).

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

Inference-side: **Klotski** (expert-aware multi-batch prefetch overlapping PCIe), **TD-Pipe** (prefill/decode temporal disaggregation), **ReMoE** (router fine-tuning for cache locality), **RTP-LLM** (production KV-cache + adaptive quant), **SGLang-JAX / Fused MoE V2** (TPU; comm/compute-overlapped fp8 kernel — see [case study below](#serving-case-study-fused-moe-v2-on-tpu-ling-26-1t)).

---

## Serving case study: Fused MoE V2 on TPU (Ling-2.6-1T)

A concrete instance of "hide the all-to-all behind the routed compute," on TPU rather
than GPU. LMSYS optimized inference of **Ling-2.6-1T** (inclusionAI) with **SGLang-JAX**
on [TPU v7x](../hardware/tpu/tpu_v7x.md).

**Model.** 1T total / 63B activated per token; 256 routed experts, top-8 + 1 shared
expert; per-channel **fp8** expert weights; hybrid **[MLA](attention.md#multi-head-attention-mha)
+ Gated Linear Attention** backbone (70 GLA layers). Mesh: `ep=32`, `tp=32`, `dp=8` on
16 chips (2×2×4 ICI torus).

**The kernel — "Fused MoE V2" (Pallas).** A single fused kernel that overlaps token
routing, fp8 weight prefetch, and the fp8 reorder behind the routed-GEMM window, using
**VMEM-resident token/accumulator buffers** and **weight double-buffering** (see
[Pallas — overlapping comm and compute](pallas_kernels.md#overlapping-communication-and-compute)).
Reported: prefill latency **5.16 → 2.42 ms (−53%)**, decode **0.249 → 0.211 ms (−15%)**.

**Supporting optimizations:**

- **Post-reduction per-channel scaling** (`direct_scaled_dot`) — apply fp8 scales *after*
  the K-reduction instead of slicing along K, avoiding a serialized rescale.
- **fp8 activation quantization** — shrink the scattered all-to-all payload from bf16 to
  fp8, cutting the scatter stage **1.39 → 0.65 ms**. Lowers the [communication wall](#the-three-walls)
  directly by halving dispatch bytes.
- **In-kernel shared expert** — schedule the always-on shared expert *inside* the scatter
  phase, adding only **+2.7%** to the critical path instead of a separate launch.
- **Hybrid memory pools** — token-indexed KV cache for MLA layers vs. fixed
  request-indexed recurrent state for the 70 GLA layers (linear-attention state is O(1)/request).
- **Single-controller data parallelism** — split the mesh into DP groups with constrained
  TP so grouped-RMSNorm stays chip-local (no cross-chip reduction).

**Lower bounds (their cost model, per layer):** scatter/gather ≥0.67 ms, weight movement
≥0.44 ms, routed compute ≥0.36 ms — i.e. communication, not GEMM, sets the floor, exactly
the [all-to-all wall](#expert-parallelism-and-the-all-to-all-wall).

**Results.** Prefill throughput **+24.8%**, decode **+18.5–35.3%** (V1→V2); end-to-end
**1.29–1.77× decode throughput vs. H200×16**; AIME-2026 86.7% with zero request errors.
**Remaining bottleneck:** the GLA *prefill* kernel is not yet V2-optimized and now dominates
prefill cost — fixing one wall shifted the bottleneck, as expected.

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
- [Pallas kernels](pallas_kernels.md) — VMEM tiling and comm/compute overlap behind the Fused MoE V2 kernel
- [TPU v7x](../hardware/tpu/tpu_v7x.md) — hardware target of the Ling-2.6 serving case study
- [References](../references/index.md) — MoE-CAP, DeepSeek-V3, MegaScale-MoE, SGLang-JAX citations
