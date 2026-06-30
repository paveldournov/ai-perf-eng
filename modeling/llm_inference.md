---
type: Concept
title: LLM Inference Analytical Model
description: Two-phase (prefill/decode) analytical model for LLM inference latency and throughput.
tags: [inference, prefill, decode, latency, kv-cache]
timestamp: 2026-05-30T23:45:33-07:00
---

# LLM Inference Analytical Model

← [Modeling Index](index.md)

---

## Two Phases

LLM inference splits into two distinct compute phases with different bottlenecks:

| Phase | Operation | Bottleneck | Batch sensitivity |
|-------|-----------|------------|-------------------|
| **Prefill** | Process input tokens in parallel | Compute (high AI) | Weak — mostly FLOP-bound |
| **Decode** | Autoregressive generation, 1 token/step | HBM bandwidth | Strong — need large batch to amortize weights |

---

## Prefill Latency Model

```
T_prefill ≈ FLOPs_prefill / (Peak_FLOPS × MFU_prefill)

FLOPs_prefill = 2 × P × S          (P = params, S = seq len; approx for large models)
```

Typical MFU_prefill: 35–55% on H100 for large batch sizes.

---

## Decode Latency Model (per token)

```
T_decode_per_token ≈ max(T_compute, T_memory)

T_memory = Bytes_per_token_step / Peak_HBM_BW
         = (2 × P × dtype_bytes + KV_cache_bytes_per_token) / BW

T_compute = FLOPs_per_token / (Peak_FLOPS × MFU_decode)
          ≈ 2 × P / (Peak_FLOPS × MFU_decode)
```

For small batch sizes, `T_memory >> T_compute` → BW-bound.

**Batch size at which compute = memory:**
```
B_crossover ≈ Peak_FLOPS / (BW × dtype_flops_per_byte)
```
For H100 BF16: ~295 (= ridge point). Practically, full batches of 100–512 are common.

---

## KV Cache Size

```
KV_bytes = 2 × n_layers × 2 × n_kv_heads × head_dim × seq_len × dtype_bytes
           ↑ K and V   ↑ per request
```

For LLaMA-3 70B (BF16, 32 layers, 8 GQA heads, head_dim=128):
```
KV_bytes per token = 2 × 32 × 2 × 8 × 128 × 2 = 262,144 bytes ≈ 256 KB/token
```

---

## Throughput Model

```
Throughput (tokens/s) = batch_size / T_decode_per_token
                      ≈ batch_size × BW / (2 × P × dtype_bytes)   [when BW-bound]
```

---

## Multi-GPU Tensor Parallel Scaling

With tensor parallelism degree `tp`:
```
T_decode_per_token(tp) ≈ (2×P×dtype_bytes) / (tp × BW_per_GPU)
                        + T_allreduce(tp)
```

T_allreduce adds NVLink latency; effective only when model doesn't fit on 1 GPU or when latency SLO demands it.

---

## Model Parameters Reference

| Model | Params | Layers | d_model | n_heads | GQA kv_heads |
|-------|--------|--------|---------|---------|--------------|
| LLaMA-3 8B | 8B | 32 | 4096 | 32 | 8 |
| LLaMA-3 70B | 70B | 80 | 8192 | 64 | 8 |
| LLaMA-3.1 405B | 405B | 126 | 16384 | 128 | 8 |
| GPT-4 (est.) | ~1.8T MoE | — | — | — | — |

---

## Batch Size Analysis (Reiner Pope Framework)

The clearest way to reason about batch size effects uses two independent time contributions:

```
T = max(t_compute, t_mem)

t_compute = B · N_active / FLOPs_hw
t_mem     = (N_total + B · len_ctx · KV_bytes/token) / mem_bw
```

`B` = batch size; `N_active` = active parameters; `N_total` = total parameters (= `N_active` for dense models, > for MoE); `FLOPs_hw` = hardware FLOP/s; `mem_bw` = HBM bandwidth.

**Lower bound on latency:** Even at B=1 you must load all active parameters from HBM, setting a floor on token latency regardless of how small the batch is.

**What can and cannot be amortized over batch size:**
- ✓ Weight fetches: load once, reuse across all B tokens → cost per token falls as 1/B
- ✗ Compute time: each token needs its own matrix multiply
- ✗ KV cache fetch: each token needs its own KV rows

**Optimal batch size for throughput** (ignoring KV cache; equate t_compute = t_mem for weights only):

```
B_optimal = (FLOPs_hw / mem_bw) × (N_total / N_active)
           ≈ 300 / sparsity
```

On modern hardware FLOPs/BW ≈ 300 FLOP/byte. For a dense model: B_optimal ≈ 300. For DeepSeek V3 (32/256 active experts, sparsity = 1/8): B_optimal ≥ 300 × 8 = **2,400**.

---

## HBM Drain Time

The **HBM drain time** — how long it takes to read all model weights from HBM once — sets the minimum time per decoding step:

```
t_drain = HBM_capacity / mem_bw
```

Example (Rubin Ultra): 288 GB / 20 TB/s ≈ **14 ms**. Example (H100 SXM): 80 GB / 3.35 TB/s ≈ **24 ms**.

This is the natural scheduling cadence for a decode server: one "train" of tokens departs every ~t_drain ms. Scheduling more frequently than t_drain is physically impossible (the weights haven't finished loading yet); scheduling less frequently leaves FLOPs idle.

---

## MoE: Rack as Natural Sharding Boundary

For Mixture-of-Experts, the critical collective is **all-to-all** (any token may route to any expert). Within a rack, NVLink provides full-bandwidth all-to-all. Across racks, scale-out bandwidth is ~8× lower.

Consequence: **one rack = natural boundary for one MoE layer**. Spanning an MoE layer across racks creates an 8× bottleneck in the all-to-all and dominates inference latency for large MoE models.

---

## Pipeline Parallelism — Bubbles and KV Cache

**Bubbles during training:** Stages handling later layers sit idle at batch start (no data yet); stages handling early layers sit idle at batch end (data already passed). Overlapping consecutive batches would require applying gradient updates mid-batch, so the bubble is irreducible in standard training.

**KV cache does *not* shrink with more pipeline stages:** Keeping P stages busy requires P micro-batches in-flight simultaneously. The number of concurrent sequences thus scales with P, so KV cache memory pressure does not decrease — the gain from sharding weights is offset by the concurrent-sequence overhead.

**Research iteration cost:** Pipeline parallelism constrains model architecture. Techniques like cross-layer attention (residuals from all previous layers) or interleaved sliding-window/global attention create load imbalances across stages that are difficult to manage. The iteration speed penalty is often the dominant cost.

---

## Total Compute Cost Accounting

**6ND formula** — pre-training FLOPs:
```
FLOPs_pretrain = 6 × N_active × D_pretrain
```
Derivation: 2 FLOPs/param/token for the forward pass (multiply + accumulate). Backward pass costs 2× forward (gradients w.r.t. both weight matrices). Total: 2 + 4 = **6**.

**Full lifecycle cost:**
```
C_total     = C_pretrain + C_RL + C_inference

C_pretrain  = 6 × N_active × D_pretrain
C_RL        = (2–6) × N_active × D_RL × inefficiency_RL
C_inference = 2 × N_active × D_inference × inefficiency_decode
```

`inefficiency_decode` ≈ 3 (decode MFU ≈ 1/3 of prefill MFU due to memory-bandwidth bottleneck per token).

**Cost-optimal training allocation:** If the three costs are substitutable (more pre-training → less RL/inference needed for fixed quality), the optimum satisfies C_pretrain ≈ C_RL ≈ C_inference, giving D_pretrain ≈ 1.5 × D_RL ≈ D_inference.

**Chinchilla over-training for inference:** Chinchilla-optimal is D ≈ 20 × N_active. A 100B-active-parameter model deployed at 50M tokens/sec for 60 days serves ≈ 200T tokens. Setting D_pretrain ≈ D_inference ≈ 200T gives:

```
200T / (20 × 100B) = 100× over Chinchilla-optimal
```

Frontier models are ~100× over Chinchilla-optimal because the inference cost of a deployed model dwarfs the compute-optimal training prescription.

---

## Cost Structure from API Pricing

**Context-length crossover** (compute-bound → KV-bound transition):
```
len_crossover = (FLOPs_hw / mem_bw) × (N_active / KV_bytes_per_token)
              ≈ 300 × N_active / KV_bytes_per_token
```
Above this length, each additional token costs linearly more (KV-bound); below it, cost is flat (compute-bound). Gemini's ~50% price premium above 200K context with ~100B active params implies **≈ 1.7 KB/token** KV cache.

**Output token premium (3–5×):** Decode must load the full weight set to generate each single token; prefill amortizes the same weight load across the entire prompt. Decode MFU ≈ 1/5 of prefill MFU → output token cost ≈ 5× input token cost.

**Cache hit discount (~10×):** A cache hit serves KV values already in memory — cost is just bandwidth to read them. Recomputing from a fresh prompt costs 2 × N_active FLOPs per token for the prefill, which is far more expensive.

---

## See Also

- [Roofline model](roofline.md)
- [GEMM](../workloads/gemm.md)
- [Memory capacity model](memory_capacity.md) — weights + KV cache sizing; does it fit?
- [Inference routing](inference_routing.md) — KV-aware routing, disaggregated prefill/decode, session affinity
- [Speculative decoding](speculative_decoding.md) — draft-and-verify decode acceleration; exploits the memory-bound decode regime
