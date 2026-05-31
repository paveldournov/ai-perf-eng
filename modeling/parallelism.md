# Parallelism Strategies

← [Modeling Index](index.md)

When a model or its training data no longer fits on one accelerator, you need to split the work across many. There are four independent axes you can parallelize along — and the right combination depends on your model architecture, hardware topology, and whether you're training or serving.

---

## The Four Axes

Think of a large language model as a 3D object: **layers** stacked vertically, **tensors** spread horizontally within each layer, and **batch samples** flowing through in parallel. Each axis of parallelism slices along one of these dimensions.

| Strategy | What it splits | Communication required | Best within |
|----------|---------------|----------------------|-------------|
| Data Parallel (DP) | Batch across GPUs | All-reduce gradients (once/step) | Any topology |
| Tensor Parallel (TP) | Individual weight matrices | All-reduce per layer (forward + backward) | NVLink domain (≤8 GPUs) |
| Pipeline Parallel (PP) | Layer groups across GPUs | P2P activations between stages | Any, but prefers low-latency links |
| Sequence Parallel (SP) | Sequence dimension | All-gather + reduce-scatter | Paired with TP |

---

## Data Parallelism (DP)

**Intuition:** Each GPU has a full copy of the model. The batch is split: GPU 0 sees samples 0–N, GPU 1 sees samples N–2N, etc. At the end of each step, gradients are summed across all GPUs (all-reduce) so every replica stays in sync.

```
GPU 0: [full model] ← batch shard 0 → gradients
GPU 1: [full model] ← batch shard 1 → gradients
                              ↓
                     all-reduce gradients
                              ↓
                     all GPUs update weights identically
```

**Communication:** One all-reduce of the full gradient tensor per step. With `D` GPUs and parameter count `P`:
```
bytes_communicated = 2 × P × dtype_bytes × (D-1)/D   (ring all-reduce)
```

**Limitations:** Every GPU must hold the full model. For a 70B parameter model in BF16: 140 GB — doesn't fit on a single H100.

**ZeRO (Zero Redundancy Optimizer):** Shards optimizer states, gradients, and weights across DP ranks, reducing per-GPU memory from `O(P)` to `O(P/D)`. Three stages:
- ZeRO-1: shard optimizer states only
- ZeRO-2: + shard gradients
- ZeRO-3: + shard parameters (requires all-gather on each forward pass)

---

## Tensor Parallelism (TP)

**Intuition:** Split each weight matrix column-wise or row-wise across GPUs. A linear layer `Y = XW` with `W ∈ [d_model, d_ff]` becomes, with TP=2:

```
GPU 0: W₀ = W[:, :d_ff/2]    Y₀ = X × W₀
GPU 1: W₁ = W[:, d_ff/2:]    Y₁ = X × W₁

Result: Y = concat(Y₀, Y₁)    ← requires all-gather
```

For a two-linear-layer MLP block (like the FFN in a transformer), the split alternates column-wise → row-wise, so the all-gather and reduce-scatter pair cancel to a single all-reduce per MLP block.

**Communication:** One all-reduce per attention block + one all-reduce per FFN block = **2 all-reduces per transformer layer** per forward pass (and the same for backward).

**Why TP is limited to ~8 GPUs:** Each all-reduce adds latency. Within an NVLink domain (one node), NVLink BW ≈ 600–900 GB/s and latency ≈ a few microseconds — fast enough to hide. Across nodes (InfiniBand), latency is 10–100× higher and TP communication starts to dominate. Practical limit: **TP ≤ 8** (one NVLink domain) for most models.

**Memory reduction:** Each GPU holds `1/tp` of each weight matrix. Total memory per GPU for weights: `P × dtype_bytes / tp`.

---

## Pipeline Parallelism (PP)

**Intuition:** Assign contiguous layer groups to different GPUs. GPU 0 handles layers 0–7, GPU 1 handles layers 8–15, etc. Activations from one stage are sent to the next via P2P communication.

```
GPU 0: layers 0–7   → activations → GPU 1
GPU 1: layers 8–15  → activations → GPU 2
GPU 2: layers 16–23 → activations → GPU 3
```

**The bubble problem:** At the start of a batch, GPUs 1–3 are idle waiting for GPU 0 to finish. At the end, GPUs 0–2 are idle waiting for GPU 3 to finish backward. The wasted fraction:

```
bubble_fraction = (p - 1) / (m + p - 1)
```

where `p` = pipeline stages, `m` = micro-batches. With `p=4` and `m=8`: bubble = 3/11 ≈ 27% waste. Use larger `m` to reduce the bubble.

**Memory:** Each GPU holds `1/p` of the model layers. But to keep all stages busy, `p` micro-batches must be in flight simultaneously — KV cache memory scales with `p`, not against it.

**Why PP suits inter-node:** P2P communication is point-to-point (one stage to the next), proportional to activation size (much smaller than weights). Tolerates higher latency than TP.

---

## Sequence Parallelism (SP)

**Intuition:** For very long sequences, the activation tensors (not weights) become the memory bottleneck. SP splits the sequence dimension across GPUs. It is almost always paired with TP — they share the same communication collective.

In Megatron-LM's SP implementation:
- LayerNorm and dropout operate on sequence shards (no communication)
- Self-attention requires an all-gather of the sequence before computing QK^T, and a reduce-scatter after the output projection

**When to use:** Sequence lengths > ~4K tokens where activation memory starts dominating.

---

## Expert Parallelism (EP) — MoE

For Mixture-of-Experts models, experts are distributed across GPUs. The routing step sends each token to its assigned expert GPU via **all-to-all** communication.

```
all-to-all cost = 2 × E × expert_capacity × dtype_bytes / link_BW
```

`E` = number of experts, `expert_capacity` = tokens per expert per step.

**Critical constraint:** All-to-all within NVLink (800–900 GB/s) is efficient. All-to-all across nodes via InfiniBand (~200 GB/s aggregate) creates a severe bottleneck. Consequence: **one rack ≈ one MoE layer shard boundary** — see [LLM Inference Model](llm_inference.md).

---

## Combining Axes: 3D / 4D Parallelism

Production training of large models (100B–1T+ parameters) uses all dimensions simultaneously. A typical configuration for a 405B model across 512 GPUs:

```
TP=8  (within a node, NVLink)
PP=8  (across nodes, one pipeline stage per node)
DP=8  (data-parallel replicas of the TP×PP group)

Total GPUs = 8 × 8 × 8 = 512
```

**Rule of thumb for choosing the combination:**
1. Set `TP` to the NVLink domain size (8 or 16), unless model is small enough for TP=1
2. Set `PP` to spread remaining layers across nodes, keeping `m >> p` to minimize bubble
3. Set `DP` to fill remaining GPUs with data-parallel replicas

---

## Memory Per GPU Summary

| Strategy | Weight memory per GPU | Gradient memory | Optimizer states |
|----------|----------------------|-----------------|-----------------|
| No parallelism | P × B | P × B | 12 × P (Adam, fp32 master) |
| DP only | P × B | P × B / DP | 12 × P / DP (ZeRO-1) |
| TP=t | P × B / t | P × B / t | 12 × P / t |
| PP=p | P × B / p | P × B / p | 12 × P / p |
| TP=t, PP=p, DP=d | P × B / (t×p) | P × B / (t×p×d) | 12 × P / (t×p×d) |

`B` = bytes per parameter (2 for BF16, 4 for FP32). Adam stores: param (4B) + gradient (4B) + momentum (4B) = 12 bytes per parameter.

---

## Worked Example: LLaMA-3 405B on 512 × H100

**Model:** 405B params, BF16 (2 bytes/param)
**Hardware:** 512 × H100-80GB, NVLink within 8-GPU nodes

**Memory requirement (weights only):** 405B × 2 = 810 GB  
**With TP=8, PP=8:** 810 GB / 64 = **~13 GB per GPU** for weights  
Optimizer states (Adam, fp32): 405B × 12 = 4.9 TB / 64 = **~76 GB per GPU**  
Total per GPU: ~89 GB → fits in 80 GB only with ZeRO or activation offloading

**Communication costs:**
- TP all-reduce per layer: 2 × 8192 × 2 bytes / 8 GPUs ≈ 4 MB per all-reduce; at 900 GB/s → **~4 μs**
- PP P2P per micro-batch: 8192 × seq_len × 2 bytes; at 400 GB/s NVLink → fast
- DP gradient all-reduce per step: 810 GB / 8 DP replicas... → use ZeRO-1 to shard optimizer

---

## See Also

- [Collective operations](../workloads/collective_ops.md) — communication primitives and latency formulas
- [Memory capacity model](memory_capacity.md) — whether your configuration fits
- [LLM inference model](llm_inference.md) — how TP affects decode latency
- [Pathways](../scheduling/pathways.md) — single-controller approach to multi-slice dispatch
