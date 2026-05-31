# Collective Operations

← [Workloads Index](index.md)

Collectives are the communication operations that make distributed training and inference possible. Every parallelism strategy has a communication cost — understanding the latency and bandwidth formulas tells you whether your parallelism configuration is communication-bound or not.

---

## Why Collectives Matter

A single all-reduce of 140 GB (LLaMA-3 70B gradients in FP32) over 8 GPUs takes:
- NVLink (900 GB/s bidirectional): ~0.3 sec
- InfiniBand HDR (200 Gb/s = 25 GB/s): ~11 sec per step

The difference determines whether your 70B training job takes 1 week or 1 month. Choosing the wrong inter-GPU topology for your parallelism strategy is one of the most common efficiency mistakes.

---

## The Alpha-Beta Model

Every collective has two cost components:

```
t = α + β × m
```

- `α` — latency (startup cost, independent of message size); typically 1–10 μs for NVLink, 2–5 μs for InfiniBand
- `β` — inverse bandwidth (time per byte); `1 / link_bandwidth`
- `m` — message size in bytes

For large messages (training gradients), `β × m` dominates. For small messages (control signals, small activations), `α` dominates — this is why pipeline parallelism works across nodes but TP does not.

---

## All-Reduce

**What it does:** Every GPU starts with a tensor; after the all-reduce, every GPU holds the element-wise sum of all tensors.

**Used for:** Gradient synchronization in data parallelism; output of each TP attention/FFN block.

**Ring all-reduce algorithm:**

Arrange GPUs in a ring. The operation proceeds in two phases:
1. **Reduce-scatter** (`N-1` steps): each GPU sends its shard to the next and accumulates
2. **All-gather** (`N-1` steps): each GPU broadcasts its final shard to all others

```
total_bytes_sent_per_gpu = 2 × (N-1)/N × m ≈ 2m   (for large N)
time = 2 × (N-1)/N × m / link_bw ≈ 2m / link_bw
```

The factor of 2 comes from the two phases. Critically: **all-reduce time is independent of N** (for large N) when measured per GPU — adding more GPUs doesn't make the all-reduce slower, as long as total link bandwidth scales with N.

**Example:** Synchronize 70B FP32 gradients (280 GB) across 8 H100s (NVLink 450 GB/s per direction):
```
t ≈ 2 × 280 GB / 450 GB/s ≈ 1.24 sec
```
At NVLink speeds this is feasible once per training step. Over InfiniBand (25 GB/s): **22 sec** — unacceptable without ZeRO or overlap.

---

## Reduce-Scatter

**What it does:** Each GPU starts with a full tensor. After the op, GPU `i` holds the sum of shard `i` from all GPUs.

**Used for:** ZeRO-2/3 gradient aggregation; first half of a ring all-reduce; TP output projection.

```
time = (N-1)/N × m / link_bw ≈ m / link_bw
```

Half the cost of an all-reduce (only one phase). Each GPU ends up with `1/N` of the data.

---

## All-Gather

**What it does:** GPU `i` starts with shard `i`. After the op, every GPU holds the full tensor.

**Used for:** ZeRO-3 parameter reconstruction before forward pass; second half of ring all-reduce; collecting sequence-parallel activations.

```
time = (N-1)/N × m / link_bw ≈ m / link_bw
```

Same cost as reduce-scatter. Together they form an all-reduce.

---

## All-to-All

**What it does:** Each GPU sends a *different* message to each other GPU. Used for MoE expert routing: each token must be sent to its designated expert GPU.

```
time ≈ (N-1) × m_per_peer / link_bw_per_pair
```

Unlike all-reduce, all-to-all does *not* scale gracefully — every GPU is talking to every other GPU simultaneously, contending for the same links. This is why MoE models are expensive to scale beyond one NVLink domain.

**Example:** 8 experts, 8 GPUs, 1024 tokens, token_dim=4096, BF16:
```
m_per_peer = 1024/8 tokens × 4096 × 2 bytes = 1 MB per GPU pair
t_alltoall ≈ 7 × 1 MB / (450/8 GB/s) ≈ 0.1 ms   [NVLink, intra-node]
t_alltoall ≈ 7 × 1 MB / 3 GB/s ≈ 2.3 ms          [InfiniBand, cross-node]
```
Cross-node is 23× slower — one of the primary reasons MoE scaling is hard.

---

## Point-to-Point (P2P)

**What it does:** One GPU sends data to one other GPU (pipeline parallelism stage boundary).

```
time = α + m / link_bw
```

P2P is used in pipeline parallelism to pass activations between stages. With NVLink, latency is negligible for large activations. The key advantage over all-reduce: **only two GPUs are involved** — pipeline parallelism communication does not scale with N.

---

## Collective Summary

| Collective | Algorithm | Time (bandwidth-bound) | Scale behavior |
|------------|-----------|----------------------|----------------|
| All-reduce | Ring | `2m / bw` | Independent of N |
| Reduce-scatter | Ring | `m / bw` | Independent of N |
| All-gather | Ring | `m / bw` | Independent of N |
| All-to-all | Direct | `(N-1)×m_peer / bw_pair` | Scales with N |
| P2P send | Direct | `m / bw` | 2-GPU only |

---

## Overlap: Hiding Communication Behind Compute

The key to efficient distributed training is **overlapping** communication with compute. If a gradient all-reduce can be launched while the next layer's forward pass is computed, the communication cost disappears from the critical path.

**Data parallel:** As each layer's backward pass finishes, its gradients can be all-reduced immediately while the backward pass continues on earlier layers. Libraries like PyTorch DDP and DeepSpeed implement this automatically.

**Tensor parallel:** All-reduces at each layer boundary are harder to overlap — they block the next layer's forward/backward. This is why TP must be fast (NVLink), not just large.

**Pipeline parallel:** Micro-batches interleave compute and P2P communication, hiding inter-stage latency for bulk of the work.

---

## Bandwidth Numbers by Topology

| Link | Bandwidth | Latency | Use case |
|------|-----------|---------|----------|
| NVLink 4 (H100) | 900 GB/s bidirectional | ~1 μs | TP within node |
| NVLink 5 (B200) | 1,800 GB/s | ~1 μs | TP within node |
| InfiniBand HDR (200 Gb/s) | 25 GB/s | 1–2 μs | DP / PP across nodes |
| InfiniBand NDR (400 Gb/s) | 50 GB/s | 1–2 μs | DP / PP across nodes |
| TPU ICI (v6e) | 800 GB/s | <1 μs | TP/PP within pod |
| DCN (TPU cross-pod) | ~100 GB/s | ~5 μs | DP across pods |

---

## See Also

- [Parallelism strategies](../modeling/parallelism.md) — which collective each parallelism type uses
- [Memory hierarchy](../hardware/memory_hierarchy.md) — NVLink vs HBM BW context
- [LLM inference model](../modeling/llm_inference.md) — TP all-reduce term in decode latency
- [Boardfly interconnect](../hardware/tpu/boardfly.md) — TPU ICI topology detail
