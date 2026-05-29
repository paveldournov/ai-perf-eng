# Attention — Mechanisms, Variants, and Hardware Efficiency

← [Back to Workloads](index.md)

---

## Core Attention Mechanism

Attention maps an input sequence of N tokens (each a d-dimensional embedding) to an output sequence by computing pairwise relevance across all token pairs.

**Standard scaled dot-product attention:**

```
Attention(Q, K, V) = softmax(QK^T / √d) · V
```

where Q, K, V ∈ ℝ^(N×d) are learned linear projections of the input. The `√d` scaling prevents dot products from growing large (and pushing softmax into vanishing-gradient regions) as d increases.

The intermediate `QK^T` is an N×N matrix of attention logits — every token attending to every other token. This is what makes attention:
- **Fully parallel** at training time (no sequential dependency, unlike RNNs)
- **O(N²) in both compute and memory** (the N×N matrix must be materialized)

### Kernel Smoothing Interpretation

Attention is equivalent to Nadaraya-Watson kernel regression:

```
f(x) = Σ_j [ K(x, x_j) / Σ_i K(x, x_i) ] · y_j
```

The softmax-normalized `QK^T` row is the kernel weighting; V provides the "output values." This framing connects attention to classical nonparametric estimation and clarifies why the mechanism generalizes: it learns which past contexts to interpolate from.

---

## Multi-Head Attention (MHA)

Running a single attention head risks different token relationships (syntactic, semantic, coreference) interfering in the same projection space. Multi-head attention runs H independent heads in parallel:

```
MHA(Q, K, V) = Concat(head_1, ..., head_H) · W_O
head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)
```

Each head uses projection matrices W_Q, W_K, W_V ∈ ℝ^(d×d/H), keeping total parameter count similar to a single full-rank head. Heads specialize to different relationship types during training.

### Variants

| Variant | Key Change | Motivation |
|---------|-----------|------------|
| MHA | H full KV heads | Original (Vaswani et al., 2017) |
| MQA (Multi-Query) | Single shared KV head | KV cache memory reduction at decode |
| GQA (Grouped-Query) | G shared KV groups | Balance between MHA quality and MQA efficiency |
| MLA (Multi-head Latent Attention) | Low-rank KV compression | DeepSeek; reduces KV cache footprint |

---

## Flash Attention

**Source:** Tri Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022). [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

### The Problem: Memory Bandwidth Is the Bottleneck

Naive attention materializes the full N×N attention matrix in HBM (GPU high-bandwidth memory). For N=8192, d=128 in fp16, this is 8192² × 2 = **128 MB per head per layer** — and it must be read/written multiple times (once for `QK^T`, once for softmax, once for the `·V` product). At long sequences, HBM bandwidth — not arithmetic throughput — dominates.

The GPU memory hierarchy:
- **SRAM (on-chip, L2/shared memory):** ~20 MB total on A100; ~19 TB/s bandwidth
- **HBM (off-chip DRAM):** 40–80 GB; ~2 TB/s bandwidth

HBM bandwidth is ~10× lower than SRAM bandwidth. Every unnecessary round-trip to HBM is expensive.

### The Solution: Kernel Fusion + Tiling

FlashAttention fuses the three attention steps (QK^T → softmax → ·V) into a single GPU kernel. The algorithm tiles the computation across SRAM blocks so the N×N matrix is **never materialized in HBM**:

1. Load a tile of Q, K, V into SRAM
2. Compute partial attention scores within the tile
3. Accumulate output with a running correction factor
4. Write only the final output O back to HBM

### Online Softmax

The challenge with tiling softmax is that the normalization denominator requires seeing the full row. Online softmax (building on Milakov et al., 2018 and Welford's 1962 numerically stable algorithm) avoids this by maintaining running statistics:

For each new tile, track:
- `m_i`: running max of logits seen so far (for numerical stability / overflow prevention)
- `l_i`: running sum of `exp(x - m_i)` terms

When merging a new tile with the running state:
```
m_new = max(m_old, m_tile)
l_new = exp(m_old - m_new) * l_old + exp(m_tile - m_new) * l_tile
O_new = (exp(m_old - m_new) * l_old * O_old + exp(m_tile - m_new) * P_tile * V_tile) / l_new
```

This produces the exact same result as materializing the full row, but requires only O(N) memory and one HBM pass.

### Impact

| Metric | Naive Attention | FlashAttention v2 |
|--------|----------------|-------------------|
| HBM reads/writes | O(N²) | O(N) |
| GPU memory | O(N²) | O(N) |
| Arithmetic | O(N²d) | O(N²d) (same) |
| Wall-clock speedup | 1× | 2–4× (A100, N=4096) |

FlashAttention does **not** reduce FLOPs — it reduces memory traffic. The speedup comes entirely from fewer HBM round-trips.

---

## State Space Models (Mamba)

**Source:** Gu et al., "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023). [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)

Mamba addresses attention's O(N²) scaling by returning to an RNN-like recurrence that remains parallelizable during training.

### Linear SSM Formulation

```
h_t = A · h_(t-1) + B · x_t
y_t = C · h_t
```

Where h_t is a fixed-size hidden state, A is a state-transition matrix, B/C are input/output projection matrices. Key properties:
- **Fully linear** — no nonlinear activations in the recurrence
- **Parallelizable** when A, B are fixed: the recurrence unrolls as a convolution, computable via FFT in O(N log N)
- **Selective parameterization (Mamba variant):** B, C, and the discretization step Δ are functions of the input x_t, enabling the model to selectively retain or discard context

### Parallel Scan

Even with input-dependent B, C, Mamba exploits the **associativity** of the recurrence to compute all h_t in parallel via a tree-reduction (parallel scan / prefix sum). This restores O(N log N) training parallelism while inference remains O(1) per step (true RNN).

### Attention vs. Mamba Tradeoffs

| | Transformer Attention | Mamba SSM |
|---|---|---|
| Training compute | O(N²d) | O(N d) |
| Training memory | O(N²) naive / O(N) FA | O(N) |
| Inference per step | O(N) KV cache read | O(1) recurrence |
| Context length | Scales poorly past ~100K | Theoretically unbounded |
| In-context recall | Strong | Weaker for exact retrieval |

---

## Performance Summary

| Operation | Arithmetic Intensity | Bound |
|-----------|---------------------|-------|
| Prefill attention (long seq, batched) | ~N/2 FLOP/byte | Compute for large N |
| Decode attention (batch=1) | << 1 FLOP/byte | Memory BW (KV cache reads) |
| FlashAttention kernel (fused) | Higher effective AI | Compute (HBM traffic hidden) |
| Mamba recurrence (decode) | — | Compute (tiny state update) |

---

## See Also

- [GEMM](gemm.md) — attention projections (QKV and output) are GEMMs
- [Pallas kernels](pallas_kernels.md) — writing custom attention kernels on GPU/TPU
- [LLM inference model](../modeling/llm_inference.md) — KV cache sizing, prefill/decode latency
- [Roofline model](../modeling/roofline.md) — classifying attention as compute vs. BW bound

---

## References

- Vaswani et al. (2017). "Attention Is All You Need." *NeurIPS.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Dao et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS.* [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Dao (2023). "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." *ICLR 2024.* [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- Gu & Dao (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- Milakov & Gimelshein (2018). "Online normalizer calculation for softmax." [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)
- George (2026). "Attention From First Principles." *metaworld.me.* https://metaworld.me/blog/public/Attention-From-First-Principles
