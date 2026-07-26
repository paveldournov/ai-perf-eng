---
type: Concept
title: Attention — Mechanisms, Variants, and Hardware Efficiency
description: Core attention, its variants (MQA/GQA/MLA), and the hardware efficiency of Flash Attention.
tags: [attention, flash-attention, transformer, mqa, gqa]
timestamp: 2026-05-31T22:58:33-07:00
---

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

## Kernel Implementations

The algorithm described above maps to real code in a handful of canonical implementations. Reading them alongside the theory is the fastest way to internalise what "tiling" and "online softmax" actually mean at the hardware level.

### GPU — Triton (reference implementation)

**Source:** Triton tutorial `06-fused-attention` — [triton-lang.org](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)

The Triton tutorial is the clearest pedagogical implementation of Flash Attention v2. It implements the full forward + backward pass in ~300 lines of Python.

Key structural decisions:
- `Q` is loaded once into SRAM and stays resident for the entire inner loop over K/V blocks — this is the defining IO-saving move
- Online softmax state `(m_i, l_i)` is maintained in registers across the inner loop; the output accumulator `acc` is rescaled each time `m_i` shifts
- Causal masking is handled with a two-stage loop: off-diagonal blocks (no masking needed, cheaper) then the diagonal block (masking applied)
- Separate backward kernels for `dK/dV` and `dQ` to avoid atomic writes

```python
# Pseudocode reflecting the Triton kernel structure
for start_n in range(0, seq_len, BLOCK_N):   # iterate K/V tiles
    k = load(K[start_n : start_n + BLOCK_N]) # load K tile from HBM
    v = load(V[start_n : start_n + BLOCK_N]) # load V tile from HBM

    qk = dot(q, k.T) * scale                 # q is already in SRAM
    if causal: apply_mask(qk)

    m_new = max(m_i, row_max(qk))
    l_new = exp(m_i - m_new) * l_i + sum(exp(qk - m_new))
    acc = (exp(m_i - m_new) * acc + exp(qk - m_new) @ v)
    m_i, l_i = m_new, l_new

o = acc / l_i   # final normalisation — only one divide per output element
```

Measured throughput (batch=4, H=32): ~160 TFLOPS at seq_len=16K in FP16 on H100. Causal masking achieves ~2× speedup over full attention due to skipping the lower triangle.

Hardware-specific tuning in the tutorial:
- `BLOCK_M=128, BLOCK_N=64` for Hopper; smaller blocks for older GPUs
- `warp_specialize=True` on Hopper/Blackwell for overlapping data loads and compute
- FP8 accumulation path for Blackwell

### GPU — JAX / Pallas

**Source:** `jax/experimental/pallas/ops/gpu/attention.py` — [github.com/google/jax](https://github.com/google/jax/blob/main/jax/experimental/pallas/ops/gpu/attention.py)

The official JAX implementation uses Pallas (see [Pallas kernels](pallas_kernels.md)) to express the same algorithm as the Triton version but within the JAX ecosystem, composing with `jit`, `vmap`, and `grad`.

Same online softmax approach: running `(m_i, l_i)` in registers, unscaled accumulator corrected at each tile boundary. Key quote from the source:

> "We keep an unscaled version of o during the scan over seq_len. Scaling it by the last l_i gives us the correct final output."

Block sizes are configured via a `BlockSizes` dataclass with separate `block_q / block_k` for forward and backward passes, allowing hardware-specific tuning without changing the algorithm.

### TPU — `jax.nn.dot_product_attention`

On TPU, the recommended path is **not** to write a custom kernel. JAX's built-in `jax.nn.dot_product_attention` emits a Flash Attention-equivalent fused XLA op when running on TPU:

```python
import jax
import jax.numpy as jnp

# Inputs: [batch, seq, heads, head_dim]
out = jax.nn.dot_product_attention(query, key, value, is_causal=True)
```

XLA tiles the computation across TPU VMEM (on-chip scratchpad) automatically, avoiding HBM materialisation. The TPU's 2D systolic array (`MXU`) processes the `QK^T` and `AV` matrix multiplies; the scalar unit handles the softmax reduction.

For custom TPU attention kernels (e.g., sliding-window attention, cross-attention with custom masking), use Pallas with explicit VMEM tiling — see [Pallas kernels](pallas_kernels.md) and the [JAX Pallas TPU matrix multiply tutorial](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html) for the tiling pattern.

### Official CUDA / C++ (production)

**Source:** Dao-AILab/flash-attention — [github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

The original production implementation used by most frameworks (PyTorch `F.scaled_dot_product_attention` dispatches here on CUDA). Written in CUDA with hand-tuned CUTLASS-style register layouts. Harder to read than the Triton version but faster in practice.

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

## Gated Linear Attention & Hybrid Backbones

**Gated Linear Attention (GLA)** is the linear-attention cousin of the SSM recurrence
above: it replaces the softmax `QK^T` with a gated, associative recurrence so that, like
Mamba, training uses a **chunk-wise parallel scan** while decode is an **O(1)-per-step**
update over a fixed-size state. The chunk-wise decomposition is what makes linear-attention
*prefill* parallelizable on matmul-oriented hardware (GPU Tensor Cores, TPU MXU).

**Hybrid backbones** interleave a few full-context layers with many linear layers to get
near-linear cost while preserving exact recall where it matters:

| Layer type | State at decode | Role |
|---|---|---|
| [MLA](#multi-head-attention-mha) (full attention) | Token-indexed **KV cache**, grows with context | Exact long-range recall |
| GLA / linear (majority of layers) | Fixed **request-indexed** recurrent state, O(1) | Cheap bulk context mixing |

This split has a direct systems consequence: the two layer families need **different memory
pools** — a growing token-indexed KV cache vs. a fixed per-request recurrent buffer. The
[Ling-2.6 serving case study](moe.md#serving-case-study-fused-moe-v2-on-tpu-ling-26-1t) uses
exactly this (MLA + 70 GLA layers) and reports that the *unoptimized GLA prefill kernel*
becomes the dominant prefill cost once the MoE layers are tuned.

---

## Tree-Causal Masking

A structured generalization of the causal mask used in **tree-based
[speculative decoding](../modeling/speculative_decoding.md)**. Instead of one linear
sequence, several candidate continuations branch from shared prefixes and are packed into a
single sequence. The mask lets each tree node attend to the **original prefix and its ancestors
only** — never to sibling branches or descendants:

```
mask[i, j] = 1  iff  j is the prefix  OR  j is an ancestor of i in the draft tree
```

This is what makes a whole candidate tree verifiable in **one** forward pass while every branch
keeps correct autoregressive dependencies (sibling branches must not leak into each other).
Production kernels combine it with paged attention so tree nodes share KV-cache pages with their
common prefix. See [Speculative Decoding → tree attention](../modeling/speculative_decoding.md#tree-drafting-and-tree-attention).

---

## Performance Summary

| Operation | Arithmetic Intensity | Bound |
|-----------|---------------------|-------|
| Prefill attention (long seq, batched) | ~N/2 FLOP/byte | Compute for large N |
| Decode attention (batch=1) | << 1 FLOP/byte | Memory BW (KV cache reads) |
| FlashAttention kernel (fused) | Higher effective AI | Compute (HBM traffic hidden) |
| Mamba recurrence (decode) | — | Compute (tiny state update) |
| GLA / linear attention (decode) | — | Compute (fixed-size state update) |

---

## See Also

- [GEMM](gemm.md) — attention projections (QKV and output) are GEMMs
- [Pallas kernels](pallas_kernels.md) — writing custom attention kernels on GPU/TPU
- [LLM inference model](../modeling/llm_inference.md) — KV cache sizing, prefill/decode latency
- [Speculative decoding](../modeling/speculative_decoding.md) — tree-causal masking for draft verification
- [Roofline model](../modeling/roofline.md) — classifying attention as compute vs. BW bound
- [Kernel optimization](inference/kernel-optimization.md#flashattention) — FlashAttention v1–v4 evolution and GPU memory hierarchy

---

## References

- Vaswani et al. (2017). "Attention Is All You Need." *NeurIPS.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Dao et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS.* [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Dao (2023). "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." *ICLR 2024.* [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- Gu & Dao (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- Milakov & Gimelshein (2018). "Online normalizer calculation for softmax." [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)
- George (2026). "Attention From First Principles." *metaworld.me.* https://metaworld.me/blog/public/Attention-From-First-Principles
