---
type: Reference
title: The Art of Transformer Programming
description: Notes on Yaniv Leviathan's writeup of expressing programs directly as transformer weights.
tags: [transformer, programming, interpretability]
resource: https://yanivle.github.io/taotp.html
timestamp: 2026-05-28T23:10:57-07:00
---

# The Art of Transformer Programming

← [Back to Workloads](index.md)

---

**Source:** Yaniv Leviathan, *The Art of Transformer Programming* (2022, updated 2024).
Full PDF and notebook: https://yanivle.github.io/taotp.html

The book manually sets the weights of a production-grade decoder-only Transformer — without training — to execute concrete algorithms: Hello World, Lookup Table, Search, Sort, and Decimal Addition. It treats the Transformer as an esoteric programming language built from linear algebra, and is one of the most thorough treatments of what the architecture can and cannot compute efficiently.

---

## Architecture Used

The implementation throughout is a **Pre-LN decoder-only Transformer** (following Xiong et al., 2020), matching how modern LLMs are structured. The full implementation fits in 30 lines of NumPy:

```python
def self_attn(x, Q, K, V, P, mask):
    queries, keys, values = (np.einsum('nj,hjk->nhk', x, M) for M in (Q, K, V))
    qk = np.einsum('nhk,mhk->nmh', queries, keys) / (Q.shape[-1] ** 0.5)
    attn_weights = softmax(np.where(mask[..., None], qk, float('-inf')), axis=1)
    x = np.einsum('nmh,mhk->nhk', attn_weights, values)
    return np.einsum('nhk,hdk->nd', x, P)

def transformer_layer(x, Q, K, V, P, M1, b1, M2, b2, ln1, ln2, mask):
    x = x + self_attn(layer_norm(x, **ln1), Q, K, V, P, mask)
    return x + np.maximum(layer_norm(x, **ln2) @ M1 + b1, 0) @ M2 + b2
```

Notable choices:
- `dff = 4 * d_model` and `n_heads * d_head = d_model` are **not** enforced — constructions use whatever dimensions are needed
- Input/output embeddings are **tied** by default (following Press & Wolf 2017)
- Efficiency is measured by **parameter count** and **non-zero parameter count** separately

---

## Core Building Blocks

These reusable primitives appear across all the programs:

### Attention Hardening

Soft attention (`softmax(QK^T / √d) V`) introduces noise when reading memory. To simulate hard attention (argmax instead of softmax), multiply Q or K by a large constant N (typically 1,000,000):

```
HardAttention(Q, K, V) ≈ Attention(N·Q, K, V)
```

This makes the softmax distribution near-one-hot for the maximum-scoring token.

### Linearly Shiftable Positional Encodings

To attend to a token exactly δ positions to the left, the positional encoding must be *linearly shiftable*: there exists a linear transformation T_δ such that T_δ(E(j)) = E(j+δ) for all j.

The 3D encoding used throughout is evenly spaced on a circle tilted relative to (1,1,1) so it is **fixed under LayerNorm** (mean=0, variance=1 by construction) and linearly shiftable:

```python
def pos_enc_3d(t):
    x, y = np.cos(t) / (2 ** .5), np.sin(t) / (6 ** .5)
    return np.array([x + y, -x + y, -2 * y]) * (3 ** .5)
```

To attend to relative offset δ, set `Q = T_δ · M` and `K = M` where M projects out the positional subspace.

### Bypassing Layer Normalization

Pre-LN means every sub-layer sees `layer_norm(x)` as input. Constructions avoid distortion by partitioning the hidden state into independent blocks each already satisfying LayerNorm invariants (Σv_i = 0, Σv_i² = d). The key theorem: if v ∈ ℝ^d1 and u ∈ ℝ^d2 are each fixed under their respective LayerNorms, then `concat(v, u)` is fixed under LayerNorm in ℝ^(d1+d2).

In practice the hidden state is split into:
```
[ positional embedding (3D) | token embedding (padded with zeros) ]
```

### Cleanup Techniques

Residual connections accumulate unwanted information. Two strategies:

**Proper cleanup** — attend to self with V=I, P=−I to subtract the residual. Can be merged into an existing self-attention head at no extra cost.

**Messy cleanup** — multiply the desired output u by a large constant N so that `layer_norm(N·u + v) ≈ layer_norm(N·u)` since ‖N·u‖² ≫ ‖v‖².

---

## Programs and Parameter Counts

### Hello World (0-layer Transformer)

Output is a function of absolute position only — no attention needed. A 0-layer Transformer sets `pos_emb[i] = tok_emb[t_{i+1}] * 1_000_000` so the position embedding dominates and the output embedding simply reads the desired token.

| Tokenizer | Total params |
|-----------|-------------|
| Message-specific | 72 |
| Generic ASCII | 807 |

### Lookup Table (1 attention layer, no MLP)

Memorizes n key-value pairs where keys are length-l sequences. The last token attends to the preceding l tokens (using l heads for relative position attending), accumulates a hash of the key, and the output embedding maps hashes to values.

For random lookup tables with n=1,000 entries, key length l=5, |V|=1,000: choosing d ≫ n·l²/|V| = 25 dimensions gives ≈100% accuracy. The construction works when token embeddings are normalized random points and hash matrices are random orthogonal matrices — the signal/noise argument uses the fact that dot products between random d-sphere points have variance ≈1/d.

### Search (2 attention layers)

Finds a prefix sequence of length l within a longer input. Layer 1 reads l tokens into a concatenated representation; Layer 2 performs a key lookup to find the matching position.

Example scale: vocab=1,000, prefix_len=10, block_size=100 → **123,486 total params, only 735 non-embedding non-zero params**.

### Sort (1 attention layer, 6 dimensions)

Uses the "Powerpoints" construction: place token i at position p_i where p_0=0 and p_{i+1} = p_i + 1/2^i. Then q_i = (3·p_{i+1} + p_{i+2})/4 is strictly closer to p_{i+1} than to any other p_j. Setting `Q` to select q_i and `K` to select p_i makes attention point each output step to the next-smallest element.

```python
def powerpoints(n, mx=np.pi):
    return stretch(np.cumsum(np.concatenate((np.zeros(1), 1 / 2 ** np.arange(n - 1)))), mx)
```

Limitation: exponentially shrinking gaps cause floating-point precision to fail beyond ~28 numbers in fp32 on a standard CPU.

| Metric | Value |
|--------|-------|
| Model dimensions (d_model) | 6 |
| Attention heads | 1 |
| Non-zero params (excl. embeddings) | 18 |
| Total non-zero params (n=28) | 166 |

### Decimal Addition (most complex)

Multi-digit decimal addition requires carry propagation — inherently sequential. Key observations:
- Attention alone cannot propagate carry across arbitrary length (attention operates in parallel; carry is a sequential dependency)
- The MLP is used as a universal function approximator for digit arithmetic
- Multiple passes/layers are used to propagate carry iteratively

Input format: `[4, 5, +, 7, 8, =]` → Output: `[1, 2, 3]`
Vocabulary: 12 symbols (digits 0-9, `+`, `=`).

---

## Key Insights for AI Hardware/Systems Work

**What's easy for the optimizer, hard for humans:** Layer normalization, residual connections, and tied embeddings — designed to smooth the optimization landscape — significantly complicate manual programming. Most of the book's techniques (bypass tricks, padding schemes) exist to work *around* these training aids.

**What's hard for both:** Sequential dependencies (carry propagation in addition) require either multiple layers or auxiliary output tokens. The Transformer architecture has no native loop primitive.

**Attention as memory read-out:** The constructions make explicit that each attention head performs one structured memory access: a relative-position-addressed read from a specific offset. Multiple heads = multiple simultaneous reads. This is the effective "instruction set" for transformer programs.

**Extreme parameter efficiency:** The Sort program uses 6 model dimensions and 18 non-zero non-embedding parameters. This demonstrates that the effective information content in a well-designed transformer program can be vastly smaller than the raw parameter count suggests — relevant for interpreting why LLMs can generalize from small fine-tuning sets.

**Pre-LN vs Post-LN:** The author found Pre-LN easier to program manually across nearly all tasks — consistent with its better gradient flow properties in practice.

---

## See Also

- [Attention mechanisms](attention.md) — Flash Attention, MHA variants, complexity
- [GEMM](gemm.md) — the Q, K, V projections are GEMMs
- [Pallas kernels](pallas_kernels.md) — writing custom attention/MLP kernels
- [LLM inference model](../modeling/llm_inference.md) — how these operations map to inference cost
