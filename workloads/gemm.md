# GEMM — General Matrix Multiply

← [Workloads Index](index.md)

GEMM is the computational core of virtually every DNN layer. Understanding its performance is foundational.

---

## Operation Definition

```
C = A × B + bias
A: [M, K]   B: [K, N]   C: [M, N]
FLOPs = 2 × M × K × N   (multiply-accumulate counts as 2)
Bytes (weights, no reuse) = (M×K + K×N + M×N) × dtype_bytes
```

---

## Arithmetic Intensity

For a square GEMM (M=N=K=d), weight-stationary:
```
AI = 2d³ / (3d² × bytes_per_elem) = (2d/3) / bytes_per_elem
```
Larger tiles → higher AI → more compute-bound.

---

## Mapping to LLM Layers

| Layer | M | K | N | Notes |
|-------|---|---|---|-------|
| Linear projection (batch=B, seq=S) | B×S | d_model | d_out | Training prefill |
| Decode step (batch=B) | B | d_model | d_out | Often M ≪ K,N → memory-BW bound |
| Attention QK^T | B×heads | seq | head_dim | seq can be large during prefill |

---

## Performance Regimes

- **M ≫ 1 (training / prefill):** matrix is large, AI high → compute-bound on modern GPUs
- **M = 1 (decode, batch=1):** degenerates to matrix-vector multiply; AI ≈ 0.1–2 → memory-BW bound

---

## Optimization Handles

- **Tensor cores:** require tile sizes aligned to 16×16×16 (or 8×16×16 for FP8)
- **Batching decode requests:** increases M, improves AI, amortizes weight loads
- **Quantization:** smaller dtype → more weights fit in cache, lower BW pressure
- **Continuous batching / in-flight batching:** keeps M high across diverse request lengths

---

## See Also

- [Roofline model](../modeling/roofline.md)
- [LLM inference model](../modeling/llm_inference.md)
- [Operators overview](operators.md)
