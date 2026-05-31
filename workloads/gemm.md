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

## Precision Modes and Hardware Paths

Every NVIDIA SM contains two physically distinct compute units.  The dtype of
the operands determines which one XLA/cuBLAS dispatches to:

```
SM
├── CUDA cores (FP32 ALUs)
│     one multiply-add per clock, per core
│     general purpose — any op, any dtype
│
└── Tensor Cores
      hardwired 16×16 block matmul per clock
      only accept reduced-precision inputs (fp16, bf16, int8, fp8)
      ~4× the transistor density of CUDA cores per TFLOP
```

Three practical precision modes span the speed/accuracy trade-off:

| Mode | Input dtype | Accumulation | Mantissa bits | Typical speedup vs fp32 | JAX how-to |
|------|------------|--------------|---------------|------------------------|------------|
| **float32** | fp32 | fp32 | 23 | 1× (baseline) | default |
| **tf32** | fp32 | fp32 | 10 (truncated internally) | ~2× | `jax.config.update("jax_default_matmul_precision", "tensorfloat32")` |
| **bfloat16** | bf16 | fp32 | 7 | ~4× | cast inputs to `jnp.bfloat16` |

**tf32** (introduced on Ampere, RTX 30xx+) is the transparent middle ground:
arrays stay float32 in memory, but XLA silently rounds each mantissa from 23
bits to 10 before feeding the Tensor Core accumulator.  No dtype cast is
needed; one config flag is the only change.  Precision loss is typically
< 0.1% for well-conditioned matmuls.

**bfloat16** goes further: both inputs and the cuBLAS kernel path change
(`cublasGemmEx` with `CUBLAS_COMPUTE_16F`).  The 7-bit mantissa loses ~2
decimal digits of precision, acceptable for most training and inference
workloads.  Maximum throughput.

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
- [JAX GPU efficiency demo](jax_gpu_efficiency_demo.py) — runnable benchmark comparing float32 / tf32 / bfloat16 with Perfetto and xprof traces
