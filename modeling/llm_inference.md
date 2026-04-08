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

## See Also

- [Roofline model](roofline.md)
- [GEMM](../workloads/gemm.md)
- [Inference decode workload](../workloads/inference_decode.md)
- [Memory capacity model](memory_capacity.md)
