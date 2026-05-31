# Memory Capacity Model

← [Modeling Index](index.md)

Before asking "how fast will it run?", ask "does it fit?" Memory capacity is the hard constraint that determines your minimum GPU count and shapes every parallelism decision.

---

## What Occupies GPU Memory

GPU memory during LLM training or inference holds four distinct things:

```
Total GPU memory = Weights + Optimizer States + Activations + KV Cache
```

Only some of these are present simultaneously depending on the workload:

| Component | Training | Inference (prefill) | Inference (decode) |
|-----------|----------|--------------------|--------------------|
| Weights | ✓ | ✓ | ✓ |
| Optimizer states | ✓ | ✗ | ✗ |
| Activations | ✓ (or checkpointed) | ✓ (transient) | ✓ (transient) |
| KV cache | ✗ | ✓ (being written) | ✓ (read each step) |

---

## 1. Weights

```
weight_bytes = N_params × bytes_per_param
```

| Dtype | bytes_per_param | LLaMA-3 8B | LLaMA-3 70B | LLaMA-3 405B |
|-------|----------------|------------|-------------|--------------|
| FP32 | 4 | 32 GB | 280 GB | 1,620 GB |
| BF16 | 2 | 16 GB | 140 GB | 810 GB |
| INT8 | 1 | 8 GB | 70 GB | 405 GB |
| FP4 / NF4 | 0.5 | 4 GB | 35 GB | 202 GB |

**Rule of thumb:** BF16 weights ≈ `2 × N_billions` GB. A 70B model needs ~140 GB just for weights.

---

## 2. Optimizer States (Training Only)

Adam optimizer stores: fp32 master weights + fp32 momentum + fp32 variance = **12 bytes/param**.

Mixed-precision training (BF16 forward, FP32 optimizer):
```
optimizer_bytes = 12 × N_params   (4B master weight + 4B momentum + 4B variance)
gradient_bytes  =  2 × N_params   (BF16 gradients)
-------------------------------------------------
Total (excl. activations) = 16 × N_params
```

For LLaMA-3 70B: 16 × 70e9 = **1,120 GB** — requires at least 14 × H100-80GB just for parameters + optimizer.

With ZeRO-3 across `D` GPUs: `16 × N / D` bytes per GPU.

---

## 3. KV Cache (Inference Only)

The KV cache grows with each token generated and with batch size. This is the key variable you control via serving configuration.

```
kv_bytes = 2          (K and V)
         × n_layers
         × 2          (per-token K and V vectors)
         × n_kv_heads
         × head_dim
         × seq_len    (current context length)
         × batch_size
         × dtype_bytes
```

Simplified per-token, per-sequence cost:
```
kv_bytes_per_token = 2 × n_layers × 2 × n_kv_heads × head_dim × dtype_bytes
```

**Reference values (BF16):**

| Model | n_layers | n_kv_heads | head_dim | KV bytes/token |
|-------|----------|------------|----------|----------------|
| LLaMA-3 8B | 32 | 8 | 128 | 32 × 2 × 8 × 128 × 2 = 131 KB |
| LLaMA-3 70B | 80 | 8 | 128 | 80 × 2 × 8 × 128 × 2 = 328 KB |
| LLaMA-3 405B | 126 | 8 | 128 | 126 × 2 × 8 × 128 × 2 = 516 KB |

**Total KV cache for a serving batch:**
```
kv_total = kv_bytes_per_token × seq_len × batch_size
```

Example: LLaMA-3 70B, seq_len=8192, batch=32:
```
328 KB/token × 8192 tokens × 32 sequences = 86 GB
```
That's more than the weights on a single GPU.

---

## 4. Activations (Training)

During the forward pass, activations must be retained for the backward pass. Without any optimization:

```
activation_bytes_per_layer ≈ B × S × d_model × bytes   (dominant term)
```

For a 70B model, B=1, S=4096, d_model=8192, BF16: ≈ 64 MB per layer × 80 layers = **5 GB** — manageable.

But with large batches or long sequences this grows linearly:

```
activation_bytes ≈ B × S × d_model × n_layers × bytes × activation_factor
```

`activation_factor` ≈ 10–34 depending on what is stored per layer (just the residual, or also intermediate FFN tensors).

**Gradient checkpointing:** Discard activations during forward pass; recompute them during backward. Trades ~33% more compute for a `√n_layers` memory reduction. Standard practice for large models.

---

## Worked Examples

### "Will LLaMA-3 70B fit for inference on 2× H100-80GB?"

```
Weights (BF16):     140 GB  / 2 GPUs = 70 GB each
KV cache (batch=8, seq=4096):
  328 KB/token × 4096 × 8 = 10.7 GB / 2 GPUs = 5.4 GB each
Activations (transient):  ~1-2 GB

Total per GPU: ~77 GB  ✓  fits in 80 GB (barely)
```

Batch=8 at seq_len=4096 is feasible. Batch=32 at seq_len=4096 would require ~43 GB for KV cache → doesn't fit.

### "How many GPUs to train LLaMA-3 405B with full optimizer state?"

```
Weights + optimizer (no ZeRO): 16 × 405B = 6,480 GB
H100-80GB: 6,480 / 80 = 81 GPUs minimum (weights + optimizer only)
With activations + overhead: ~100+ GPUs

With ZeRO-3 across 64 GPUs: 6,480 / 64 = ~101 GB/GPU → still too large
With ZeRO-3 across 128 GPUs: 6,480 / 128 = ~51 GB/GPU ✓
```

In practice, Meta trained 405B on 16,000 GPUs using TP=8, PP=16, DP=128.

### "How long can the context be before KV cache overflows?"

Rearrange: `max_seq_len = (HBM_budget_for_KV) / (kv_per_token × batch_size)`

For 70B, single H100, budgeting 40 GB for KV (other 40 GB for weights):
```
max_seq_len = 40 GB / (328 KB × 1) = 122,000 tokens
```
At batch=8: 40 GB / (328 KB × 8) = 15,000 tokens per sequence.

---

## Memory Budget Worksheet

```
Available:  HBM_capacity × n_gpus
Consumed:
  weights:          2 × N_params / tp / pp                 [BF16]
  optimizer:        12 × N_params / tp / pp / dp           [Adam, ZeRO-1+]
  gradients:        2 × N_params / tp / pp                 [BF16, during backward]
  activations:      B × S × d × L × 10 / tp / pp          [approx, w/o checkpointing]
  kv_cache:         kv_per_token × seq_len × batch_size    [inference only]
  framework overhead: ~1-2 GB
```

---

## See Also

- [LLM inference model](llm_inference.md) — how KV cache size drives decode throughput
- [Parallelism strategies](parallelism.md) — how TP/PP/DP reduce per-GPU memory
- [Attention](../workloads/attention.md) — where KV cache comes from; GQA/MQA reduce kv_heads and thus KV memory
- [Hardware roofline params](../hardware/roofline_params.md) — HBM capacity by chip
