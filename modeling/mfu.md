---
type: Concept
title: MFU — Model FLOP Utilization
description: The single most useful number for how efficiently a workload uses its hardware.
tags: [mfu, utilization, efficiency, flops]
timestamp: 2026-05-30T23:45:33-07:00
---

# MFU — Model FLOP Utilization

← [Modeling Index](index.md)

MFU is the single most useful number for characterizing how efficiently you're using your hardware. A high MFU means your code is well-optimized; a low MFU tells you there's headroom and points you toward where to look.

---

## Definition

```
MFU = (Observed FLOPs/sec) / (Peak FLOPs/sec of hardware)
    = (FLOPs per step) / (step_time × Peak FLOPs/sec)
```

MFU is a number between 0 and 1 (or 0% to 100%). It compares what you actually achieved to the theoretical maximum the hardware could do if running at full speed on every cycle.

---

## How to Compute It

**Step 1 — Count model FLOPs per forward pass.**

For a transformer with `L` layers, hidden dim `d`, FFN dim `d_ff`, sequence length `S`, batch size `B`:

```
FLOPs_forward ≈ 2 × B × S × (
    L × (4×d² + 2×d×d_ff)   ← attention projections + FFN
  + L × 4×d×S               ← QK^T and AV attention (O(S²))
)
```

The dominant term for long context is the `L×4×d×S` attention FLOPs; for moderate sequence lengths the MLP term dominates. For a rough estimate, use the well-known approximation:

```
FLOPs_forward ≈ 2 × N × B × S       (N = number of parameters)
```

This holds because each parameter is touched once per token — one multiply-accumulate (2 FLOPs) — ignoring the attention quadratic term.

**Step 2 — Scale for training.**

Training backward pass costs ≈ 2× the forward (gradients w.r.t. inputs and weights):
```
FLOPs_train_step ≈ 6 × N × B × S
```

**Step 3 — Measure step time and compute MFU.**

```python
import time
t0 = time.perf_counter()
loss = model(batch)
loss.backward()
optimizer.step()
t1 = time.perf_counter()

step_time = t1 - t0
achieved_flops_per_sec = flops_per_step / step_time
mfu = achieved_flops_per_sec / peak_flops_per_sec
```

---

## Typical MFU Values

| Workload | Hardware | Typical MFU |
|----------|----------|------------|
| Training, large batch, BF16 | H100 (989 TFLOPS) | 35–55% |
| Training, large batch, BF16 | A100 (312 TFLOPS) | 40–50% |
| Inference prefill, large batch | H100 | 40–60% |
| Inference decode, batch=1 | H100 | 1–5% |
| Inference decode, batch=128 | H100 | 15–30% |

**Why is MFU never 100%?**

- Memory latency stalls: even compute-bound ops spend cycles waiting for data to arrive
- Kernel launch overhead: CPU dispatch, CUDA graph amortizes this
- Pipeline bubbles (PP): idle time between micro-batches
- Communication overhead (TP/DP): all-reduce time blocks compute
- Numerical housekeeping: LayerNorm, softmax, non-fused elementwise ops have low AI

**Why is decode MFU so low?** Decode is fundamentally memory-bandwidth-bound at small batch sizes. The GPU must load all 140 GB of a 70B model weights to produce *one token*. At batch=1, the tensor cores are nearly idle — they can multiply faster than HBM can supply data. See [LLM Inference Model](llm_inference.md) for the full picture.

---

## MBU — Memory Bandwidth Utilization

The bandwidth analog of MFU:

```
MBU = (Bytes transferred per step) / (step_time × Peak HBM BW)
```

For memory-bound operations (decode, LayerNorm, elementwise), MBU is the relevant metric. A decode step with MBU=80% is performing well; MFU of 5% for the same step is not a problem — it's expected.

**Reading MFU and MBU together:**

| MFU | MBU | Interpretation |
|-----|-----|----------------|
| High | Low | Compute-bound — well utilized |
| Low | High | Memory-bound — well utilized |
| Low | Low | Something is wrong: kernel launch overhead, synchronization, fragmentation |
| High | High | Physically impossible — the roofline model says one must be the bottleneck |

---

## Worked Example: LLaMA-3 8B Training on H100

**Model:** 8B params, BF16, L=32 layers, d=4096, d_ff=14336, S=4096 tokens, B=2

```
FLOPs per step ≈ 6 × N × B × S
              = 6 × 8e9 × 2 × 4096
              = 6 × 8e9 × 8192
              ≈ 3.9 × 10^14 FLOPs  (390 TFLOPs)

Peak H100 BF16: 989 TFLOPS

Target step time (at 40% MFU): 390 / (989e12 × 0.40) ≈ 0.99 sec
```

If your measured step time is 2.0 sec, your MFU = 390 / (989 × 2.0) ≈ **20%** — there's significant room to optimize (batch size, kernel fusion, TP configuration).

---

## What Drives MFU Up

- **Larger batch size** — amortizes launch overhead, keeps tensor cores saturated
- **Longer sequences** — more FLOPs per parameter load (higher arithmetic intensity)
- **Flash Attention** — reduces HBM round-trips, keeps attention compute-bound
- **Kernel fusion** — fewer kernel launches, fewer intermediate HBM reads/writes
- **BF16 / FP8** — uses Tensor Cores instead of CUDA cores; 4–8× higher peak
- **Proper TP sizing** — don't exceed NVLink domain; communication overhead kills MFU

---

## See Also

- [Roofline model](roofline.md) — MFU and roofline are complementary: roofline predicts the ceiling, MFU measures how close you are
- [LLM inference model](llm_inference.md) — decode MFU is low by design; throughput is the right metric
- [Parallelism strategies](parallelism.md) — TP and PP both eat into MFU via communication overhead
- [GEMM](../workloads/gemm.md) — GEMM AI determines whether you're on the compute or BW side of the roofline
