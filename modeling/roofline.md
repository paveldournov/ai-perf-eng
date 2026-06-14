---
type: Concept
title: Roofline Model
description: The foundational tool for bounding performance — compute-bound vs memory-bandwidth-bound.
tags: [roofline, arithmetic-intensity, compute-bound, memory-bound]
timestamp: 2026-06-01T23:29:24-07:00
---

# Roofline Model

← [Modeling Index](index.md)

The roofline model is the foundational tool for bounding achievable performance. It answers: *is this operation compute-bound or memory-bandwidth-bound?*

---

## Model Definition

```
Attainable GFLOPS/s = min(Peak FLOPS/s,  Arithmetic_Intensity × Peak_BW)
                           ─────────────  ────────────────────────────────
                           compute roof         bandwidth roof (slope)
```

**Arithmetic Intensity (AI):**
```
AI = FLOPs / Bytes_transferred_from_DRAM    [FLOP / byte]
```

**Ridge point:** the AI at which compute and BW roofs intersect:
```
AI_ridge = Peak_FLOPS / Peak_BW
```

---

## Interpreting the Chart

```
 GFLOPS/s |          _________________________  ← compute roof (Peak FLOPS)
          |         /
          |        /  ← slope = Peak BW
          |       /
          |______/
          +-----------> Arithmetic Intensity (FLOP/B)
                  ↑
              AI_ridge
```

- Left of ridge → **memory-bandwidth-bound**: more BW or lower AI needed
- Right of ridge → **compute-bound**: more FLOPS or lower AI counts

---

## Worked Example: H100 + GEMM decode

- H100 Peak BF16: 989 TFLOPS, HBM BW: 3.35 TB/s → **ridge ≈ 295 FLOP/B**
- Decode GEMM (batch=1): AI ≈ 1 FLOP/B → 280× below ridge → memory-BW bound
- Achievable FLOPS = 1 × 3.35e12 = 3.35 TFLOPS (0.34% of peak!)

---

## Multi-Level Roofline

Extend the model with per-level BW ceilings:

```
Attainable = min(Peak FLOPS,
                 AI_L1   × BW_L1,
                 AI_L2   × BW_L2,
                 AI_HBM  × BW_HBM)
```

Useful when kernels have significant L2 reuse (e.g., Flash Attention).

---

## Worked Example: Hardware Selection via Roofline (H100 vs H200 vs B200)

**Source:** Paolo Perrone, "H100 vs H200 vs B200" — [medium.com/@paoloperrone](https://medium.com/@paoloperrone/h100-vs-h200-vs-b200-eaa2e5b32e45)

The roofline model directly answers which GPU to buy: identify your bottleneck first, then buy the chip that addresses it. Buying peak TFLOPS when your workload is bandwidth-bound is wasted money.

**The decision framework:**

| Constraint | Buy | Reason |
|------------|-----|--------|
| Model doesn't fit in 80 GB | H200 (141 GB) or B200 (192 GB) | Capacity is the blocker |
| Model fits but decode is slow | H200 (4.8 TB/s) | 43% more bandwidth, same compute as H100 |
| Need 4× inference throughput or FP4 | B200 (8 TB/s, native FP4) | Full generational leap |
| 70B model, good price/perf | H100 (80 GB, 3.35 TB/s) | Already sufficient; don't over-buy |

**Key insight:** Most LLM inference decode is memory-bandwidth-bound (arithmetic intensity << ridge point). Upgrading from H100 to H200 is purely a bandwidth/capacity upgrade — the Hopper compute die is identical, so existing TensorRT-LLM, vLLM, and SGLang deployments run unchanged. The B200 is a new architecture (dual-die, native FP4, 5th-gen Tensor Cores) and offers genuine compute gains only for workloads that can exploit FP4.

**Priority order for GPU selection:** fit the model (capacity) → saturate HBM BW (bandwidth) → saturate compute (TFLOPS). Compute is the last thing to optimize for in LLM serving.

---

## Limitations

- Assumes perfect memory access patterns (no latency, no bank conflicts)
- Does not model pipeline bubbles, synchronization, or launch overhead
- Sparse / irregular operations (MoE routing) need careful FLOPs counting
- Use as a **ceiling** — actual performance is always ≤ attainable

---

## See Also

- [Roofline params by chip](../hardware/roofline_params.md)
- [GEMM arithmetic intensity](../workloads/gemm.md)
- [LLM inference model](llm_inference.md)
