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

## Intuition: a GPU has two speeds

A GPU has **two independent speed limits**, not one: how fast it does arithmetic
(*compute*, FLOP/s) and how fast it moves operands between memory and the compute
units (*bandwidth*, byte/s). On modern hardware these are wildly out of balance —
an H100 can do ~300 arithmetic ops in the time it moves a single byte
(989 TFLOPS ÷ 3.35 TB/s ≈ 295 FLOP/B).

Adam Mainz's *chef-and-runner* analogy captures it: a blazing-fast chef (compute)
stands idle while a slow runner (memory) fetches one ingredient per trip, so output
is set by the runner — until the *recipe* changes so each ingredient takes real
work (more math per byte), at which point the chef becomes the limit. "How much math
per byte does this kernel do?" (its [arithmetic intensity](#model-definition)) is
what decides which ceiling you hit — often guessable from the shape of the code
alone, before measuring.
— *[Source](https://x.com/MainzOnX/status/2077757143592186262)*

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

## Worked Contrast: Vector Add vs. Matmul

The same machine puts these two ops on opposite roofs purely by arithmetic intensity:

| Kernel (BF16) | Math | Bytes moved | AI (FLOP/B) | Roof |
|---|---|---|---|---|
| Vector add `c = a + b`, 1M elems | 1M FLOP | 6 MB (read a, b; write c) | ~0.17 | Memory-bound |
| 4096×4096 matmul | 2·4096³ ≈ 137 GFLOP | 96 MB (3 × 32 MB) | ~1,365 | Compute-bound |

On the H100 (ridge ≈ 295 FLOP/B) the add sits ~1,700× below the ridge (memory-bound);
the matmul sits ~5× above it (compute-bound).

**A matmul isn't inherently compute-bound — its shape decides.** For an N×N BF16
matmul, FLOPs = 2N³ and bytes ≈ 6N² (each of A, B, C read/written once), so:

```
AI_matmul = 2N³ / 6N² = N/3   [FLOP/B]
```

Small matmuls fall below the ridge (N=64 → ~21 FLOP/B, memory-bound); large ones
rise far above it (N=4096 → ~1,365 FLOP/B, compute-bound). The N/3 bound assumes a
well-written kernel that loads each operand from HBM once (tiling into on-chip
cache); a naive kernel that re-reads rows/columns moves more bytes and does worse.
Chained *memory-bound* elementwise ops benefit from [kernel fusion](../workloads/gpu_kernels.md):
fusing avoids writing/reading the intermediate to HBM, cutting traffic on the ceiling
you were actually hitting.
— *Worked examples: [Adam Mainz](https://x.com/MainzOnX/status/2077757143592186262)*

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
- [What is a GPU kernel?](../workloads/gpu_kernels.md) — kernel fusion and memory traffic, the *why* behind memory-bound ops
- [LLM inference model](llm_inference.md)
