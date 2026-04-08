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
