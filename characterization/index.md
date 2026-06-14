---
type: Index
title: Characterization — Benchmarking & Profiling
description: Measuring real hardware behavior through microbenchmarks and end-to-end profiling.
tags: [characterization, benchmarking, profiling]
timestamp: 2026-05-29T21:29:57-07:00
---

# Characterization — Benchmarking & Profiling

← [Back to README](../README.md)

Characterization bridges the gap between analytical models and reality. It measures actual hardware behavior through microbenchmarks and end-to-end profiling.

---

## Techniques

| Technique | Purpose | Tools |
|-----------|---------|-------|
| Microbenchmarks | Measure raw HW parameters (BW, latency, FLOPS) | custom CUDA/HIP, STREAM |
| Kernel profiling | Per-kernel time, occupancy, memory transactions | Nsight Compute, ROCm profiler |
| End-to-end profiling | Trace full model step, pipeline stalls | Nsight Systems, PyTorch profiler |
| Scaling experiments | MFU vs batch size, sequence length, parallelism | custom harnesses |
| Roofline measurement | Plot actual ops on roofline chart | Nsight Compute built-in |

---

## What to Measure

### Hardware Limits
- Peak HBM bandwidth (STREAM-like test, stride-1 reads)
- Peak tensor core FLOPS (large square GEMMs)
- L2 bandwidth and effective capacity
- NVLink / Infinity Fabric point-to-point BW

### Kernel Efficiency
- **Memory utilization (MBU):** `Achieved BW / Peak HBM BW`
- **Model FLOP utilization (MFU):** `Achieved FLOPS / Peak FLOPS`
- **Occupancy:** active warps / max warps (indicator, not direct perf predictor)

### System-Level
- Inter-GPU all-reduce BW (NCCL tests)
- Host-to-device PCIe BW
- End-to-end tokens/s, TTFT, inter-token latency

---

## Key Metrics Definitions

```
MFU  = (Model FLOPs per step) / (Observed step time × Peak FLOPS)
MBU  = (Bytes expected from model) / (Observed step time × Peak BW)
```

High MFU + low MBU → compute-bound (good for training)
Low MFU + high MBU → BW-bound (typical for decode)

---

## Profiling Workflow

1. Run microbenchmarks → establish hardware baseline
2. Profile a single kernel with Nsight Compute → identify bottleneck
3. Compare achieved AI against roofline → quantify headroom
4. Profile end-to-end with Nsight Systems → find pipeline stalls
5. Sweep batch size / sequence length → build empirical scaling curves

---

## See Also

- [Roofline parameters](../hardware/roofline_params.md)
- [Roofline model](../modeling/roofline.md)
- MFU model
