---
type: Index
title: Apple Silicon Accelerators
description: Landing page for Apple's on-device AI accelerators (the Neural Engine).
tags: [apple, ane, npu, edge, on-device]
timestamp: 2026-07-03T00:00:00-07:00
---

# Apple Silicon Accelerators

← [Hardware Index](../index.md)

Apple integrates a fixed-function matrix accelerator — the **Neural Engine (ANE)** — into every
recent A-series and M-series SoC. Unlike the datacenter GPUs and TPUs covered elsewhere, these are
**edge / on-device** parts: single-chip, unified-memory, ~1–2 W, and optimized for inference energy
efficiency rather than throughput or scale-out.

---

## Chip Coverage

| Chip | Type | Primary Use |
|------|------|-------------|
| [Neural Engine (ANE)](ane.md) | Fixed-function fp16 matrix accelerator | On-device inference (A11–A18, M1–M5) |

---

## See Also

- [Hardware family overview](../index.md)
- [Roofline parameters by chip](../roofline_params.md)
- [Roofline model](../../modeling/roofline.md)
