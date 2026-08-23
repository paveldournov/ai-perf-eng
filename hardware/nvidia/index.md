---
type: Index
title: NVIDIA GPU Architecture
description: Landing page for NVIDIA data-center GPUs (H100, H200, B200, GB200).
tags: [nvidia, gpu, hopper, blackwell]
timestamp: 2026-08-23T00:00:00-07:00
---

# NVIDIA GPU Architecture

← [Hardware Index](../index.md)

---

## Chip Coverage

| Chip | Architecture | Process | HBM | Peak dense FLOPS | HBM BW (TB/s) |
|------|-------------|---------|-----|-----------------|----------------|
| [H100 SXM5](h100.md) | Hopper | TSMC 4N | HBM3 80 GB | 989 TF BF16 · 1,979 TF FP8 | 3.35 |
| H200 SXM5 | Hopper | TSMC 4N | HBM3e 141 GB | 989 TF BF16 · 1,979 TF FP8 | 4.8 |
| B200 SXM | Blackwell | TSMC 4NP | HBM3e 192 GB | ~4,500 TF FP8 · ~9,000 TF FP4 | 8.0 |
| B300 (Blackwell Ultra) | Blackwell | TSMC 4NP | HBM3e 288 GB | ~7,500 TF FP8 · ~15,000 TF FP4 | 8.0 |
| Rubin* | Rubin | — | HBM4 288 GB | ~17 PF FP8 · ~50 PF FP4 | ~13 |
| GB200 NVL72 | Blackwell | — | 13.4 TB HBM3e (72 GPUs) | 360 PF FP8 · 720 PF FP4 | — |
| GB300 NVL72 | Blackwell | — | 20.7 TB HBM3e (72 GPUs) | 540 PF FP8 · 1,100 PF FP4 | — |

> The B200 row previously listed ~4,500 TFLOPS as **FP4**; that figure is the **FP8**
> dense rate — B200's dense FP4 is ~9,000 TFLOPS. Corrected August 2026 against
> [AI chip architectures](../architectures.md).
> `*` Rubin figures are analyst-derived, not vendor-published.

---

## Generational Throughline

The matrix instruction has migrated off the threads for five generations — the single
most useful lens for reading modern kernel code:

| Gen | Instruction | Issuer | Operand location |
|-----|-------------|--------|------------------|
| Volta | `mma.sync` (16×16×16) | 32-thread warp, synchronous | registers |
| Ampere | `mma.sync` + `cp.async` | warp | registers; async HBM→SMEM copies |
| Hopper | `wgmma.mma_async` (64×256×16) | 128-thread warp-group, async | B in SMEM descriptor |
| Blackwell | `tcgen05.mma` (256×256×16, two-SM) | **one thread**, async | descriptors; accumulator in **TMEM** |

Data movement decoupled on the same curve: synchronous per-thread loads → `cp.async`
→ **TMA** (descriptor-driven DMA engine with cluster multicast) → TMA direct-to-TMEM.
See [AI chip architectures](../architectures.md#nvidia-gpu--programmability).

---

## Architecture Deep Dives

- Hopper architecture — SM structure, TMA, FP8 tensor cores, NVLink 4
- Blackwell architecture — 2nd-gen NVLink Switch, FP4, 5th-gen tensor cores, transformer engine v2

---

## Key References

- NVIDIA H100 Whitepaper
- NVIDIA Hopper Architecture In-Depth (blog)
- Blackwell Architecture Technical Brief
