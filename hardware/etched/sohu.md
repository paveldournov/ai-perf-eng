---
type: Hardware
title: Etched Sohu — Transformer-Specific Inference ASIC
description: Etched's Sohu is a transformer-only ASIC that hardwires the transformer graph into silicon (dual systolic-array + attention engines) for ~20× H100 inference throughput at high batch.
tags: [etched, sohu, asic, transformer, inference, hardwired, systolic-array, attention, hbm3e, throughput, fp8]
resource: https://wafer.substack.com/p/breaking-down-etcheds-sohu
timestamp: 2026-07-25T00:00:00-07:00
---

# Etched Sohu — Transformer-Specific Inference ASIC

← [Etched Index](index.md) | [Hardware Index](../index.md) | [Roofline params](../roofline_params.md)

**Sohu** is an inference ASIC from Etched that does one thing: run **transformers**.
Where a GPU spends most of its die on programmability (schedulers, register files,
caches, a general SIMT datapath) and can execute any model, Sohu **burns the
transformer dataflow directly into the silicon** — no CUDA, no XLA, no
data-dependent control flow beyond what a transformer forward pass needs. The bet
is that if the model architecture is fixed, the ~2-3× of silicon a GPU spends on
being general-purpose can be reclaimed for raw matmul and attention throughput.

> **Sourcing caveat.** Figures below combine Etched's public marketing claims
> (500k+ tokens/s, ~20× H100, 144 GB HBM3E) with the analysis in Ludic /
> *Wafer* (2026), which reverse-engineers the dual-engine design from Etched's
> patents and *estimates* several unpublished specs (HBM bandwidth, per-chip
> capacity, the sources of the speedup). Rows marked **(est.)** are the author's
> inference, not disclosed silicon specs. This is a **not-yet-shipped** part; treat
> all performance numbers as vendor claims under favorable benchmark conditions.

---

## The Core Idea: a Hardwired Transformer Graph

A GPU is a **mono-sized systolic array** problem: one general matmul datapath has
to serve both

- **feed-forward / QKV projections** — large, regular GEMMs, compute-intensive,
  large reduction dimension `K`; and
- **self-attention** — small-`K`, memory-intensive, interleaved matmul→softmax→matmul.

A single fixed-size compute unit is underutilized for one whenever it is tuned for
the other. Sohu splits them into **two dedicated engines**, each with its own HBM,
so weight reads never contend with KV-cache reads:

| Engine | Handles | Character |
|---|---|---|
| **Systolic-array engine** | QKV / FFN projections, layer norm — the big regular GEMMs | Weight-stationary, compute-bound |
| **Self-attention circuit** | Q·Kᵀ scoring, softmax, value aggregation | Native interleaved matmul↔softmax, memory-bound |

The transformer graph itself is **fixed at manufacturing** — the chip is not
programmable in the GPU sense. Multi-chip scaling is **tileable**: identical ICs in
a grid with **weight-stationary dataflow** — weights loaded once and flowing
top-down, activations streaming bidirectionally.

---

## Key Specifications

| Parameter | Value | Source |
|---|---|---|
| Type | Transformer-only inference ASIC | Etched |
| Execution model | Hardwired transformer graph — not programmable | Etched |
| Process node | TSMC 3 nm | inferred |
| HBM per chip | ~144 GB HBM3E | Etched |
| HBM bandwidth | ~4,800 GB/s **(est.)** — ≈1.4× an H100's 3,350 GB/s | est. |
| Server config | 8 chips → **~1,152 GB** aggregate HBM | Etched / est. |
| Peak FLOPS | **not disclosed** | — |
| Headline throughput | **500,000+ tokens/s** on an 8-chip server | Etched |
| Headline speedup | **~20× an H100** for transformer inference | Etched |
| Compute path | FP8 (benchmark), dual-engine parallel attention + FFN | Etched |

**Benchmark conditions for the 20× / 500k tok/s claim:** LLaMA-70B, **2048 input
tokens, 128 output tokens, FP8, 8-way model parallelism.** These conditions
(long prefill, short decode, very high batch) maximize the advantage.

---

## Where the ~20× Comes From

The *Wafer* analysis decomposes the claimed speedup into roughly:

- **~2–3×** from **silicon reallocation** — die area freed by dropping
  programmability overhead goes to compute.
- **~2.5–3×** from **utilization** — a fixed-function transformer pipeline can run
  near **~90%** MFU where an H100 on the same workload sits at **~30–40%**.
- **Remainder** from **parallel attention + FFN** execution across the two engines.

Multiplying these plausibly reaches ~20× *at high batch and favorable
prefill/decode ratios* — the numbers are internally consistent, not obviously
inflated. See [MFU](../../modeling/mfu.md) and [roofline](../../modeling/roofline.md)
for the utilization lens.

---

## "A Throughput Machine, Not a Latency Machine"

Sohu is optimized for **high-batch serving**, and its behavior degrades outside that regime:

- **Low batch size** → bandwidth-limited; the compute reallocation advantage
  shrinks because there is not enough work to keep the systolic engine fed.
- **Long context** → **KV-cache capacity** becomes the wall. Against the ~1,152 GB
  aggregate:

  | Batch | Context | KV precision | Fits in 1,152 GB? |
  |---|---|---|---|
  | 1000 | 2048 | FP8 | ✅ comfortably |
  | 1000 | 4096 | FP8 | ✅ still fits |
  | 1000 | 8192 | FP8 | ❌ exceeds capacity |
  | 1000 | any of the above | BF16 | KV doubles → tighter still |

So single-request latency and very-long-context serving are **not** where Sohu
wins; dense, high-throughput short/medium-context serving is.

---

## Software & Compiler

Because the target graph is fixed, the compiler is a **much simpler problem than
CUDA or XLA**:

- It recognizes specialized PyTorch / TensorFlow functions and translates **fused
  ops** (e.g. *layer-norm → linear*) into an intermediate representation for the
  hardwired pipeline.
- **Same model code compiles for both** a GPU (for training) and Sohu (for
  inference) — Sohu is an inference target, not a training chip.

Contrast with the general-purpose compiler stacks: [XLA / GPU kernels](../../workloads/gpu_kernels.md),
[Pallas](../../workloads/pallas_kernels.md).

---

## Roofline Placement

Sohu does not publish a peak-FLOPS number, so a precise ridge point can't be
computed. Qualitatively:

- HBM bandwidth **(est. ~4.8 TB/s)** is H200-class, above an H100's 3.35 TB/s.
- The dual-engine split is a **direct attack on the decode-side memory wall**:
  dedicating an attention engine + its own HBM to the small-`K`, low-arithmetic-
  intensity attention step is exactly the operation that sits far left of the
  ridge point on a GPU (see [Roofline params — LLM decode example](../roofline_params.md#worked-example-llm-decode-on-h100)).

---

## Strategic Risk: the Transformer Attractor

Sohu's entire premise is that **the transformer architecture is stable enough** to
be worth freezing in silicon. The *Wafer* piece frames this as the "transformer
attractor" — transformer/GPU co-evolution creates institutional lock-in, so
alternatives face both a hardware-compatibility and a research-mindshare barrier.

The counter-risk: **unlike SHA-256 in a Bitcoin ASIC, transformer internals shift
every 6–12 months.** Hybrid attention-SSM stacks, new attention/routing variants,
and MoE changes could each erode a hardwired design's advantage. A Bitcoin ASIC
targets an immutable spec; Sohu targets a moving one. This is the central execution
risk, distinct from whether the silicon works.

---

## Business Context

- **Funding:** ~$625 M raised.
- **Chief architect:** Saptadeep Pal, previously co-founder of Auradine (Bitcoin-
  mining ASICs at TSMC 3 nm) — i.e. a team with a track record of shipping a
  fixed-function ASIC against an immutable spec.
- **Signal:** Peter Thiel reportedly sold his entire NVIDIA position after
  investing.

Author's conclusion: the **dual-engine architecture is a real, defensible technical
idea**, and the ~20×-at-high-batch claim is plausible under the stated conditions —
but shipping-execution and architecture-drift risk remain substantial.

---

## References

- Ludic, N. / *Wafer* (2026). "Breaking Down Etched's Sohu." *Wafer* (Substack).
  [wafer.substack.com/p/breaking-down-etcheds-sohu](https://wafer.substack.com/p/breaking-down-etcheds-sohu)
- Etched — Sohu product claims (500k+ tokens/s, ~20× H100, 144 GB HBM3E).

---

## See Also

- [Hardware index](../index.md) — accelerator families (Sohu is a custom inference ASIC)
- [Apple ANE](../apple/ane.md) — the other fixed-function example here; trades programmability for edge energy efficiency, where Sohu trades it for datacenter throughput
- [Roofline parameters by chip](../roofline_params.md) — where Sohu's HBM sits vs H100/H200
- [MFU](../../modeling/mfu.md) — the utilization lens the ~90% claim rests on
- [GEMM](../../workloads/gemm.md) & [attention](../../workloads/attention.md) — the two operations Sohu splits across dedicated engines
