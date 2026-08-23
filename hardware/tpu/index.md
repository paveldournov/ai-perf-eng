---
type: Index
title: Google TPU Family
description: Landing page for Google's TPU generations and their MXU/ICI architecture.
tags: [google, tpu, mxu, ici]
timestamp: 2026-08-23T00:00:00-07:00
---

# Google TPU Family

← [Hardware Index](../index.md)

Google's Tensor Processing Units (TPUs) are custom ASICs designed for matrix-multiply-heavy ML workloads. Each generation is built around systolic-array MXUs (Matrix Multiply Units), on-chip SRAM (Vmem), HBM, and Google's proprietary Inter-Chip Interconnect (ICI) fabric.

---

## Generations

| Generation | Code Name | Primary Use | BF16 TFLOPS/chip | HBM/chip | ICI BW/chip |
|------------|-----------|-------------|-----------------|----------|-------------|
| TPU v4     | —         | Training    | 275             | 32 GB    | 600 GB/s    |
| TPU v5p    | —         | Training    | 459             | 95 GB    | ~600 GB/s   |
| [TPU v6e](tpu_v6e.md) | Trillium | Training + Inference | 918 | 32 GB | 800 GB/s |
| [TPU v7x](tpu_v7x.md) | Ironwood (probable) | Training + Inference | ~1,533 (BF16 est.) | 192 GB | 1.2 TB/s |
| [TPU v8t](tpu_v8.md)  | Sunfish  | Training    | ~4,200 (BF16 est.) | 216 GB | 19.2 Tbps |
| [TPU v8i](tpu_v8.md)  | Zebrafish | Inference  | ~3,360 (BF16 est.) | 288 GB | 19.2 Tbps |

> TPU v8 specs are reported at FP4; BF16 estimates assume 3× ratio typical for recent MXU designs.
> TPU v7x figures are third-party-reported (FP8 ~4.6 PFLOP/s, HBM 7.38 TB/s); BF16 estimate assumes the same 3× FP8→BF16 ratio. See [TPU v7x](tpu_v7x.md).
> **v5p ICI corrected** (Aug 2026): this table previously read 4.8 TB/s, which is the
> published **4,800 Gbps** figure read as bytes — ~600 GB/s per chip. The old value was
> also internally inconsistent, sitting 6× above v6e and 4× above v7x in the same column.
> Per-chip ICI scales ~250 GB/s (v2) → 1.2 TB/s bidirectional (Ironwood) → 2× that on v8t.

---

## Common Architectural Features

- **MXU (Matrix Multiply Unit):** systolic array executing `output += weight × activation`, **weight-stationary** (weights preload one per cell; activations propagate one column per cycle; once data enters the array no memory access occurs). MXU shape history: 256×256 INT8 on v1 → 128×128 BF16 from v2 → back to 256×256 on v6e (65,536 MACs/array/cycle), kept on v7/v8. Per-TensorCore MXU counts: 1 (v2) → 2 (v3) → 4 (v4/v5e/v5p). The tax is **underfill** — a 128×128 matmul on a 256×256 array wastes 75% of the silicon, which is why XLA pads to multiples of 128 (256 on v6e+)
- **TensorCore internals:** MXU(s) + **VPU** (a *2D* vector machine — VREGs shaped (8, 128) on v4/v5p, 4 FP ALUs per lane×sublane) + **Scalar Unit** (the only block that fetches instructions; issues a **322-bit VLIW bundle** every cycle across 8 slots) + **XLU** for cross-lane reductions + Transpose/Permute Unit. Most of the speedup in modern TPU programs comes from **VPU/MXU overlap**
- **Vmem (vector memory / on-chip SRAM):** fast scratchpad for activations and intermediate results
- **ICI (Inter-Chip Interconnect):** direct chip-to-chip links forming torus or Boardfly topologies; no CPU/PCIe in the critical path
- **SparseCore:** dedicated dataflow units for embedding lookup — the inverse access pattern to dense matmul (irregular, indirect, all-to-all), delivering **5–7× on embedding-heavy models for ~5% of die area and power**. Counts: 4/chip on v4, v5p and v7; 2 on v6e; **removed entirely on v8i**, which substitutes the CAE on its I/O chiplet
- **Optical Circuit Switching (v4+):** **Palomar** 3D-MEMS switches (physically rotating mirrors) between 64-chip cubes — a v4 superpod uses 48 of them to wire 4,096 chips into one 3D torus. Reconfiguration is millisecond-class, which is fine because it is *circuit*-switched: pick a topology at job start, run it for a week. Three problems collapse into one component — per-workload topology (twisted tori give up to **70% better bisection**), sub-pod slicing, and fault tolerance (a dead chip is optically swapped for a spare cube). The same primitive runs at datacenter scale as **Apollo** inside Jupiter

---

## Pod & Cluster Scale

| Generation | Pod Size | Pod BF16 PFLOPs | Max Cluster |
|------------|----------|-----------------|-------------|
| TPU v5p    | 8,960 chips | 4.1 EFLOPS  | —           |
| TPU v6e    | 256 chips   | 235 PFLOPs  | 91 EFLOPS   |
| TPU v8t    | 9,600 chips | 121 EFLOPS (FP4) | >1 M chips |

---

## Scale-out fabrics

Through v7, scale-out ran over a single fabric — **Jupiter**, all-optical at the spine
since 2022 via Apollo OCS, carrying 13 Pb/s of bisection per building. With v8t it
**split in two**: **Virgo** took east-west TPU-to-TPU traffic (flat, two-layer,
non-blocking high-radix — every TPU at most two switches from any other; one cluster
links 134,000+ chips at 47 Pb/s bisection, with 4× the per-chip bandwidth and 40% lower
unloaded latency than the prior DCN generation), while Jupiter retained north-south
(storage, general compute, inter-site). Each layer can now iterate on its own cadence.

Per-chip scale-out bandwidth is ~100 Gbps on v7 and 4× that on v8t — still **two orders
of magnitude below per-chip ICI**, which is what dictates the partitioning: tensor
parallelism and MoE expert routing stay inside ICI; data and pipeline parallelism cross
the scale-out fabric.

---

## See Also

- [AI chip architectures](../architectures.md) — the TPU read against NVIDIA, AMD, Cerebras, Trainium, and Groq
- [TPU v6e (Trillium)](tpu_v6e.md)
- [TPU v7x](tpu_v7x.md) — third-party-reported specs; MoE serving case study
- [TPU v8t / v8i](tpu_v8.md)
- [Interactive TPU v8 Architecture Explorer](https://paveldournov.github.io/ai-perf-eng/hardware/tpu/google_tpu_v8_architectural_analysis.html) — standalone interactive HTML report on TPU 8t/8i
- [Boardfly interconnect topology](boardfly.md)
- [Pathways on Cloud](../../scheduling/pathways.md) — single-controller runtime for multi-slice JAX workloads; PathwaysJob CRD, resilient training, multihost inference
- [Roofline parameters](../roofline_params.md)
- [References — TPU papers & docs](../../references/index.md)
- [Pallas — kernel programming for TPU/GPU](../../workloads/pallas_kernels.md) — how to write JAX kernels that explicitly control VMEM tiling and MXU dispatch
