---
type: Hardware
title: Google TPU v6e (Trillium)
description: Google's sixth-generation TPU, first positioned for both training and inference at scale.
tags: [google, tpu, trillium, v6e]
resource: https://cloud.google.com/tpu/docs/v6e
timestamp: 2026-04-25T23:28:29-07:00
---

# Google TPU v6e (Trillium)

← [TPU Index](index.md) | [Roofline params](../roofline_params.md)

Trillium is Google's sixth-generation TPU, announced at Google I/O in May 2024 and generally available since December 2024. It is the first TPU generation explicitly positioned for both training and inference at scale.

---

## Key Specifications

| Parameter | Value |
|-----------|-------|
| Generation | v6e (sixth-gen, "Trillium") |
| Process | —  |
| MXU tile size | 256 × 256 (up from 128×128 in v5) |
| On-chip SRAM (Vmem) | 128 MB |
| HBM capacity | 32 GB |
| HBM bandwidth | 1,638 GB/s (~1.64 TB/s) |
| Peak BF16 (dense) | 918 TFLOPS |
| Peak Int8 (dense) | 1,836 TOPs |
| ICI bandwidth | 800 GB/s per chip |
| Pod topology | 2D torus, 256 chips |
| Pod peak BF16 | ~235 PFLOPs |
| Cluster scale | ~91 ExaFLOPs |
| SparseCore | 3rd-generation |
| Energy efficiency vs v5e | +67% |

---

## Roofline Parameters

| Precision | Peak FLOPS | HBM BW | Ridge Point |
|-----------|-----------|--------|-------------|
| BF16 | 918 TFLOPS | 1.64 TB/s | ~560 FLOP/B |
| Int8 | 1,836 TOPs | 1.64 TB/s | ~1,120 FLOP/B |

LLM decode (memory-bandwidth-bound) sits far left of the ridge point (~1–20 FLOP/B); matrix-multiply-intensive prefill can approach the compute roof.

---

## Architecture Improvements vs TPU v5e

| Dimension | v5e | v6e (Trillium) | Change |
|-----------|-----|---------------|--------|
| Peak BF16 TFLOPS | 197 | 918 | **+4.7×** |
| HBM capacity | 16 GB | 32 GB | **+2×** |
| HBM bandwidth | ~820 GB/s | 1,638 GB/s | **+2×** |
| ICI bandwidth | 400 GB/s | 800 GB/s | **+2×** |
| SparseCore gen | 2nd | 3rd | +2× embedding perf |
| Energy efficiency | baseline | +67% | — |

---

## Pod Interconnect

- **All-reduce bandwidth (per pod):** 102.4 TB/s
- **Bisection bandwidth (per pod):** 3.2 TB/s
- **Data-center network (per pod):** 25.6 Tbps
- **Topology:** 2D torus; multi-slice training supported via DCN

---

## Benchmark Highlights (vs TPU v5e)

| Workload | Speedup |
|----------|---------|
| Llama 2-70B training | 4×+ |
| Gemma 2-27B training | 4×+ |
| MaxText Default-32B  | 4×+ |
| Llama 2-7B training  | 3×+ |
| Stable Diffusion XL (offline) | 3.1× throughput |
| Stable Diffusion XL (server)  | 2.9× throughput |
| DLRM DCNv2 (embedding) | 5× |

- **Price-performance vs v5e:** 2.1×
- **Price-performance vs v5p:** 2.5×

---

## References

- Google Cloud Blog — "Introducing Trillium, sixth-generation TPUs" (May 2024): https://cloud.google.com/blog/products/compute/introducing-trillium-6th-gen-tpus
- Google Cloud Blog — "Trillium sixth-generation TPU is in preview" (Oct 2024): https://cloud.google.com/blog/products/compute/trillium-sixth-generation-tpu-is-in-preview
- Google Cloud Blog — "Trillium TPU is GA" (Dec 2024): https://cloud.google.com/blog/products/compute/trillium-tpu-is-ga
- Official TPU v6e Docs: https://cloud.google.com/tpu/docs/v6e
- Jouppi et al. (2023) "TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning." *ISCA 2023*. arXiv:2304.01433

---

## See Also

- [TPU v8t / v8i](tpu_v8.md) — successor generation
- [TPU family overview](index.md)
- [Roofline params](../roofline_params.md)
