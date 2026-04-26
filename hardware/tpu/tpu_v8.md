# Google TPU v8 (Eighth Generation)

← [TPU Index](index.md) | [Roofline params](../roofline_params.md)

TPU v8 is Google's eighth-generation TPU, announced in April 2025. Unlike prior generations with a single chip SKU, v8 ships as **two purpose-built variants**: **TPU 8t** optimized for training and **TPU 8i** optimized for inference. Both target the "agentic era" of AI — large-scale, latency-sensitive, continuous-serving workloads.

---

## TPU 8t (Training Variant)

### Key Specifications

| Parameter | Value |
|-----------|-------|
| Peak FP4 compute | 12,600 TFLOPS (12.6 PetaFLOPS) |
| On-chip SRAM (Vmem) | 128 MB |
| HBM capacity | 216 GB |
| HBM bandwidth | 6,528 GB/s (~6.5 TB/s) |
| Chip-to-chip ICI bandwidth | 19.2 Tbps |
| Network topology | 3D torus |
| Superpod scale | 9,600 chips |
| Superpod peak (FP4) | 121 ExaFLOPs |
| Cluster scale | >1,000,000 chips |
| Special units | SparseCore, LLM Decoder Engine |
| Storage | TPUDirect Storage (10× faster I/O) |

### Roofline Parameters

| Precision | Peak FLOPS | HBM BW | Ridge Point |
|-----------|-----------|--------|-------------|
| FP4 | 12,600 TFLOPS | 6.53 TB/s | ~1,930 FLOP/B |
| BF16 (est.) | ~4,200 TFLOPS | 6.53 TB/s | ~643 FLOP/B |

### Performance vs TPU v7

- **Training price-performance:** +2.7×
- **Performance-per-watt:** +2×

---

## TPU 8i (Inference Variant)

### Key Specifications

| Parameter | Value |
|-----------|-------|
| Peak FP4 compute | 10,100 TFLOPS (10.1 PetaFLOPS) |
| On-chip SRAM (Vmem) | 384 MB (3× larger than 8t) |
| HBM capacity | 288 GB |
| HBM bandwidth | 8,601 GB/s (~8.6 TB/s) |
| Chip-to-chip ICI bandwidth | 19.2 Tbps |
| Network topology | [Boardfly](boardfly.md) (novel; −56% network diameter vs torus) |
| Scale-out (DCN) bandwidth | 400 Gbps per chip (4× vs prior gen) |
| Max connected cluster | 1,152 chips (1,024 active) |
| Special units | CAE (Collectives Acceleration Engine) |
| On-chip collective latency reduction | 5× vs prior gen |

### Roofline Parameters

| Precision | Peak FLOPS | HBM BW | Ridge Point |
|-----------|-----------|--------|-------------|
| FP4 | 10,100 TFLOPS | 8.60 TB/s | ~1,174 FLOP/B |
| BF16 (est.) | ~3,360 TFLOPS | 8.60 TB/s | ~391 FLOP/B |

### Performance vs TPU v7

- **Inference price-performance:** +80%
- **Serving capacity at equivalent cost:** ~2×
- **Performance-per-watt:** +2×

---

## Architecture Highlights

### What's New in v8

| Feature | Details |
|---------|---------|
| Split SKU design | Separate chips for training (8t) and inference (8i); each optimized for its workload |
| [Boardfly topology](boardfly.md) (8i) | Hierarchical high-radix topology (3-level: ring → copper full-mesh → OCS); reduces diameter from 16 to 7 hops (−56%) vs equivalent torus; inspired by Dragonfly |
| CAE — Collectives Acceleration Engine (8i) | Dedicated on-chip unit for all-reduce / all-gather operations; 5× on-chip latency reduction |
| LLM Decoder Engine (8t) | Specialized unit for autoregressive decode during training data generation |
| TPUDirect Storage (8t) | 10× faster direct chip-to-storage path; reduces checkpoint and data-loading bottlenecks |
| Large Vmem (8i) | 384 MB on-chip SRAM vs 128 MB on 8t; enables larger KV cache on-chip for decode |
| 3D torus (8t) | Extended from 2D torus (v6e) to 3D; reduces all-reduce hops in large pods |

---

## TPU 8t vs TPU 8i at a Glance

| Dimension | TPU 8t (Training) | TPU 8i (Inference) |
|-----------|-------------------|-------------------|
| FP4 TFLOPS | 12,600 | 10,100 |
| HBM capacity | 216 GB | 288 GB |
| HBM bandwidth | 6.5 TB/s | 8.6 TB/s |
| On-chip SRAM | 128 MB | 384 MB |
| ICI bandwidth | 19.2 Tbps | 19.2 Tbps |
| Topology | 3D torus | Boardfly |
| Max pod size | 9,600 chips | 1,152 chips |
| Special engine | LLM Decoder Engine | CAE |
| Target workload | Pre-training / fine-tuning | Serving / real-time inference |

---

## References

- Google Blog — "Eighth Generation TPUs: Two Chips for the Agentic Era" (April 2025): https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/
- Google Cloud Blog — "TPU 8t and TPU 8i Technical Deep Dive" (April 2025): https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive
- Cloud TPU Release Notes: https://cloud.google.com/tpu/docs/release-notes
- Cloud TPU Performance Guide: https://cloud.google.com/tpu/docs/performance-guide

---

## See Also

- [Boardfly topology deep-dive](boardfly.md)
- [TPU v6e (Trillium)](tpu_v6e.md) — previous generation
- [TPU family overview](index.md)
- [Roofline params](../roofline_params.md)
