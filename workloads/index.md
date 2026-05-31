# Workloads — AI Workload Taxonomy

← [Back to README](../README.md)

AI workloads decompose into a hierarchy: **regime → model type → operator → kernel**. Performance characteristics differ sharply between regimes.

---

## Regimes

| Regime | Compute Pattern | Memory Pattern | Bottleneck |
|--------|----------------|----------------|------------|
| Training | High FLOPs/batch, backward pass | Activations + weights in HBM | Compute or NVLink BW |
| Inference — prefill | Long prompt, high parallelism | KV cache write, weight load | Compute (high AI) |
| Inference — decode | One token/step, low batch | KV cache read, weight load | HBM bandwidth |

---

## Model Families

- Transformer / LLM — attention, MLP blocks; dominant modern workload
- CNN — convolutions; image/video; historically dominant
- MoE (Mixture of Experts) — sparse activation; routing overhead
- Diffusion models — iterative denoising; mixed attention + CNN

---

## Operator-Level View

- Operators overview — GEMM, attention, convolution, elementwise, reduction
- [GEMM](gemm.md) — general matrix multiply; workhorse of all DNN workloads
- [Attention](attention.md) — FlashAttention, paged attention, MLA
- [All-reduce / All-gather](collective_ops.md) — distributed training communication

## Kernel Development

- [Pallas](pallas_kernels.md) — JAX extension for custom GPU/TPU kernels; grids, Refs, memory spaces, TPU VMEM tiling

## Mechanistic Understanding

- [Transformer as a programmable computer](transformer_programs.md) — manually-set weights implementing Hello World, Lookup Table, Search, Sort, Decimal Addition; attention hardening, linearly shiftable encodings, LayerNorm bypass

---

## Arithmetic Intensity by Operation

| Operation | Typical AI (FLOP/B) | Bound |
|-----------|---------------------|-------|
| Large GEMM (training) | 100–1000+ | Compute |
| Small GEMM (decode, batch=1) | < 1 | Memory BW |
| Attention (prefill, long seq) | ~seq_len/2 | Compute for long seq |
| Attention (decode) | << 1 | Memory BW |
| LayerNorm / Softmax | ~1–5 | Memory BW |
| All-reduce | — | NVLink / network BW |

---

## See Also

- [Roofline model](../modeling/roofline.md)
- [Analytical LLM inference model](../modeling/llm_inference.md)
