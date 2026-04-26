# Roofline Parameters by Chip

← [Hardware Index](index.md) | [Roofline model](../modeling/roofline.md)

Roofline ridge point = Peak FLOPS / Peak Memory Bandwidth.
Operations with arithmetic intensity **above** the ridge point are compute-bound; below it, memory-bandwidth-bound.

---

## Data Table

| Chip | Precision | Peak FLOPS (dense) | Peak HBM BW | Ridge Point |
|------|-----------|-------------------|-------------|-------------|
| H100 SXM5 | BF16 tensor | 989 TFLOPS | 3.35 TB/s | ~295 FLOP/B |
| H100 SXM5 | FP8 tensor | 1979 TFLOPS | 3.35 TB/s | ~591 FLOP/B |
| H200 SXM5 | BF16 tensor | 989 TFLOPS | 4.8 TB/s | ~206 FLOP/B |
| H200 SXM5 | FP8 tensor | 1979 TFLOPS | 4.8 TB/s | ~412 FLOP/B |
| B200 SXM | FP4 tensor | ~4500 TFLOPS | 8.0 TB/s | ~563 FLOP/B |
| MI300X | BF16 tensor | 1307 TFLOPS | 5.3 TB/s | ~247 FLOP/B |
| TPUv5p | BF16 | 459 TFLOPS | 2.76 TB/s | ~166 FLOP/B |
| TPU v6e (Trillium) | BF16 | 918 TFLOPS | 1.64 TB/s | ~560 FLOP/B |
| TPU v6e (Trillium) | Int8 | 1,836 TOPs | 1.64 TB/s | ~1,120 FLOP/B |
| TPU v8t | FP4 | 12,600 TFLOPS | 6.53 TB/s | ~1,930 FLOP/B |
| TPU v8i | FP4 | 10,100 TFLOPS | 8.60 TB/s | ~1,174 FLOP/B |
| Gaudi3 | BF16 | 1835 TFLOPS | 3.7 TB/s | ~496 FLOP/B |

> Numbers are for peak dense (non-sparse) unless noted. Sparse (2:4) doubles FLOPS on NVIDIA chips.

---

## Worked Example: LLM Decode on H100

A single attention head query during decode:
- FLOPs ≈ 2 × seq_len × d_model (matmul)
- Bytes moved ≈ KV cache size loaded from HBM

Typical arithmetic intensity for decode: **< 10 FLOP/B** → deeply memory-bandwidth-bound on all chips above.

---

## See Also

- [Roofline model](../modeling/roofline.md)
- [H100 specs](nvidia/h100.md)
