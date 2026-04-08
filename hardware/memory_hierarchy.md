# Memory Hierarchy in AI Accelerators

← [Hardware Index](index.md)

Memory hierarchy is the dominant factor in AI accelerator performance. Understanding bandwidth and capacity at each level is essential for any analytical model.

---

## Hierarchy Levels (NVIDIA Hopper example)

```
Registers (per SM)
    └─ 256 KB, ~10s of TB/s effective, zero-latency within warp
Shared Memory / L1 (per SM)
    └─ up to 228 KB, ~10 TB/s, ~30 cycles latency
L2 Cache (per chip)
    └─ 50 MB (H100), ~6-12 TB/s, ~200 cycles
HBM (off-chip DRAM)
    └─ 80 GB (H100), 3.35 TB/s, ~500+ ns latency
NVLink (GPU-to-GPU)
    └─ 900 GB/s bidirectional (H100), used for tensor/pipeline parallel
PCIe / NVLink-C2C (host-to-GPU)
    └─ 128 GB/s (PCIe 5.0 x16), 900 GB/s (NVLink-C2C in GB200)
```

---

## Bandwidth Numbers by Level

| Level | H100 | MI300X | Notes |
|-------|------|--------|-------|
| HBM | 3.35 TB/s | 5.3 TB/s | Aggregate across stacks |
| L2 | ~6 TB/s | ~6 TB/s | Estimated |
| Shared Mem / L1 | ~10 TB/s per SM | — | Measured via microbenchmarks |
| NVLink | 900 GB/s | 896 GB/s (Infinity Fabric) | Bidirectional |

---

## Design Implications

- **Weight loading during inference:** large transformer weights must stream from HBM → bottleneck for decode
- **Activation recomputation (gradient checkpointing):** trades compute for HBM BW reduction during training
- **Flash Attention:** fuses attention to keep activations in SRAM, avoiding HBM round-trips
- **Kernel fusion:** reduces HBM traffic by keeping intermediate tensors in L1/registers

---

## See Also

- [Roofline parameters](roofline_params.md)
- [Roofline model](../modeling/roofline.md)
- [Operators and memory access patterns](../workloads/operators.md)
