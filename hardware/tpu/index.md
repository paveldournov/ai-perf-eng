# Google TPU Family

← [Hardware Index](../index.md)

Google's Tensor Processing Units (TPUs) are custom ASICs designed for matrix-multiply-heavy ML workloads. Each generation is built around systolic-array MXUs (Matrix Multiply Units), on-chip SRAM (Vmem), HBM, and Google's proprietary Inter-Chip Interconnect (ICI) fabric.

---

## Generations

| Generation | Code Name | Primary Use | BF16 TFLOPS/chip | HBM/chip | ICI BW/chip |
|------------|-----------|-------------|-----------------|----------|-------------|
| TPU v4     | —         | Training    | 275             | 32 GB    | 600 GB/s    |
| TPU v5p    | —         | Training    | 459             | 95 GB    | 4.8 TB/s    |
| [TPU v6e](tpu_v6e.md) | Trillium | Training + Inference | 918 | 32 GB | 800 GB/s |
| [TPU v8t](tpu_v8.md)  | —        | Training    | ~4,200 (BF16 est.) | 216 GB | 19.2 Tbps |
| [TPU v8i](tpu_v8.md)  | —        | Inference   | ~3,360 (BF16 est.) | 288 GB | 19.2 Tbps |

> TPU v8 specs are reported at FP4; BF16 estimates assume 3× ratio typical for recent MXU designs.

---

## Common Architectural Features

- **MXU (Matrix Multiply Unit):** systolic array executing `output += weight × activation`; v6e expanded from 128×128 to 256×256 tiles (4× FLOPs/cycle)
- **Vmem (vector memory / on-chip SRAM):** fast scratchpad for activations and intermediate results
- **ICI (Inter-Chip Interconnect):** direct chip-to-chip links forming torus or Boardfly topologies; no CPU/PCIe in the critical path
- **SparseCore:** dedicated units for embedding lookup (present since v4; 3rd-gen in v6e)
- **Optical Circuit Switching (v4+):** dynamically reconfigures pod topology at runtime

---

## Pod & Cluster Scale

| Generation | Pod Size | Pod BF16 PFLOPs | Max Cluster |
|------------|----------|-----------------|-------------|
| TPU v5p    | 8,960 chips | 4.1 EFLOPS  | —           |
| TPU v6e    | 256 chips   | 235 PFLOPs  | 91 EFLOPS   |
| TPU v8t    | 9,600 chips | 121 EFLOPS (FP4) | >1 M chips |

---

## See Also

- [TPU v6e (Trillium)](tpu_v6e.md)
- [TPU v8t / v8i](tpu_v8.md)
- [Roofline parameters](../roofline_params.md)
- [References — TPU papers & docs](../../references/index.md)
