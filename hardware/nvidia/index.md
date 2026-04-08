# NVIDIA GPU Architecture

← [Hardware Index](../index.md)

---

## Chip Coverage

| Chip | Architecture | Process | HBM | Peak BF16 TFLOPS | HBM BW (TB/s) |
|------|-------------|---------|-----|-----------------|----------------|
| [H100 SXM5](h100.md) | Hopper | TSMC 4N | HBM3 80 GB | 989 | 3.35 |
| [H200 SXM5](h200.md) | Hopper | TSMC 4N | HBM3e 141 GB | 989 | 4.8 |
| [B200 SXM](b200.md) | Blackwell | TSMC 4NP | HBM3e 192 GB | ~4500 (FP4) | 8.0 |
| [GB200 NVL72](gb200.md) | Blackwell | — | HBM3e ×36 | rack-scale | — |

---

## Architecture Deep Dives

- [Hopper architecture](hopper.md) — SM structure, TMA, FP8 tensor cores, NVLink 4
- [Blackwell architecture](blackwell.md) — 2nd-gen NVLink Switch, FP4, 5th-gen tensor cores, transformer engine v2

---

## Key References

- NVIDIA H100 Whitepaper
- NVIDIA Hopper Architecture In-Depth (blog)
- Blackwell Architecture Technical Brief
