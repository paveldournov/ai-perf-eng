---
type: Kernel
title: Pallas — Custom Kernel Programming in JAX
description: JAX extension for writing custom GPU/TPU kernels when XLA leaves performance on the table.
tags: [pallas, jax, kernels, gpu, tpu, triton]
resource: https://docs.jax.dev/en/latest/pallas/index.html
timestamp: 2026-05-31T23:00:31-07:00
---

# Pallas — Custom Kernel Programming in JAX

← [Workloads Index](index.md)

Pallas is a JAX extension for writing custom kernels that run on GPUs and TPUs.
It maintains JAX's Python feel while exposing the level of hardware control needed
when XLA's automatic code generation leaves performance on the table.

Source: [Pallas for Beginners](https://huggingface.co/blog/ariG23498/pallas-for-beginners) (Aritra Roy Gosthipaty, HuggingFace, 2025)

---

## Why Pallas Exists

Standard JAX operations (`jnp.matmul`, `jnp.sum`, etc.) are lowered to XLA, which
generates kernels automatically.  This is sufficient for most workloads but leaves
two gaps:

1. **Fused custom ops** — XLA may not fuse a sequence of operations the way you want,
   causing unnecessary HBM round-trips between steps.
2. **Hardware-specific layouts** — Tensor Core tile alignment, TPU MXU blocking,
   and scratchpad management require explicit control that XLA's heuristics may miss.

Pallas fills this gap the same way Triton does for CUDA — but as a first-class
JAX extension that composes with `jit`, `vmap`, and `grad`.

---

## Backend Support

| Backend | Compiler | Target hardware |
|---------|----------|-----------------|
| Mosaic (TPU) | Google Mosaic | TPU v4 / v5 / v6e / v7x / v8 |
| Mosaic GPU | Google Mosaic GPU | NVIDIA Hopper (H100) and newer |
| Triton (legacy) | OpenAI Triton | NVIDIA GPUs (best-effort, not recommended for new code) |

---

## Core Concepts

### Refs vs. Arrays

The fundamental shift from normal JAX: kernels are **not pure functions**.
They read from and write to mutable memory references (`Ref`), not arrays.

```python
# Normal JAX — pure, returns a value
def add(x, y):
    return x + y

# Pallas kernel — reads Refs, writes to Ref, returns nothing
def add_kernel(x_ref, y_ref, o_ref):
    x = x_ref[...]       # load from memory → jax.Array
    y = y_ref[...]
    o_ref[...] = x + y   # store back to memory (no return)
```

### pallas_call

Wraps a kernel into a callable JAX operation.  `out_shape` is required because
the kernel has no return value — JAX needs the shape/dtype upfront to allocate
output buffers before dispatch.

```python
from jax.experimental import pallas as pl

out = pl.pallas_call(
    add_kernel,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
)(x, y)
```

---

## Programming Model: Grids and Program IDs

A kernel is not the full computation — it is the work one **program instance**
does on one **tile** of data.  `pallas_call` launches a grid of instances:

```
grid=(n,)    →  n instances (like a 1D loop)
grid=(n, m)  →  n×m instances (like a 2D nested loop)
```

Each instance calls `pl.program_id(axis)` to discover its position and compute
which tile of the input it owns:

```python
BLOCK = 4

def add_blocked_kernel(x_ref, y_ref, o_ref):
    pid    = pl.program_id(0)               # which instance am I?
    start  = pid * BLOCK
    idx    = start + jnp.arange(BLOCK)
    o_ref[idx] = x_ref[idx] + y_ref[idx]   # operate only on my tile

out = pl.pallas_call(
    add_blocked_kernel,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    grid=(x.size // BLOCK,),               # one instance per tile
)(x, y)
```

**Questions to answer when writing a kernel (in order):**
1. What is one program instance responsible for?
2. How many instances do I need?
3. What block of each input does each instance read?
4. What block of each output does each instance write?
5. Are blocks shaped in a way the hardware likes? (multiples of 16 for Tensor Cores; 128 for TPU MXU)
6. Debug with `interpret=True` first.
7. Only then: optimize for performance.

---

## Memory Hierarchy (TPU)

TPUs expose three memory spaces through Pallas:

```
HBM  (device DRAM)
  large, slow (~2 TB/s)
  holds full tensors between kernel calls

VMEM  (vector SRAM / on-chip scratchpad)
  medium size (~tens of MB per chip), fast
  holds the active tile during compute
  maps to "shared memory" in CUDA terminology

SMEM  (scalar SRAM)
  very small, very fast
  holds loop indices, scalars, control values
```

Performance comes from **staged data movement**:

```
HBM → VMEM  (load tile)
VMEM → MXU  (compute)
MXU → VMEM  (accumulate)
VMEM → HBM  (store result)
```

Minimising HBM traffic — by reusing tiles in VMEM across the inner loop — is
the primary lever for TPU kernel performance.  This is the same tiling principle
as shared-memory blocking in CUDA (see [GEMM tiling](gemm.md)).

---

## Overlapping Communication and Compute

The staged HBM→VMEM→MXU pattern above is also the lever for hiding *communication*
behind compute on multi-chip workloads. The pattern, as applied in production MoE
serving (see [Fused MoE V2 case study](moe.md#serving-case-study-fused-moe-v2-on-tpu-ling-26-1t)):

- **VMEM residency** — keep the active token tile and output accumulator in VMEM across
  the inner loop so the routed GEMM never round-trips through HBM mid-computation.
- **Weight double-buffering** — prefetch the next expert's weights into a second VMEM
  buffer while the current expert's GEMM runs on the MXU, so DMA latency is masked.
- **Schedule heterogeneous units in parallel** — MXU (matmul), VPU (scale/quant), DMA
  (HBM↔VMEM), and ICI-DMA (chip↔chip all-to-all) can all be in flight at once; a good
  kernel keeps the routed-compute window full while routing, fp8 reorder, and dispatch
  proceed underneath it.

The result is that all-to-all dispatch/combine — normally the
[MoE communication wall](moe.md#expert-parallelism-and-the-all-to-all-wall) — is hidden
behind the expert GEMMs rather than serialized in front of them.

---

## Debugging

| Technique | How | When to use |
|-----------|-----|-------------|
| `interpret=True` | `pl.pallas_call(..., interpret=True)` | Logic debugging on CPU; only way to run Pallas without a GPU/TPU |
| `debug=True` | `pl.pallas_call(..., debug=True)` | Prints intermediate compilation stages |
| `pl.debug_print` | Inside kernel body | Inspect values mid-kernel (TPU only) |
| Compare modes | Run with and without `interpret=True` | If results diverge, file a bug — the compiled path is wrong |

---

## Relationship to Other Abstractions

```
jnp.matmul / jax.lax ops
    ↓  (XLA auto-generates kernels — sufficient for most cases)
Pallas  ←  you are here: explicit tiles, explicit memory movement
    ↓  (Mosaic / Mosaic GPU compiles to hardware)
TPU MXU instructions  /  NVIDIA Tensor Core PTX
```

Pallas sits one level below standard JAX and one level above raw hardware
assembly.  It composes fully with `jax.jit`, `jax.vmap`, and `jax.grad` —
a Pallas kernel can be differentiated through just like any JAX operation.

---

## See Also

- [GEMM tiling and hardware paths](gemm.md) — tile sizing, Tensor Cores, tf32/bf16
- [CUDA PTX](cuda_ptx.md) — the virtual ISA Triton/Pallas lower to on NVIDIA
- [JAX GPU efficiency demo](jax_gpu_efficiency_demo.py) — runnable benchmark
- [TPU hardware family](../hardware/tpu/index.md) — MXU, VMEM, ICI specs
- [Pallas official docs](https://jax.readthedocs.io/en/latest/pallas/index.html)
