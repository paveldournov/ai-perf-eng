---
type: Kernel
title: CUDA PTX — NVIDIA's Virtual ISA
description: PTX is the virtual-machine ISA between CUDA C++ and hardware SASS; forward-compatible, JIT-able, and the first inspectable compilation stage.
tags: [ptx, cuda, nvidia, gpu, compilation, sass, isa, kernels]
resource: https://philipfabianek.com/posts/cuda-ptx-introduction
timestamp: 2026-07-14T00:00:00-07:00
---

# CUDA PTX — NVIDIA's Virtual ISA

← [Workloads Index](index.md)

**PTX (Parallel Thread Execution)** is NVIDIA's virtual-machine instruction set:
an ISA for an *abstract* GPU that captures the common features of all NVIDIA
hardware. It sits between the developer's CUDA C++ and the hardware-specific
machine code (SASS), providing forward compatibility and the first inspectable
stage of the compilation pipeline.

Source: [An Introduction to CUDA PTX](https://philipfabianek.com/posts/cuda-ptx-introduction)
(Philip Fabianek, 2025)

---

## Where PTX Sits in the Pipeline

```
CUDA C++  →  NVVM IR  →  PTX  →  SASS
             (LLVM IR)    (virtual ISA)   (hardware machine code)
```

- **NVVM IR** — NVIDIA's specialized LLVM IR. Third-party languages and compilers
  (Triton, Rust GPU) target this or PTX rather than emitting raw machine code.
- **PTX** — virtual ISA for an abstract GPU; the first human-inspectable stage.
- **SASS** — actual streaming-assembly executed by a specific architecture.

`nvcc` performs two-stage compilation: CUDA C++ → PTX → SASS. By default it
embeds **both** PTX and SASS in the executable — SASS for immediate performance on
known architectures, PTX as a portable fallback.

---

## Forward Compatibility and JIT

- PTX compiled for `compute_70` runs on **newer** architectures (8.x, 9.x) but
  **not older** ones.
- When the driver encounters hardware for which no matching SASS is embedded, it
  **JIT-compiles the embedded PTX** at runtime to that hardware's SASS.
- This is why shipping PTX matters: it future-proofs a binary against GPUs that
  did not exist when it was compiled.

---

## Register Conventions

PTX uses an unbounded set of **virtual registers**, typed by naming convention:

| Prefix | Type |
|--------|------|
| `%rd`  | 64-bit addresses / pointers |
| `%r`   | signed 32-bit integer |
| `%u`   | unsigned 32-bit integer |
| `%f`   | 32-bit float |
| `%p`   | predicate (boolean) |

---

## Instruction Categories

**Data movement**
- `ld` — load from memory
- `st` — store to memory
- `mov` — copy between registers

**Computation**
- `add` — addition
- `mad` — fused multiply-add
- `mul.wide` — widening multiply (avoids overflow when computing addresses)

**Control flow**
- `setp` — set a predicate from a comparison
- `bra` — (conditional) branch
- `ret` — return from kernel

### Special registers

Read-only registers that map to CUDA C++ built-ins:

| PTX register | CUDA C++ |
|--------------|----------|
| `%tid.x`     | `threadIdx.x` |
| `%ntid.x`    | `blockDim.x` |
| `%ctaid.x`   | `blockIdx.x` |
| `%nctaid.x`  | `gridDim.x` |

---

## Worked Example: Vector Addition

A complete element-wise vector-add kernel in PTX demonstrates the full shape of a
kernel:

1. **Signature & parameters** declared with `.param` directives.
2. **Thread identification** — compute a global index from `%ctaid`, `%ntid`, `%tid`.
3. **Boundary check** — `setp` produces a predicate, `bra` skips out-of-range threads.
4. **Address calculation** — `mul.wide` + `add` to index into the arrays.
5. **Memory access & arithmetic** — `ld` operands, `add`, `st` the result.
6. **Return** — `ret`.

---

## Compilation & Inspection Tooling

### `nvcc` flags controlling embedded code

| Flag | Effect |
|------|--------|
| `-arch sm_XX` | embed both PTX and SASS for `sm_XX` |
| `-arch compute_XX` | embed PTX only |
| `-gencode arch=compute_XX,code=sm_XX` | precise control over what is embedded |

### Inspection

- **`cuobjdump`** — extract embedded PTX or disassemble SASS from a binary.
- **Nsight Compute (NCU)** — links source lines to compiled PTX/SASS in profiles.
- **Compiler Explorer (Godbolt)** — live view of C++ → PTX → SASS.

---

## Why It Matters for Performance Work

- **Bleeding-edge hardware features** are often reachable only through PTX before
  high-level CUDA C++ exposes them — e.g. `wgmma` warpgroup-level matrix-multiply
  instructions used by high-performance GEMM/attention kernels.
- The most common production pattern is **inline PTX assembly** inside CUDA
  `__global__` functions, for the few operations C++ cannot express.
- PTX is the **first inspectable compilation stage**, making it a key tool for
  performance debugging and understanding what the compiler actually emitted.

---

## See Also

- [What is a GPU kernel?](gpu_kernels.md) — the launch/thread/fusion basics one level above PTX
- [Pallas — Custom Kernel Programming in JAX](pallas_kernels.md) — higher-level
  kernel authoring that ultimately lowers through Triton → PTX on NVIDIA.
- [GEMM](gemm.md) — the workhorse operator whose fastest implementations rely on
  PTX-only instructions like `wgmma`.
- [NVIDIA GPU Architecture](../hardware/nvidia/index.md) — the SASS targets PTX
  compiles down to.
