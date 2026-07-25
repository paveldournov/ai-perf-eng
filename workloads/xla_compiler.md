---
type: Tool
title: XLA — The Compiler Beneath ML Frameworks
description: XLA compiles static-shape tensor programs from JAX/TF/PyTorch to CPU/GPU/TPU; StableHLO → classic HLO optimizer (fusion, layout, buffer assignment) → LLVM/Triton/Mosaic backends, with MLIR living at the interfaces.
tags: [xla, mlir, hlo, stablehlo, compiler, fusion, jax, tpu, pjrt, codegen]
resource: https://hiraditya.github.io/posts/a-tour-of-xla-where-mlir-lives/
timestamp: 2026-07-24T00:00:00-07:00
---

# XLA — The Compiler Beneath ML Frameworks

← [Workloads Index](index.md)

**XLA (Accelerated Linear Algebra)** is the compiler that powers JAX,
TensorFlow, and PyTorch/XLA. It takes tensor programs with **statically known
shapes**, compiles the whole program ahead of time, and emits optimized machine
code for CPUs, GPUs, and TPUs. Static shapes are the enabling assumption: they
unlock whole-program fusion, layout optimization, and static buffer allocation
with no runtime allocator overhead.

Source: [A Tour of XLA: Where MLIR Lives](https://hiraditya.github.io/posts/a-tour-of-xla-where-mlir-lives/)
(A. Hiranandani, 2025)

See also: [CUDA PTX](cuda_ptx.md) · [Pallas kernels](pallas_kernels.md) ·
[What is a GPU kernel?](gpu_kernels.md)

---

## The Pipeline at a Glance

```
Framework (JAX / TF / PyTorch)
      │  emits
      ▼
   StableHLO          ← portable, versioned MLIR interchange (the stable contract)
      │  xla/hlo/translate/
      ▼
  classic HLO         ← SSA graph over statically-shaped tensors; the mature core optimizer
      │  optimizer passes: simplify → layout → fusion → buffer assign → schedule
      ▼
  ┌───────────┬─────────────┬────────────┐
  │ CPU/GPU   │  cuBLAS/    │   TPU      │
  │ via LLVM  │  cuDNN,     │  via       │
  │ (→PTX)    │  Triton     │  libtpu /  │
  │           │  (→PTX)     │  Mosaic    │
  └───────────┴─────────────┴────────────┘
      │  executed through
      ▼
   PJRT runtime       ← hardware-agnostic client; compile-once-per-shape + caching
```

**Where MLIR lives:** at the *interfaces* — the StableHLO front-end, the Shardy
partitioner, and the Triton/Mosaic backends are all MLIR dialects. **Classic HLO
remains the mature core optimizer** and is *not* MLIR — it is XLA's own SSA IR.

---

## The Dialect Family (MLIR front-end)

Three related MLIR dialects sit above classic HLO:

| Dialect | Level | Role |
|---|---|---|
| **CHLO** (Client HLO) | Highest | Implicit broadcasting + composite ops; meant to be *decomposed*, not compiled directly |
| **MHLO** | Mid | MLIR-native rendering of classic HLO semantics; used during MLIR processing |
| **StableHLO** | Interchange | MHLO + versioned, backward-compatible serialization — the durable framework↔compiler contract |

StableHLO carries SSA values, tensor types, and ops like `dot_general`,
`broadcast_in_dim`, and `add`. The `xla/hlo/translate/` component bridges it into
classic HLO via passes such as `stablehlo_to_hlo`, `hlo_to_mhlo`, and `mhlo_to_hlo`.

---

## Classic HLO — the Internal IR

Classic HLO is an SSA graph of operations over statically-shaped tensors, with
**~150 opcodes**: `dot` (general contraction), `convolution`, `reduce`,
`dynamic-slice`, collectives like `all-reduce`, and `custom-call` as the escape
valve to external libraries.

Shape-and-layout notation appears throughout dumps:

```
f32[4,16]{1,0}
│    │    └── minor-to-major dimension order: {1,0} = row-major
│    └─────── shape: 4 × 16
└──────────── element type: float32
```

Reading HLO dumps and recompilation patterns is the day-to-day skill this whole
stack rewards.

---

## The Optimizer — "the reason XLA exists"

The optimizer runs on classic HLO. Key pass groups:

- **Simplification** — algebraic simplification, CSE, DCE. Strong enough to
  collapse whole expressions to identities or copies.
- **Layout assignment** — every array gets a physical layout (the `{1,0}`
  minor-to-major order). When a consumer demands a different layout than the
  producer supplies, a **copy** is inserted.
- **Fusion** — *the optimization that matters most.* Chains ops into a single
  kernel so intermediates stay in registers instead of streaming through HBM.
  Kinds: **loop, input, output, custom**.
  Example: `tanh(exp(x) * 2 + 1)` → one fused kernel, four ops, **one pass over
  memory**.
- **Buffer assignment** — static memory planning across the whole program.
  Computes buffer lifetimes, assigns concrete offsets, reuses space between
  non-overlapping lifetimes — all before execution. Enables **donation**
  (in-place update) with no runtime allocation.
- **Scheduling** — orders instructions to hide latency and overlap collective
  communication with compute in distributed runs.

Fusion, layout, and buffer assignment are exactly the levers a
[roofline analysis](../modeling/roofline.md) predicts: fusion raises arithmetic
intensity by killing memory round-trips.

---

## Code Generation — Three Backends

**CPU & GPU** route through **LLVM** via an emitters framework that lowers HLO
fusions to LLVM IR, then to object files or **PTX**. GPU codegen fans out:

- **cuBLAS / cuDNN** — rewriter passes turn dense matmuls/convolutions into
  `custom-call`s targeting NVIDIA libraries (e.g. `__cublas$lt$matmul`).
- **Triton** — for matmul-and-softmax fusions, XLA generates Triton that compiles
  to [PTX](cuda_ptx.md), enabling fused epilogues and fast softmax kernels.

**TPU** routes to closed-source **libtpu** via the **Mosaic** dialect; kernel
specifics stay proprietary.

---

## Distribution — GSPMD and Shardy

For multi-device execution, XLA uses **GSPMD** and its successor **Shardy**
(`sdy`). Users annotate array placement on **named device meshes** with
per-dimension mappings; the partitioner **propagates** consistent sharding and
**inserts collectives** (e.g. `all-reduce` for a sharded contracting dimension —
see [collective ops](collective_ops.md)).

Design principle: **distribution is a separate concern** from single-device
optimization — expressed as annotations, not baked into fusion/scheduling.

---

## Escape Hatch — Pallas and Helion

When HLO fusion isn't enough, [**Pallas**](pallas_kernels.md) (JAX's kernel
language) bypasses HLO fusion and layout assignment, compiling directly to device
kernels — **Triton on GPU, Mosaic on TPU** — as opaque `custom-call`s with launch
grids. **Helion** (PyTorch's kernel DSL) compiles to Pallas, so one kernel
definition can target both TPU and GPU through the same path.

---

## Runtime — PJRT and Caching

**PJRT** (`xla/pjrt/`) is XLA's hardware-agnostic runtime API. A PJRT client
represents a backend (CPU, GPU set, TPU slice) and manages devices, buffer
movement, and execution via `PjRtClient`, `PjRtDevice`, `PjRtBuffer`,
`PjRtLoadedExecutable`.

**Caching** amortizes compile cost — dominated by autotuning on accelerators:

- **In-process** — executables cached, keyed on input shape/dtype.
- **Persistent** — on-disk cache surviving process restarts.

The pattern: **compile once per shape, cache it, amortize over many executions.**

---

## Strengths and Limits

**Strengths.** Static shapes enable whole-program fusion, layout optimization, and
static buffer allocation with zero runtime-allocator overhead; clean separation of
single-device optimization from the distribution layer.

**Where it binds:**

- **Dynamic shapes** — bounded-dynamic tensors are padded to static bounds via
  `PadToStatic` / `SliceToDynamic`; data-dependent output shapes are rejected.
- **Shape-based compilation** — every new shape triggers a recompile; compile
  time (autotuning-dominated) is operationally costly.
- **Placement as annotation** — sharding and memory spaces are checked *late*,
  yielding silent performance cliffs rather than compile-time errors.

---

## See Also

- [CUDA PTX](cuda_ptx.md) — the PTX/SASS layer XLA's GPU backend targets
- [Pallas kernels](pallas_kernels.md) — the escape hatch below HLO fusion
- [What is a GPU kernel?](gpu_kernels.md) — fusion and memory-traffic fundamentals
- [GEMM](gemm.md) · [Attention](attention.md) — the `dot`/softmax fusions XLA specializes
- [Collective ops](collective_ops.md) — what the partitioner inserts
- [Roofline model](../modeling/roofline.md) — why fusion is the pass that matters most
