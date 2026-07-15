---
type: Concept
title: What Is a GPU Kernel? — Launches, Threads & Fusion
description: A GPU kernel is one program launched over data; PyTorch runs one kernel per op in eager mode, round-tripping intermediates through memory — fusion collapses them into one pass.
tags: [kernels, gpu, pytorch, fusion, torch-compile, triton, threads, warps, memory-traffic, profiling]
resource: https://x.com/MainzOnX/status/2077049378703892539
timestamp: 2026-07-14T00:00:00-07:00
---

# What Is a GPU Kernel? — Launches, Threads & Fusion

← [Workloads Index](index.md)

The on-ramp concept beneath every kernel-level page here: what a GPU kernel
actually *is*, why eager PyTorch runs one per op, and why fusing them into a
single kernel is the fundamental performance lever.

Source: [What even is a kernel?](https://x.com/MainzOnX/status/2077049378703892539)
(Adam Mainz, @MainzOnX — PyTorch/TPU @ Google, 2026), Article 1 of a series.

---

## A Kernel Is One Program the GPU Runs Over Data

Write `c = a + b` on GPU tensors and the CPU tells the GPU to **launch** a
**kernel**: one small program, ready to run over the data. (Not the OS kernel —
same word, different world.)

- **Launches are cheap** — microseconds each.
- The cost lives in the data movement *around* the launch: operands read from GPU
  memory, results written back.

**Mental model:** one PyTorch op → one launch → one kernel → **one pass over the data.**

### Threads and warps

Inside a kernel, work is done by tiny workers called **threads** — a GPU has
thousands. A length-8 add uses 8 threads (thread 0 handles element 0, etc.). In
practice the GPU launches threads in fixed-size groups called **warps** — always
32 threads on NVIDIA. None of this changes as tensors grow: same op, same one
kernel, just more threads.

---

## The Cost Is at the Edges: Memory Traffic

Whatever a kernel does inside its body, the **reads and writes at its edges** —
the trips out to GPU memory — are what dominate. Count them per op.

A length-N `add` by itself: reads `a`, reads `b`, writes `c` → **3 array-sized
transfers**.

### Two ops in eager mode

```python
c = (a + b).relu()   # an add, then a relu
```

In eager mode PyTorch runs each op *now* — it doesn't know a `.relu()` is coming
next. So the `add` writes a full intermediate `tmp` out to memory, and the `relu`
reads it back:

| Op | Traffic |
|----|---------|
| add | reads `a`, reads `b`, writes `tmp` (3) |
| relu | reads `tmp`, writes `c` (2) |

**5 array-sized transfers.** `tmp` never got to stay in a register or cache — it
made a full round trip through GPU memory. **Every intermediate value in eager
PyTorch is physically written out to memory and read back.**

---

## Fusion: Two Ops, One Kernel

Imagine one kernel that reads its element of `a`, reads its element of `b`, adds,
applies relu, and writes `c` — all before touching memory again:

- Reads `a`: 1, reads `b`: 1, writes `c`: 1 → **3 transfers.** Same math, two fewer
  round trips.

At length 8 nobody cares. At length 1M–100M, those saved trips dominate runtime.
**Fusion** is exactly this: combine ops that would be separate kernels into one so
the intermediate never hits memory. That's the whole idea. (*Why* memory traffic
dominates is the [roofline](../modeling/roofline.md) story — the article's Article 2.)

### `torch.compile` does the rewrite for you

```python
model = torch.compile(model)
```

`torch.compile` performs exactly this fusion automatically, emitting a single
fused **Triton** kernel. For cases it can't fuse (unrecognized custom ops, unusual
reductions), you drop to writing kernels by hand — e.g. in
[Pallas](pallas_kernels.md) or [Triton](pallas_kernels.md), or inline
[PTX](cuda_ptx.md).

---

## See It Yourself: Counting Kernels with `torch.profiler`

```python
import torch
from torch.profiler import profile, ProfilerActivity

def add_relu(a, b):
    return (a + b).relu()

a = torch.randn(1_000_000, device="cuda")
b = torch.randn(1_000_000, device="cuda")

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    add_relu(a, b)
torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

**Eager output — two rows, two kernels:**

```
vectorized_elementwise_kernel<...BinaryFunctor_add...>    1   12us
vectorized_elementwise_kernel<...threshold_kernel...>     1    9us
```

(relu often shows up as `threshold`; exact template names vary by PyTorch version.)

**Compiled** — wrap in `torch.compile`, run once to warm up (the first call is slow
because that's when compilation happens), then profile:

```
triton_poi_fused_add_relu_0    1   14us
```

**One row. One kernel.** The name spells out what it did: fused add and relu, one
launch instead of two. `torch.cuda.synchronize()` just ensures the GPU finished
before timings are read.

---

## Takeaways

- PyTorch code becomes a **sequence of kernels**; each kernel is one launch = one
  pass over its data.
- In **eager mode** you get roughly one kernel per op, with intermediates
  round-tripping through GPU memory.
- **Fusion** collapses adjacent ops into one kernel so intermediates stay on-chip —
  fewer memory trips, and memory traffic is what dominates at scale.
- **`torch.compile`** fuses automatically (via Triton); `torch.profiler` lets you
  *count the kernels* and verify.

---

## See Also

- [Roofline model](../modeling/roofline.md) — *why* memory traffic dominates (the AI cliff)
- [CUDA PTX](cuda_ptx.md) — the virtual ISA a fused Triton kernel lowers to
- [Pallas — custom kernels](pallas_kernels.md) — hand-writing fused kernels when the compiler can't
- [GEMM](gemm.md) — the workhorse operator and its arithmetic intensity
