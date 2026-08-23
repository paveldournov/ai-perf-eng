---
type: Concept
title: AI Chip Architectures — A Comparative Survey
description: Six accelerator architectures (NVIDIA GPU, Google TPU, AMD Instinct, Cerebras WSE, AWS Trainium, Groq LPU) read through one frame — where data lives, how it moves, what the compute units look like, and how chips talk at scale.
tags: [hardware, architectures, gpu, tpu, amd, cerebras, trainium, groq, dataflow, systolic-array, wafer-scale, scale-up, scale-out, comparison]
resource: https://www.jacobpeake.com/ai-chip-architectures
timestamp: 2026-08-23T00:00:00-07:00
---

# AI Chip Architectures — A Comparative Survey

← [Hardware Index](index.md)

Digest of **"AI Chip Architectures"** by **Jacob Peake**
([jacobpeake.com](https://www.jacobpeake.com/ai-chip-architectures)) — a survey of
the architectures that have won real deployment, each read through the same four
questions.

Where this knowledge base already has a deep-dive page ([H100](nvidia/h100.md),
[TPU v6e](tpu/tpu_v6e.md), [TPU v8](tpu/tpu_v8.md), [Boardfly](tpu/boardfly.md)),
this page links rather than repeats. Its job is the **cross-vendor comparison**
and the four architectures the KB had no page for at all: AMD Instinct, Cerebras
WSE, AWS Trainium, and the Groq LPU.

---

## The frame

Hennessy & Patterson's 2018 Turing Lecture, *A New Golden Age for Computer
Architecture*, predicted "a Cambrian explosion of novel computer architectures."
Their worked example was TPU v1: 29× the throughput of a CPU on neural-network
inference at 80× better energy efficiency. Single-threaded CPU performance had
grown 52%/year in the 1980s; by 2018, with Dennard scaling and Moore's Law over,
it was 3%.

The prediction landed. The architectures that have won real deployment: GPUs
(NVIDIA, AMD), systolic-array accelerators (TPU, Trainium), the Cerebras
Wafer-Scale Engine, and the Groq LPU.

### The problem all of them solve

AI compute is dominated by matrix multiplication — a transformer is a sequence of
matmuls (Q/K/V projection, attention, output projection, FFN) interleaved with
element-wise ops (normalisation, activation, residual adds). Training a frontier
model runs on the order of **10²⁵ multiply-accumulate operations**.

The *shape* of those matmuls is what varies, and it is the same split the
[LLM inference model](../modeling/llm_inference.md) is built on:

| Phase | Shape | Regime |
|---|---|---|
| Training | Many tokens against the same weight matrix | GEMM, high arithmetic intensity → compute-bound |
| Prefill | Full input sequence in one pass | GEMM → compute-bound |
| Decode | One token at a time; full pass over every weight + full KV read | **GEMV**, AI drops orders of magnitude → bandwidth-bound |

Inference systems claw intensity back by promoting GEMVs to GEMMs — continuous
batching, [speculative decoding](../modeling/speculative_decoding.md), multi-token
prediction. With continuous batching each user still reads their own KV cache, so
long-context decode shifts from **weight-bandwidth-bound to KV-bandwidth-bound**.

> The architecture problem is moving the numbers to where the matmuls happen fast
> enough. This is the **memory wall**: compute has scaled exponentially, memory
> bandwidth has not.

### The four questions

Understanding any of these chips reduces to:

1. **Where does data live?**
2. **How does it move to the compute units?**
3. **What do the compute units look like?**
4. **How do chips talk to each other at scale?**

Question 4 splits every time into **scale-up** (bind chips into one memory or
message domain — nanoseconds, huge bandwidth, bounded size) and **scale-out**
(network those domains — microseconds, explicit RDMA, tens of thousands of chips).
The partitioning consequence is universal: bandwidth-hungry collectives (tensor
parallelism, MoE expert routing) stay inside scale-up; data and pipeline
parallelism cross scale-out. See [parallelism](../modeling/parallelism.md) and
[collective algorithms](../workloads/collective_algorithms.md).

---

## The six architectures at a glance

| | Compute unit | Memory thesis | Scheduling | Scale-up domain |
|---|---|---|---|---|
| **[NVIDIA GPU](nvidia/index.md)** | Programmable SM + Tensor Core | HBM + hardware caches + explicit scratchpads | Hardware warp scheduler | NVLink/NVSwitch coherent rack (NVL72 → NVL576) |
| **[Google TPU](tpu/index.md)** | Systolic MXU + VPU | HBM + software-managed VMEM/CMEM, **no caches** | **Compiler** (VLIW) | ICI torus + optical circuit switch; superpod of 9,216+ |
| **AMD Instinct** | Conservative CU + Matrix Core | HBM + **256 MB Infinity Cache** | Hardware scheduler | 8-GPU box → 72-GPU Helios rack (2026) |
| **Cerebras WSE** | 900,000 dataflow cores, **no matrix unit** | **SRAM only** (44 GB on-wafer) | Data arrival | The wafer itself (fixed) |
| **AWS Trainium** | Systolic Tensor Engine + 3 sibling engines | HBM + software-managed SBUF/PSUM | **Compiler** (OpenXLA) | NeuronLink torus → NeuronSwitch (144 chips) |
| **Groq LPU** | Spatial functional slices | **SRAM only** (230 MB on-chip) | **Compiler, cycle-exact** | Switchless compiled Dragonfly |

Two axes organise the whole field:

- **Who schedules?** Hardware (NVIDIA, AMD) vs. compiler (TPU, Trainium, Groq)
  vs. data arrival (Cerebras). The compiler camp deletes warp schedulers, caches,
  and reorder buffers, and spends the silicon on MACs — with no fallback path when
  the schedule is wrong.
- **Where do weights live?** HBM (everyone conventional) vs. SRAM only (Cerebras,
  Groq). The SRAM camp buys enormous bandwidth-per-FLOP and pays in capacity,
  which sets chip count by *capacity*, not compute.

---

## NVIDIA GPU — programmability

Covered in depth at [H100](nvidia/h100.md) and [NVIDIA index](nvidia/index.md);
this section records what the blog adds beyond those pages.

**The bets:** (1) programmability — the workload is a moving target, so keep
everything programmable; (2) hide latency with massive multithreading (up to 64
resident warps/SM); (3) wrap the matmul in the warp abstraction rather than
exposing a fixed-function pipe, so one kernel can fuse matmul + softmax +
element-wise; (4) an explicit **async** memory hierarchy; (5) accept the SIMT tax
and amortise it across an ever-larger Tensor Core.

**The matrix instruction has been migrating off the threads for five generations** —
the single most useful throughline for reading modern kernel code:

| Gen | Instruction | Issuer | Operands |
|---|---|---|---|
| Volta | `mma.sync` (16×16×16) | 32-thread warp, **synchronous** | A, B, D in registers |
| Hopper | `wgmma.mma_async` (64×256×16) | 128-thread warp-group, async | B in SMEM descriptor; A optional |
| Blackwell | `tcgen05.mma` (256×256×16, two-SM) | **one thread**, async | A + B descriptors; D in **TMEM** |

With every operand off the lanes there is no per-thread state to coordinate, so a
single thread fires the instruction and returns; completion is signalled by an
`mbarrier`. **That decoupling is what makes attention kernels efficient** — the
warp runs softmax, applies a mask, or preloads the next tile while the matmul is in
flight. It is the structure of every modern attention kernel (see
[attention](../workloads/attention.md) and
[kernel optimization](../workloads/inference/kernel-optimization.md)).

The data-movement path decoupled on the same trajectory: synchronous per-thread
loads → Ampere `cp.async` (HBM→SMEM, bypassing registers) → Hopper **TMA** (a DMA
engine driven by one descriptor, with cluster-level multicast so one HBM read fans
out to every SM in a cluster) → Blackwell TMA direct-to-TMEM. The Hopper-era idiom
is **warp specialisation**: producer warps issue TMA loads, consumer warps fire
`wgmma`, handshaking through `mbarrier` rather than block-wide `__syncthreads()`.

**Memory tiers** (Blackwell adds a fifth): HBM → L2 (60 MB on B200, split as two
30 MB banks across the two dies with locality-aware residency) → 256 KB unified
L1/SMEM per SM → ~256 KB register file → **TMEM**, 256 KB per SM addressed only by
the Tensor Core, holding accumulators so they stop bleeding into the register file.
See [memory hierarchy](memory_hierarchy.md).

**Scaling.** NVL72 is 72 GPUs + 36 Grace CPUs as one coherent address space (13.5 TB
HBM + 17 TB LPDDR5X), held together by **5,184 passive copper cables** (~2 miles per
rack, SerDes on the GPU and switch ASICs, no in-cable retimers) carrying ~130 TB/s
all-to-all. NVIDIA estimates copper saves ~20 kW/rack against an optical equivalent.
Copper tops out around 1.5–2 m at 200 G/lane, which is why the rack boundary is
exactly where the fabric turns optical — and why Rubin Ultra's NVL576 reshapes the
chassis (Kyber, ~2× the height of Oberon) rather than lengthening cables. At the
network layer Rubin collapses pluggable optics into the switch package
(Quantum-X/Spectrum-X Photonics, co-packaged via TSMC COUPE): claimed ~4× fewer
lasers and ~3.5× lower link power.

**Software.** The moat is not CUDA itself — it is two decades of third-party
kernels and the developers who learned the API, most of it written by people
NVIDIA does not pay. NVIDIA also embeds engineers inside frontier labs, so
whatever a lab wants to train next month runs well on NVIDIA first. See
[CUDA PTX](../workloads/cuda_ptx.md) and [GPU kernels](../workloads/gpu_kernels.md).

---

## Google TPU — the compiler is the system

Covered at [TPU family](tpu/index.md), [v6e](tpu/tpu_v6e.md),
[v7x](tpu/tpu_v7x.md), [v8](tpu/tpu_v8.md), [Boardfly](tpu/boardfly.md). What the
blog adds:

**The bets:** systolic array; software scratchpads instead of caches; compiler
scheduling (VLIW, no speculation, no OoO); delete every transistor that does not
multiply-add; and carve out **dedicated off-array engines** for workloads the dense
array is the wrong shape for (SparseCore, CAE) rather than warping the main core.

**Inside the TensorCore:** one or more MXUs, a VPU, a Scalar Unit, an XLU for
cross-lane reductions, and a Transpose/Permute Unit — all on a single VLIW issue
plane. The **Scalar Unit** is the small block that matters most: single-threaded,
dual-issue, 32 registers, ~4 KiB SMEM, and the *only* block that fetches
instructions. Every cycle it pulls a **322-bit bundle** and dispatches eight slots —
2 scalar (local), 2 vector ALU, 2 vector load/store, 2 matrix push/pop. Sync flags
and compiler-inserted barriers replace hardware dependency tracking.

**The VPU is a 2D vector machine**, not 1D SIMD: VREGs are shaped (8, 128) on
v4/v5p — 128 lanes × 8 sublanes, 4 independent FP ALUs per (lane, sublane). The
lane axis matches the systolic array's input width; the sublane axis streams tiles
through the MXU. **Most of the speedup in modern TPU programs comes from VPU/MXU
overlap** — quantisation, layernorm, softmax, bias-add running in the same cycles
the MXU grinds behind them. Cross-lane reductions (the awkward case for a 2D ISA)
go to the XLU and are a known compiler hot spot.

**Weight-stationary dataflow** is the choice that distinguishes TPUs from
output-stationary arrays: weights preload one per cell, activations enter from the
left and propagate one column per cycle, partial sums flow down into accumulator
queues. Once data enters the array **no memory access occurs** — reuse is wired
into the silicon rather than arbitrated by a cache. The dominant cost in computing
is not the multiply (a few picojoules) but memory access at 100–1000× the energy.
The tax is **underfill**: a 128×128 matmul on a 256×256 array wastes 75% of the
silicon, so XLA pads and tiles to multiples of 128 (256 on v6e+) and model code is
written with those quanta in mind. See [GEMM](../workloads/gemm.md).

**SparseCore** exists because embedding lookup is the inverse of dense matmul —
irregular, indirect, all-to-all. It is a dataflow processor with 16 compute tiles
and SPMEM scratchpads delivering **5–7× on embedding-heavy models for ~5% of die
area and power**. Counts: 4 per chip on v4/v5p/Ironwood, 2 on Trillium, and **zero
on v8i**, which replaces it with the CAE on the I/O chiplet — same move, different
workload (collective reductions during decode).

**Optical circuit switching is the signature nobody else has.** Palomar OCS —
3D-MEMS mirrors that physically rotate — sits between 64-chip cubes; a v4 superpod
uses 48 of them to wire 4,096 chips into one 3D torus. Reconfiguration is
millisecond-class, which is fine because it is *circuit*-switched: pick a topology
at job start, run it for a week. Three problems collapse into one component:
per-workload topology (twisted tori give up to **70% better bisection**), sub-pod
slicing, and fault tolerance (a dead chip is optically swapped for a spare cube and
the run continues). Google uses the same primitive at every layer — Palomar at the
rack, Apollo in Jupiter at the datacenter spine (13 Pb/s bisection per building).

With v8t, scale-out **split into two fabrics**: **Virgo** for east-west TPU traffic
(flat two-layer non-blocking high-radix; one cluster links 134,000+ chips at 47
Pb/s bisection, 4× per-chip bandwidth and 40% lower unloaded latency than the prior
DCN generation) and **Jupiter** retained for north-south. Each layer can now evolve
on its own cadence.

**Software:** JAX → JAXpr → StableHLO → HLO → LLO → VLIW bundles, with XLA owning
fusion, layout assignment (harder than on 1D SIMD because both registers and
systolic inputs are 2D), buffer assignment across VMEM/CMEM/HBM, SPMD partitioning,
and bundle scheduling. GSPMD is being replaced by **Shardy** (MLIR-native, default
in early 2026); `shard_map` is the manual-SPMD escape hatch; [Pallas](../workloads/pallas_kernels.md)
is the kernel escape hatch via Mosaic. See [XLA compiler](../workloads/xla_compiler.md)
and [Pathways](../scheduling/pathways.md).

---

## AMD Instinct — the bet lives between the CUs

Where NVIDIA's ambition lives *inside* each SM, AMD's lives *between* the CUs — in
how many can be bonded into one coherent package. The CU has barely changed since
GCN (2012): four 16-lane SIMDs, one scalar unit, 64 KB LDS, an L1 vector cache, and
(since CDNA 1) a Matrix Core. What scales is the count — 120 CUs on MI100, 220 on
MI250X, 304 on MI300X, 256 on MI355X — and the packaging.

### Terminology map

| AMD | NVIDIA |
|---|---|
| Compute Unit (CU) | Streaming Multiprocessor (SM) |
| SIMD | SM Sub-Partition |
| SIMD Lane | CUDA Core (FP32 ALU) |
| Wavefront (wave64) | Warp (warp32) |
| Matrix Core | Tensor Core |
| MFMA | `mma.sync` / `wgmma` / `tcgen05.mma` |
| VGPR / SGPR | Register File |
| LDS (Local Data Share) | SMEM (Shared Memory) |
| Infinity Fabric | NVLink |

### The MFMA divergence

**AMD's Matrix Core stayed put while NVIDIA's climbed the thread hierarchy.** Every
MFMA generation from MI100 (2020) through MI355X (2025) is **wavefront-scoped**:
one wave64 issues `V_MFMA_*`, four SIMDs cooperate, operands come from VGPRs and the
dedicated AGPR file. The instruction got faster and the format set widened; the
issuer and the scope did not. Two costs follow:

- **Divergence** — a half-empty wave64 wastes 32 lanes where a half-empty warp32
  wastes 16.
- **Overlap** — AMD's wave-collective MFMA gives the issuing wave no way to do
  useful vector work while the matmul is pending. Overlap must be staged across
  separate wavefronts in software, with explicit barriers.

**How much this matters is workload-dependent**, and this is the sharpest analytical
point in the piece. Pure dense GEMM has nothing useful to do during the matmul —
both engines saturate, async buys little, and these are exactly the workloads where
AMD has led at exascale HPC (Frontier on MI250X, El Capitan on MI300A). Transformer
attention interleaves matmul with softmax, masking, and KV reads, and async overlap
*is* the structure of those kernels; AMD must recreate the pipeline by hand. MoE
dispatch, paged attention, and speculative decode sit in the same camp.

### Chiplets — where CDNA stops looking like NVIDIA

CDNA 3's MI300X 3D-stacks eight XCDs (TSMC N5, ~115 mm² each) onto four I/O dies
(N6) via **TSMC SoIC hybrid bonding** — sub-micron TSVs, no microbumps. The IODs
carry the Infinity Cache, HBM3 PHYs, Infinity Fabric, and PCIe Gen5, stitched at
4.8 TB/s bisection so the 153 B-transistor package looks like one GPU. **NVIDIA
stayed monolithic through H100 and only reached two dies on B200 via 2.5D CoWoS-L;
AMD got to 3D stacking a generation earlier.** MI355X moves XCDs to N3P (32 active
CUs each), collapses four IODs to two at 5.5 TB/s, and reaches 288 GB HBM3E at
8 TB/s in a 185 B-transistor package.

**MI300A** pushes further: replace 2 of the 8 XCDs with three Zen 4 CCDs and let CPU
and GPU share one physical address space backed by HBM3 with hardware coherence. No
host-device copy, no pinned memory, no PCIe in the path. Grace-Hopper bridges two
packages over NVLink-C2C; **MI300A is one package**. El Capitan (11,039 nodes of
4× MI300A) is the deployment that justified it.

**Infinity Cache** is the tier NVIDIA does not have: 256 MB on MI300X across four
IODs, ~12 TB/s measured — more than 2× MI300X's 5.3 TB/s of HBM3. It originated on
RDNA gaming GPUs to compensate for narrow GDDR buses, and attention KV reuse fits a
large LLC unusually well. **NVIDIA bet on bigger HBM bandwidth; AMD bet on the
cache.**

**The capacity bet** has a scaling consequence worth carrying into
[memory capacity](../modeling/memory_capacity.md) work: 8× MI300X hold 1.5 TB and
8× MI350X hold 2.3 TB, so a 405B model in FP8 fits inside a single 8-GPU box where
8× H100 (640 GB) requires careful sharding. For inference through 2024–25, AMD's
scale-up did not need to match NVL72 at the rack to compete at the box. For frontier
training it did, and **Helios** (2H 2026, 72 GPUs, ~31 TB HBM4, 2.9 EF FP4, 260 TB/s
scale-up) is the answer — shipping on Meta's Open Rack Wide rather than a
proprietary chassis, over **UALoE** (Infinity Fabric tunnelled over Ethernet) until
native UALink switching arrives in 2027.

**Software.** ROCm mirrors NVIDIA's library tier name-for-name (rocBLAS/cuBLAS,
MIOpen/cuDNN, RCCL/NCCL, Composable Kernel/CUTLASS) with **no first-party
TensorRT-LLM analog** — AMD backs vLLM plus AITER operators instead. Bulk HPC ports
at 80–95% via `hipify`; modern AI kernels port far worse, because anything reaching
for TMA descriptors, `wgmma`, or `tcgen05.mma` has no clean ROCm analog. The
strategic bet is that **Triton becomes the cross-vendor lingua franca**, so a kernel
going through `torch.compile` works on both without source change.

The honest gap, per the blog: independent benchmarks (Phoronix, March 2026) put
ROCm 7.2 at **10–25% slower** than equivalent CUDA on standard PyTorch/vLLM/SGLang
workloads at equivalent precision on equivalent silicon — feature parity, not
performance parity. FlashAttention is the load-bearing case: FA2 is production on
MI300X, FA3 partial, **FA4 (Blackwell, March 2026) has no ROCm port at all**.

---

## Cerebras WSE — the memory wall as a packaging choice

**The philosophy:** the memory wall is a consequence of cutting the wafer. A fab
prints dozens of dies and saws them apart; the industry then spends its most exotic
engineering (HBM, NVLink, CoWoS, 5,184 cables per rack) wiring the pieces back
together at a fraction of on-die bandwidth. Cerebras skips the saw.

**The wafer.** A stepper exposes ~850 mm² per shot — which is why every conventional
chip lives under that ceiling, and why B200 became two dies the moment NVIDIA
pressed against it. Cerebras prints the same ~550 mm² die **84 times in a 12×7 grid**
and, in a process co-developed with TSMC, lays extra high-level metal across the
<1 mm scribe lines where the saw would run. The mesh crosses each seam at 2,880 GB/s
per die on WSE-3, and the entire inter-die layer costs ~97 W. To software the seams
do not exist.

Wafer-scale failed on yield in the 1980s. **Cerebras's answer is granularity:** a
defect on an H100 disables an entire ~6 mm² SM; the same defect on a WSE disables one
**0.05 mm²** core. WSE-3 fabricates ~970,000 cores and ships 900,000 — a ~7% spare
pool plus redundant links lets the hardware remap around every defect.

**No hierarchy, no matrix unit.** 900,000 identical cores tile edge-to-edge in a flat
2D mesh with no shared cache, no global memory, and no boundary of any kind. Each core
is ~38,000 µm², peaks at 30 mW, and holds 48 kB of SRAM, 16 registers, a six-stage
pipeline, a 4-wide FP16 FMAC (8-wide on WSE-3), and a five-port router. Execution is
**dataflow**: a core idles until a wavelet arrives, control bits select which handler
fires, and eight hardware microthreads switch cycle-by-cycle. *The arrival of data is
the schedule.*

There is no matmul engine — GEMM is assembled from the fabric. Each arriving weight
broadcasts along a row of cores holding activations, every core fires an AXPY against
its resident slice, and partial sums reduce across the mesh. **Activations never move**;
the only operand in flight is the one being multiplied. The reuse a Tensor Core gets
from a register tile and an MXU gets from its wiring, the WSE gets from geometry.

**The bytes-per-FLOP argument is the one to take away.** 44 GB of SRAM at an
aggregate 21 PB/s feeds **~1.3 bytes per dense FP16 FLOP, where a B200 gets ~0.002
from HBM**. On that axis every GPU and TPU is starved and the WSE is the only machine
in balance — which is exactly why decode, a pure bandwidth problem, is the phase it is
shaped for. Read this against the [roofline model](../modeling/roofline.md): the WSE
is a machine built to sit on the other side of the ridge point.

Three cautions the blog is careful about, all worth carrying into any comparison:

- **The headline FLOPs are sparse.** WSE-3's 125 PFLOPS assumes ~8× zero-skipping;
  dense is roughly **15.8 PFLOPS FP16** (derived — Cerebras publishes no dense
  figure). Per watt, dense FLOPs on the wafer lose to every contemporary GPU. *The
  wafer was never a FLOPs machine; the FLOPs exist to keep up with the SRAM.*
- **21 PB/s is an on-wafer aggregate** — the sum of 900,000 local SRAM ports, not a
  point-to-point link, and not comparable to an HBM figure.
- **The island has a cliff at its edge.** The wafer's only external connection is
  12×100 GbE (1.2 Tb/s) — barely more than the single ConnectX-8 NIC on one Blackwell
  GPU. Five orders of magnitude separate on-wafer SRAM from off-wafer Ethernet.

**Weight streaming inverts the usual flow:** on a GPU or TPU weights are resident and
activations stream; on a WSE **activations are resident and weights stream** from
MemoryX, with the optimizer step running on MemoryX CPUs (a weight update is
O(parameters) of element-wise work with no reuse, so CPU-class compute keeps pace).
What this buys is the programming model — one wafer holds a full layer's activations,
so there is no tensor parallelism, no pipeline parallelism, no FSDP sharding. **The
parallelism-strategy spreadsheet that dominates GPU training has no Cerebras page.**
Multi-system scaling is pure data parallelism through SwarmX.

**Two structural limits.** SRAM density has effectively stopped scaling: WSE-3 carries
just **10% more SRAM than WSE-2** despite a full node shrink and a 54% transistor
jump — the architecture's scarcest resource is the one thing the next node no longer
buys. And the scale-up domain is a *constant*: 46,225 mm² since 2019, with 300 mm the
largest wafer the industry runs. NVIDIA's scale-up domain grows every generation; the
wafer's cannot.

**Where it wins:** batch-one decode speed, independently verified. Artificial Analysis
measured 1,850 tok/s on Llama 3.1 8B, 446 on 70B, 969 on Llama 405B, and 2,522 on
Llama 4 Maverick (~2.4× the best published Blackwell number of the time). **Where it
does not:** per-token API pricing runs ~3–5× GPU providers; a frontier-class model
consumes fleets (SemiAnalysis estimates ~24 CS-3s for a 1.6T-class model that fits in
a handful of GPU racks); KV cache competes with weights for the same 44 GB, capping
context at 131K where frontier providers serve 256K–1M; and **MFU has never been
disclosed for any Cerebras run**. The largest disclosed cluster is 64 systems against a
2,048 spec, and the largest from-scratch model is Jais 2 at 70B — nothing above 70B, from
anyone, in seven years.

**Software:** the compiler is a *kernel matcher*, not a general code generator, and
there is **no user kernel path at all** — CUDA's answer to a novel attention variant
is write a kernel, the TPU's is Pallas, ROCm's is Triton; Cerebras's is a Cerebras
engineer. There is a strange immunity in this: FlashAttention is a scheme for tiling
attention through a memory hierarchy, and the WSE has no hierarchy to tile against, so
the optimisation class that costs AMD years of porting lag simply does not apply. But
the immunity and the poverty are the same fact.

---

## AWS Trainium — the TPU thesis inside a different cloud

Annapurna Labs built Trainium as a **fast-follower**: the compute core takes the TPU's
playbook (128×128 weight-stationary systolic array, software-managed scratchpads,
whole-program compilation) down to **sharing Google's XLA compiler outright**. What is
genuinely Amazon's is narrow and deliberate — dedicated collective-communication
silicon, and the vertical integration to price a chip that only has to beat NVIDIA
*inside AWS*.

**The unit is assembled differently from a TPU.** A chip carries a few NeuronCores (2
on Trn1, 8 on Trn2/Trn3), and each NeuronCore is not one monolithic engine but a
cluster of four decoupled specialists: a **Tensor Engine** (the 128×128 array, 16,384
MACs), a **Vector Engine** (reductions — layernorm, softmax, pooling), a **Scalar
Engine** (pointwise), and a programmable **GPSIMD Engine** (eight 512-bit vector
processors running C) for whatever fits none of the others. Around them sit 128 DMA
engines, a Sync Engine, and CC-Cores. A well-compiled step overlaps all four — the same
producer/consumer overlap that makes TPU and GPU attention kernels efficient, expressed
as separate *physical engines* rather than warps or VLIW slots. The tax is at the edges:
an operator fitting none of the specialists falls to the slower GPSIMD path, which is
the part most likely to bottleneck a novel architecture.

**The array is physically fixed at 128×128 across all three generations**; what changes
is how many products it packs per cell. Trn1 ran BF16/FP16 with FP8 at *no* speedup;
Trn2 double-pumps FP8 to an effective 256×128; Trn3 packs microscaling operands to an
effective 512×128 at 4× the BF16 rate. *The count of physical MAC cells never moves;
the datapath just feeds them narrower numbers.*

**Memory** is the TPU's bet restated: three tiers, all software-managed, no hardware
cache anywhere ("all memory movement is explicit in the program itself"). HBM → **SBUF**
(the main scratchpad, ~20× HBM bandwidth, 128 partitions, 24/28/32 MiB per NeuronCore
across v2/v3/v4) → Tensor Engine → **PSUM** (2 MiB accumulator) → SBUF.

**Numerics wrinkles worth knowing:** FP8 is *configurable* (adjustable exponent bias
across E5M2/E4M3/E3M4, so the compiler trades range for precision per tensor), and
**Trn3's FP4 buys no extra throughput** — MXFP4 operands are up-converted to MXFP8
before reaching the array, so FP4 saves memory and bandwidth, not compute. The blog
also flags that AWS headlines a 4× FP8 figure its own architecture pages put at 2× over
dense FP8.

**Collectives in silicon** is the one block with no clean GPU analogue. On a GPU,
collectives run as NCCL kernels on the same SMs doing the math, so communication and
compute contend for the same silicon. Trainium carves it out: **20 CC-Cores per Trn2
chip**, wired straight to the NeuronLink ports, running all-reduce/all-gather/
reduce-scatter/all-to-all while the Tensor and Vector engines keep going. *Communication
becomes something the chip does concurrently, not something it pauses to do.* Same move
as SparseCore and the CAE. See [collective ops](../workloads/collective_ops.md).

**The interconnect roadmap is a compressed re-run of the industry's:** adopt the torus
while the workload is nearest-neighbour, switch to a crossbar when it is not. Trn2 is a
4×4×4 torus of 64 chips with a deliberately thin third axis (~256 GB/s inter-instance
vs 1.28 TB/s inside); Trn3's **NeuronSwitch** flattens the diameter so any chip reaches
any other in one switched hop, scaling to 144 chips. **The motivation is the same one
that pushed Google to Boardfly** — expert routing is all-to-all, the worst case for a
torus. Scale-out is not bespoke: EFA with the **SRD** transport, which sprays each
message across up to 64 paths and delivers reliably but *out-of-order*, pushing
reassembly up to the collective library to sidestep head-of-line blocking.

**Project Rainier** — ~500,000 Trainium2 chips across multiple US datacenters for
Anthropic in late 2025; by early 2026 Claude was running on **over a million chips**,
the largest commitment any external lab has made to a non-NVIDIA training platform. AWS
claims 30–40% better price-performance than its Hopper-class instances (measured against
last-generation NVIDIA, not Blackwell).

**Software:** `neuronx-cc` ingests XLA HLO and emits a NEFF binary; OpenXLA lists
Trainium as a first-class PJRT device alongside the TPU. **NKI** (Neuron Kernel
Interface) is the tile-level escape hatch — Trainium's Pallas. The tell about maturity is
how the anchor tenant works: Anthropic does not simply target Trainium through PyTorch,
it embeds with Annapurna, writes its own NKI kernels, and upstreams fixes. *Trainium is
production-viable at the frontier, but at the frontier it is co-engineered, not turnkey.*

---

## Groq LPU — determinism

**Every other chip spends silicon tolerating uncertainty** — caches to hide memory
latency, schedulers to fill stalls, arbiters to resolve contention. The LPU deletes all
of it: no cache, no branch predictor, no arbiter, no reorder buffer, not even an on-chip
crossbar. What is left is a chip whose latency is known before it runs. Founded by
Jonathan Ross, who started Google's TPU as a 20% project.

**Built inside-out from the rest of the field.** Everyone else replicates a core and
farms work to the copies; the LPU takes a *single* conventional core and pulls it apart.
Instruction control, vector ALUs, matrix units, memory, and network each become a
**functional slice** — a full-height column — standing side by side across the die.
Homogeneous down each slice, heterogeneous across the chip. Data streams *horizontally*
through the slices like parts down an assembly line (East and West, one register hop per
cycle) while VLIW instructions issue *Northward* to meet it. The chip is 320 lanes tall
(20 superlanes × 16, plus a spare fused out for yield), with 64 stream registers per lane.

- **MXM:** four independent 320×320 MAC planes, 409,600 multipliers, INT8/FP16 into
  INT32/FP32. ~750 INT8 TOPS and 188 FP16 TFLOPS at 900 MHz — and **the number carries
  no sparsity asterisk, because a data-dependent skip would make execution time
  data-dependent**, and determinism is the one property it will not trade.
- **VXM:** 16 ALUs per lane in a 4×4 mesh (5,120 32-bit ALUs). Because compute is
  spatial, an operand can march through a chain of VXM ALUs straight into an MXM plane
  on consecutive cycles without touching memory — **the operator fusion a GPU kernel
  builds by hand is here just the physical order of the slices**.
- **SXM:** lane shifts, 320-lane permute, transposes, and the chip-to-chip links.

**Determinism as a design output:** the ISA carries each instruction's execution latency,
datapaths are fixed-latency by construction, and the compiler computes the exact cycle
every result appears. Groq's proof: **24,240 runs of BERT-Large inside a ~75 µs band,
with predicted latency within 2% of measured.** This is the TPU's instinct taken one step
further — the TPU compiler schedules a chip, the LPU compiler schedules a *system*. And
it is the exact inverse of Cerebras: **the WSE reacts to data, the LPU is timed to it.
Both delete the scheduler; one replaces it with arrival, the other with a clock.**

**The capacity constraint defines everything.** 230 MB of SRAM, no HBM, no DRAM. That
does not hold a model — Llama-2 70B in FP16 is 140 GB, so the deployed configuration was
**~576 LPUs**. The chip count is set by *capacity, not compute*. Same trade as Cerebras
from the opposite direction: Cerebras keeps one enormous die and gives up capacity per
wafer; Groq keeps a normal die and gives up ever fitting a model on one.

**Scaling has no separate fabric because the chip is already a switch** — up to 16
chip-to-chip RealScale links per LPU, acting simultaneously as compute endpoint and
router. A node is 8 LPUs fully connected (7 links each) presenting a 32-port virtual
router; nodes wire into a compiled **Dragonfly**, 9 per rack (72 chips, one a hot spare),
spec'd to 10,440 chips within six hops. The fabric is *scheduled, not routed*: no
back-pressure, no dynamic arbitration (the compiler has proven the receiver is ready),
and forward error correction instead of retransmission, **because a retry would perturb
the schedule**. Keeping independently-clocked chips in lockstep needs Hardware-Aligned
Counters exchanged every 256 cycles with periodic deskew.

**The economics are the sharp edge.** A model replica is a rack or eight: ~576 chips for
Llama-2 70B carried 144 host CPUs and 144 TB of host RAM, against two CPUs for an 8-GPU
server. SemiAnalysis: the LPU wins bill-of-materials per token when you optimise for
latency, and **loses to GPUs by roughly an order of magnitude on throughput per dollar
once you batch**. Software is the purest form of *the compiler is the machine* — there
are no kernels at all; Groq brought up LLaMA in four days with under ten people.

**The epilogue:** in December 2025 NVIDIA took a non-exclusive license to the LPU
technology and hired Ross and much of the team (~$13B at closing; not an acquisition —
no products, contracts, or equity changed hands, per NVIDIA's 10-K). At GTC 2026 it
reappeared as the **NVIDIA Groq 3 LPU**, a rack of 256 SRAM-only chips beside Rubin
NVL72, splitting the transformer between them: **GPUs run attention, LPUs run the
feed-forward and MoE layers**, with Dynamo orchestrating the hand-off. That is
attention-FFN disaggregation as a hardware boundary — the same logic as
[prefill/decode disaggregation](../workloads/inference/optimization.md#prefill-decode-disaggregation),
one level down. *The most deterministic architecture in AI ended up as a latency
co-processor inside the most programmable one.*

---

## Cross-cutting patterns

Five patterns recur across all six architectures, and they are the transferable part:

1. **Carve out an off-core engine rather than warping the main one.** Google's
   SparseCore (embeddings) and CAE (collectives), Trainium's CC-Cores, Cerebras's
   sender-side zero filter. Same move every time: find a workload the main engine is the
   wrong shape for, and spend a little area beside it.
2. **The matmul instruction keeps shedding its issuer.** Warp → warp-group → single
   thread on NVIDIA; a VLIW slot on TPU; two explicit instructions against a named
   scratchpad on Trainium; a tensor descriptor (DSR) on Cerebras, where a tensor
   instruction *has no other form*. AMD is the one that stayed put, and pays for it in
   attention-shaped workloads.
3. **Precision halves every generation, with finer-grained scaling buying the accuracy
   back.** FP32 → FP16 → FP8 → FP4, now converging on the **OCP MX** block-scale formats
   — the format in MI355X is identical to B200's and TPU v8's. The two exceptions are
   the two SRAM-only machines: **Cerebras and Groq both stopped at 16-bit**, despite
   being the architectures most starved for capacity, where 8-bit weights would halve
   the chip count. Conviction or a datapath roadmap gap is an open question in both cases.
4. **Topology follows the collective.** A torus is optimal when collectives are
   nearest-neighbour (ring all-reduce uses every link every cycle) and worst-case when
   they are all-to-all. **MoE expert routing is all-to-all**, and three vendors
   independently made the same move: Google (torus → Boardfly on v8i, 16-hop diameter →
   7), AWS (torus → NeuronSwitch on Trn3), NVIDIA (crossbar from the start). See
   [MoE](../workloads/moe.md) and [collective algorithms](../workloads/collective_algorithms.md).
5. **Copper sets the rack boundary.** Passive DAC tops out at ~1.5–2 m at 200 G/lane. That
   single physical fact explains NVL72's 5,184-cable spine, why NVL576 needed a taller
   chassis rather than longer cables, and why co-packaged optics is arriving at the switch
   ASIC now.

---

## Comparison tables

Reproduced from the blog. **Read the caveats first** — these numbers are not
apples-to-apples:

- All figures are **peak** at the stated precision, dense unless the vendor publishes no
  dense basis.
- **Memory bandwidth is the native tier**: HBM for GPUs/TPUs/Trainium, **aggregate
  on-chip SRAM** for Cerebras and Groq. *An on-wafer aggregate is not comparable to an
  HBM figure.*
- **Scale-up bandwidth follows each vendor's convention** and may mean per-chip
  aggregate, rack aggregate, or true bisection.
- `*` marks analyst-derived, era-inferred, or vendor-aggregate-derived figures; `n/d`
  marks undisclosed specs.

### Per-chip

| Year | Chip | Memory | Memory BW | Flagship dense FLOPs | TDP |
|---|---|---|---|---|---|
| 2023 | H100 SXM5 | 80 GB HBM3 | 3.4 TB/s | 1.98 PF FP8 | 700 W |
| 2024 | H200 SXM | 141 GB HBM3e | 4.8 TB/s | 1.98 PF FP8 | 700 W |
| 2024 | B200 | 192 GB HBM3e | 8 TB/s | 4.5 PF FP8 / 9 PF FP4 | 1,000 W |
| 2025 | B300 | 288 GB HBM3e | 8 TB/s | 7.5 PF FP8 / 15 PF FP4 | 1,400 W |
| 2026 | Rubin | 288 GB HBM4\* | ~13 TB/s\* | ~17 PF FP8\* / ~50 PF FP4\* | ~1,500 W\* |
| 2027 | Rubin Ultra | 1 TB HBM4e\* | ~32 TB/s\* | ~33 PF FP8\* / ~100 PF FP4\* | ~1,800 W\* |
| 2023 | TPU v5p | 95 GB HBM2e | 2.8 TB/s | 0.46 PF BF16 | n/d |
| 2025 | TPU Ironwood (v7) | 192 GB HBM3e | 7.4 TB/s | 4.6 PF FP8 | n/d |
| 2026 | TPU v8t (Sunfish) | 216 GB HBM3e | 6.5 TB/s | 12.6 PF FP4 | n/d |
| 2023 | MI300X | 192 GB HBM3 | 5.3 TB/s | 2.6 PF FP8 | 750 W |
| 2024 | MI325X | 256 GB HBM3e | 6.0 TB/s | 2.6 PF FP8 | 1,000 W |
| 2025 | MI355X | 288 GB HBM3e | 8 TB/s | 10 PF FP8 / 20 PF FP4 | 1,400 W |
| 2026 | MI455X | TBD | TBD | ~40 PF FP4\* | TBD |
| 2021 | WSE-2 | 40 GB SRAM (on-wafer) | 20 PB/s (aggregate) | 7.5 PF FP16 | 23 kW (system) |
| 2024 | WSE-3 | 44 GB SRAM (on-wafer) | 21 PB/s (aggregate) | ~15.8 PF FP16\* | 23 kW (system) |
| 2022 | Trainium1 | 32 GB HBM2e\* | 820 GB/s | 0.19 PF BF16/FP8 | n/d |
| 2024 | Trainium2 | 96 GB HBM3 | 2.9 TB/s | 1.3 PF FP8 | ~500 W\* |
| 2025 | Trainium3 | 144 GB HBM3e | 4.9 TB/s | 2.5 PF FP8 | n/d |
| 2020 | GroqChip (TSP/LPU) | 230 MB SRAM | 80 TB/s (on-chip aggregate) | 0.188 PF FP16 | 215 W |
| 2026 | NVIDIA Groq 3 LP30 | 500 MB SRAM | 150 TB/s (on-chip aggregate) | ~1.2 PF FP8\* | n/d |

### Per-rack / pod

| Year | System | Chips | Aggregate dense FLOPs | Accelerator memory |
|---|---|---|---|---|
| 2023 | HGX H100 | 8 | 16 PF FP8 | 640 GB |
| 2024 | HGX H200 | 8 | 16 PF FP8 | 1.1 TB |
| 2024 | GB200 NVL72 | 72 | 360 PF FP8 / 720 PF FP4 | 13.4 TB |
| 2025 | GB300 NVL72 | 72 | 540 PF FP8 / 1,100 PF FP4 | 20.7 TB |
| 2026 | NVL144 | 144 | ~1.2 EF FP8 / ~3.6 EF FP4 | ~21 TB |
| 2027 | NVL576 (Kyber) | 576 | ~5 EF FP8 / ~15 EF FP4 | ~144 TB |
| 2023 | TPU v5p pod | 8,960 | 4.1 EF BF16 | 852 TB |
| 2025 | TPU Ironwood pod | 9,216 | 42.5 EF FP8 | 1.77 PB |
| 2026 | TPU v8t (Sunfish) pod | 9,600 | 121 EF FP4 | ~2 PB |
| 2023 | MI300X 8-GPU OAM | 8 | 21 PF FP8 | 1.5 TB |
| 2024 | MI325X 8-GPU OAM | 8 | 21 PF FP8 | 2.0 TB |
| 2025 | MI355X 8-GPU OAM | 8 | 80 PF FP8 / 160 PF FP4 | 2.3 TB |
| 2026 | Helios (MI455X) | 72 | 1.4 EF FP8 / 2.9 EF FP4 | 31 TB |
| 2024 | Condor Galaxy 3 | 64 wafers | ~1 EF FP16\* | 2.8 TB SRAM + MemoryX |
| 2022 | Trn1 instance | 16 | 3 PF BF16 | 512 GB |
| 2024 | Trn2 UltraServer | 64 | 83 PF FP8 | 6.1 TB |
| 2025 | Trn3 UltraServer | 144 | 362 PF FP8 | 20.7 TB |
| 2022 | GroqRack | 64 active (72 installed) | 12 PF FP16 | 14 GB SRAM |
| 2026 | NVIDIA Groq 3 LPX | 256 | 315 PF FP8 | 128 GB SRAM + 12 TB DDR5 |

### What the tables show

- **Per-chip FP8 has converged.** B200 (4.5 PF), Ironwood (4.6 PF), and MI355X (10 PF)
  sit within ~2×. The per-chip arms race is close; **the rack and pod are where the
  architectures diverge.**
- **HBM capacity is AMD's persistent win** — 192 → 256 → 288 GB across 2023–25, matching
  or beating NVIDIA every generation. NVIDIA caught up at 288 GB only with B300.
- **Rack-scale scale-up was NVIDIA's win until 2026.** NVL72 was the only coherent
  rack-scale domain shipping in 2024–25; AMD scaled up at the *box* and didn't reach the
  rack until Helios. The TPU sidesteps the question — its torus is the rack and the
  cluster at once.
- **TPU pods dwarf any NVIDIA rack in chip count.** Ironwood pod = 9,216 chips for 42.5
  EF FP8; NVL576 = 576 GPUs for ~5 EF FP8. More aggregate compute per system, at the cost
  of per-chip bandwidth.
- **Power per chip is rising fast:** 700 W (Hopper) → 1,000 W (Blackwell, MI325X) → 1,400 W
  (B300, MI355X) → ~1,800 W (Rubin Ultra, analyst). **Liquid cooling becomes mandatory
  above ~1,000 W; air cooling effectively ends with Hopper.**
- **Cerebras breaks the table's axes:** ~1.3 bytes per dense FLOP where the GPU rows sit
  near 0.002 — at the cost of less total memory than a single H200 and an empty scale-up
  column, because the coherent domain *is* the wafer.
- **Trainium competes on economics, not the spec sheet.** Trn2's 1.3 PF FP8 is roughly a
  quarter of MI355X, but AWS owns every layer from the Nitro card to the API.

---

## See Also

- [Roofline parameters by chip](roofline_params.md) — per-device ridge points
- [Memory hierarchy](memory_hierarchy.md) — the tiers these architectures rearrange
- [H100](nvidia/h100.md) · [TPU family](tpu/index.md) · [TPU v8](tpu/tpu_v8.md) · [Boardfly](tpu/boardfly.md) · [Sohu](etched/sohu.md) · [Apple ANE](apple/ane.md)
- [Roofline model](../modeling/roofline.md) — the frame the bytes-per-FLOP argument uses
- [Parallelism](../modeling/parallelism.md) — what the scale-up/scale-out split dictates
- [MoE efficiency](../workloads/moe.md) — why all-to-all reshaped three interconnect roadmaps
- [XLA compiler](../workloads/xla_compiler.md) · [Pallas](../workloads/pallas_kernels.md) · [CUDA PTX](../workloads/cuda_ptx.md)
- [Pathways](../scheduling/pathways.md) — the single-controller runtime above TPU pods
