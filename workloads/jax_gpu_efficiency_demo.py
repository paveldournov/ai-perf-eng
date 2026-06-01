#!/usr/bin/env python3
"""
JAX GPU Efficiency Demo
=======================
Same algorithm, two different implementation choices → dramatic performance difference.

Demonstrates two independent axes of GPU efficiency:

  Part 1 — Core utilization  : Python-loop dispatch vs single batched kernel
  Part 2 — Hardware path     : float32 (CUDA cores) vs bfloat16 (Tensor Cores)

In both cases the numerical result is identical (within floating-point precision).
The ONLY change is one parameter per experiment.

────────────────────────────────────────────────────────────────────────────────
Hardware target : NVIDIA RTX 30xx / 40xx
Runtime target  : JAX + CUDA on WSL2

Install (WSL2, CUDA 12):
    pip install -U "jax[cuda12]"

Run:
    python jax_gpu_efficiency_demo.py
────────────────────────────────────────────────────────────────────────────────
"""

import os
import time

import jax
import jax.profiler
import jax.numpy as jnp
import numpy as np


# ─── timing helpers ──────────────────────────────────────────────────────────

def benchmark(fn, *args, warmup: int = 5, runs: int = 20) -> float:
    """Return median wall-clock time (ms) over `runs` calls after `warmup`."""
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1e3   # → ms

def tflops(flop_count: int, ms: float) -> float:
    return flop_count / (ms * 1e-3) / 1e12

def row(label: str, ms: float, flops: int) -> None:
    print(f"  {label:<46s}  {ms:7.2f} ms   {tflops(flops, ms):.2f} TFLOPS")

# ─── profiling helpers ───────────────────────────────────────────────────────
#
# Two trace formats, same XLA data, different viewers:
#
#  profile_run()       → Perfetto (.json.gz)  drag into https://ui.perfetto.dev
#  profile_run_xprof() → XPlane  (.pb)        open with:  xprof --logdir=<dir>
#                                               install:   pip install xprof
#
# Both helpers run one warmup call first so compilation is excluded from the
# captured window.  Traces land under /mnt/d/... so Windows Explorer can reach
# them directly without copying out of WSL2.
#
# What each viewer shows that the other doesn't:
#  Perfetto  — fine-grained timeline, CPU↔GPU async gaps, Python dispatch stalls
#  xprof     — op-level breakdown, memory bandwidth utilisation, roofline chart,
#              kernel occupancy, input pipeline analysis

TRACE_ROOT       = "/mnt/d/Projects/AI-Perf-Sim/workloads/traces/perfetto"
XPROF_TRACE_ROOT = "/mnt/d/Projects/AI-Perf-Sim/workloads/traces/xprof"

def _slug(label: str) -> str:
    return label.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "").replace("/", "-")

def profile_run(label: str, fn, *args) -> None:
    trace_dir = os.path.join(TRACE_ROOT, _slug(label))
    os.makedirs(trace_dir, exist_ok=True)
    jax.block_until_ready(fn(*args))                    # warmup — skip compile
    with jax.profiler.trace(trace_dir, create_perfetto_trace=True):
        jax.block_until_ready(fn(*args))
    print(f"  perfetto trace → {trace_dir}")

def profile_run_xprof(label: str, fn, *args) -> None:
    # XPlane .pb files — xprof reads these natively; no Perfetto conversion needed
    trace_dir = os.path.join(XPROF_TRACE_ROOT, _slug(label))
    os.makedirs(trace_dir, exist_ok=True)
    jax.block_until_ready(fn(*args))                    # warmup — skip compile
    with jax.profiler.trace(trace_dir):
        jax.block_until_ready(fn(*args))
    print(f"  xprof trace    → {trace_dir}")


# ─── prologue ────────────────────────────────────────────────────────────────

print("=" * 70)
print("JAX GPU Efficiency Demo")
print("=" * 70)
print(f"Devices : {jax.devices()}")
print()

key = jax.random.PRNGKey(42)


# ════════════════════════════════════════════════════════════════════════════
# PART 1 — Core utilization
# ────────────────────────────────────────────────────────────────────────────
#
# Algorithm : sum of (Aᵢ @ Bᵢ) over a batch of BATCH matrix pairs
#
# Variant A  Python loop — each iteration dispatches a separate GPU kernel.
#             Problems:
#               • ~50 µs Python overhead per dispatch call
#               • Each kernel is tiny: most SMs sit idle between launches
#               • No opportunity for inter-batch fusion or wave quantisation
#
# Variant B  Single batched einsum — XLA compiles one kernel that sees the
#             full BATCH dimension.  All SMs stay busy; no repeated dispatch
#             overhead; the scheduler fills the GPU completely.
#
# The ONLY change: whether the loop is in Python or inside jit.
# ════════════════════════════════════════════════════════════════════════════

BATCH = 512
N     = 64      # intentionally small so kernel-launch overhead dominates in A

A_batch = jax.random.normal(key, (BATCH, N, N), dtype=jnp.float32)
B_batch = jax.random.normal(key, (BATCH, N, N), dtype=jnp.float32)

# Variant A: per-sample jit, called BATCH times from Python.
# jax.jit compiles _single once, but each Python-level call is a separate
# async dispatch — the GPU processes 512 tiny back-to-back kernel launches.
@jax.jit
def _single_matmul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return (a @ b).sum()

def serial_loop(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    acc = jnp.zeros(())
    for i in range(A.shape[0]):         # Python for-loop → 512 kernel launches
        acc = acc + _single_matmul(A[i], B[i])
    return acc

# Variant B: single batched call inside jit.
# XLA sees the full (BATCH, N, N) problem and emits one optimised kernel.
@jax.jit
def batched_matmul(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum("bij,bjk->bik", A, B).sum()

flops_p1 = 2 * BATCH * N**3

print(f"PART 1 — Core utilization  (batch={BATCH} × {N}×{N} matmuls)")
print()
print(f"  {'label':<46s}  {'time':>7s}   {'throughput':>10s}")
print(f"  {'-'*46}  {'-'*7}   {'-'*10}")
t_serial  = benchmark(serial_loop,    A_batch, B_batch)
t_batched = benchmark(batched_matmul, A_batch, B_batch)
row("A: Python loop   (512 separate kernels)", t_serial,  flops_p1)
row("B: batched einsum (1 kernel)",            t_batched, flops_p1)
print()
print(f"  → Speedup  B / A : {t_serial / t_batched:.1f}×")

# Numerical check
out_serial  = serial_loop(A_batch, B_batch)
out_batched = batched_matmul(A_batch, B_batch)
err_p1 = float(jnp.abs(out_serial - out_batched) / jnp.abs(out_serial))
print(f"  → Relative error : {err_p1:.2e}  (should be ~0 — identical fp32 ops)")

# Profile one steady-state execution of each variant
print()
print("  Collecting traces (Part 1) ...")
profile_run("p1_A_serial_loop",    serial_loop,    A_batch, B_batch)
profile_run("p1_B_batched_einsum", batched_matmul, A_batch, B_batch)
profile_run_xprof("p1_A_serial_loop",    serial_loop,    A_batch, B_batch)
profile_run_xprof("p1_B_batched_einsum", batched_matmul, A_batch, B_batch)


# ════════════════════════════════════════════════════════════════════════════
# PART 2 — Tensor core utilization
# ────────────────────────────────────────────────────────────────────────────
#
# Algorithm : large square matrix multiplication  C = A @ B,  A,B ∈ ℝ^{M×M}
#
# Each SM (Streaming Multiprocessor) on an NVIDIA GPU contains two physically
# distinct compute units:
#
#   CUDA cores  — general-purpose FP32 FMA units.  One multiply-add per clock
#                 per core.  Any op, any dtype.
#
#   Tensor Cores — hardwired matrix-multiply units.  One 16×16 block matmul
#                 per clock.  Only accept reduced-precision inputs (fp16, bf16,
#                 int8, fp8).  Introduced in Volta; improved each generation.
#
# Three precision modes expose three points on the speed/accuracy trade-off:
#
#   Variant A  float32  → CUDA cores.  Full 23-bit mantissa.
#               Slowest; RTX 4090 peak ~82 TFLOPS.
#
#   Variant B  tf32  → Tensor Cores with float32 I/O.
#               NVIDIA introduced tf32 on Ampere (RTX 30xx) as a transparent
#               speed-up: inputs are float32, but XLA silently truncates each
#               mantissa from 23 bits to 10 before feeding the Tensor Core.
#               No dtype change required — one JAX config flag enables it.
#               RTX 4090 peak ~165 TFLOPS (≈2× float32 CUDA cores).
#               Precision loss is usually < 0.1% for well-conditioned matmuls.
#
#   Variant C  bfloat16  → Tensor Cores with bf16 I/O.
#               Inputs and outputs are bf16 (8-bit exponent, 7-bit mantissa).
#               XLA calls a different cuBLAS kernel (cublasGemmEx with
#               CUBLAS_COMPUTE_16F).  Fastest path; RTX 4090 peak ~330 TFLOPS.
#               Precision loss ~1–2% vs float32.
#
# The ONLY changes between variants:
#   A → B : one jax.config flag (jax_default_matmul_precision)
#   B → C : the dtype of the input arrays
# The matmul function body is byte-for-byte identical across all three.
# ════════════════════════════════════════════════════════════════════════════

M = 8192

A_f32  = jax.random.normal(key, (M, M), dtype=jnp.float32)
B_f32  = jax.random.normal(key, (M, M), dtype=jnp.float32)
A_bf16 = A_f32.astype(jnp.bfloat16)
B_bf16 = B_f32.astype(jnp.bfloat16)

# Variant A: float32 → CUDA cores
@jax.jit
def matmul_f32(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    return A @ B

# Variant B: tf32 → Tensor Cores, float32 I/O
# jax_default_matmul_precision = "tensorfloat32" tells XLA to truncate fp32
# mantissas to 10 bits before passing to Tensor Cores.  The arrays stay fp32;
# only the internal GEMM accumulation changes.  Ampere+ only (RTX 30xx/40xx).
@jax.jit
def matmul_tf32(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    return A @ B    # identical body — the config flag is the only difference

# Variant C: bfloat16 → Tensor Cores, bf16 I/O
# dtype drives XLA to call cublasGemmEx with CUBLAS_COMPUTE_16F.
@jax.jit
def matmul_bf16(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    return A @ B    # identical body — dtype drives which HW path XLA picks

flops_p2 = 2 * M**3

print()
print(f"PART 2 — Tensor core utilization  ({M}×{M} matmul)")
print()
print(f"  {'label':<50s}  {'time':>7s}   {'throughput':>10s}")
print(f"  {'-'*50}  {'-'*7}   {'-'*10}")

t_f32 = benchmark(matmul_f32, A_f32, B_f32)

# Enable tf32: set precision flag, benchmark, then restore default so Part 1
# results above are unaffected if this script is re-ordered.
jax.config.update("jax_default_matmul_precision", "tensorfloat32")
t_tf32 = benchmark(matmul_tf32, A_f32, B_f32)
jax.config.update("jax_default_matmul_precision", "default")

t_bf16 = benchmark(matmul_bf16, A_bf16, B_bf16)

row("A: float32  (CUDA cores,          23-bit mantissa)", t_f32,  flops_p2)
row("B: tf32     (Tensor Cores, fp32 I/O, 10-bit mantissa)", t_tf32, flops_p2)
row("C: bfloat16 (Tensor Cores, bf16 I/O,  7-bit mantissa)", t_bf16, flops_p2)
print()
print(f"  → Speedup  B / A : {t_f32  / t_tf32:.1f}×  (tf32 vs float32)")
print(f"  → Speedup  C / A : {t_f32  / t_bf16:.1f}×  (bfloat16 vs float32)")
print(f"  → Speedup  C / B : {t_tf32 / t_bf16:.1f}×  (bfloat16 vs tf32)")

# Numerical sanity: tf32 loses ~1 decimal digit; bf16 loses ~2
out_f32  = matmul_f32(A_f32, B_f32)
jax.config.update("jax_default_matmul_precision", "tensorfloat32")
out_tf32 = matmul_tf32(A_f32, B_f32)
jax.config.update("jax_default_matmul_precision", "default")
out_bf16 = matmul_bf16(A_bf16, B_bf16).astype(jnp.float32)
ref      = jnp.max(jnp.abs(out_f32))
err_tf32 = float(jnp.max(jnp.abs(out_f32 - out_tf32)) / ref)
err_bf16 = float(jnp.max(jnp.abs(out_f32 - out_bf16)) / ref)
print(f"  → Max relative error  tf32 : {err_tf32:.2e}  (expected ~1e-3)")
print(f"  → Max relative error  bf16 : {err_bf16:.2e}  (expected ~1e-2)")

# Profile one steady-state execution of each variant
print()
print("  Collecting traces (Part 2) ...")
profile_run("p2_A_float32",  matmul_f32,  A_f32,  B_f32)
jax.config.update("jax_default_matmul_precision", "tensorfloat32")
profile_run("p2_B_tf32",     matmul_tf32, A_f32,  B_f32)
jax.config.update("jax_default_matmul_precision", "default")
profile_run("p2_C_bfloat16", matmul_bf16, A_bf16, B_bf16)
profile_run_xprof("p2_A_float32",  matmul_f32,  A_f32,  B_f32)
jax.config.update("jax_default_matmul_precision", "tensorfloat32")
profile_run_xprof("p2_B_tf32",     matmul_tf32, A_f32,  B_f32)
jax.config.update("jax_default_matmul_precision", "default")
profile_run_xprof("p2_C_bfloat16", matmul_bf16, A_bf16, B_bf16)

print()
print("=" * 70)
print()
print("Traces saved under: D:\\Projects\\AI-Perf-Sim\\workloads\\traces\\")
print()
print("  Perfetto  (timeline, dispatch gaps):")
print("    drag any .json.gz from traces/perfetto/ into https://ui.perfetto.dev")
print()
print("  xprof  (op breakdown, memory BW, roofline, kernel occupancy):")
print("    pip install xprof")
print("    xprof --logdir=/mnt/d/Projects/AI-Perf-Sim/workloads/traces/xprof")
