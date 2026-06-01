#!/usr/bin/env python3
"""
Pallas vector addition kernel — NVIDIA RTX GPU (Triton backend)
===============================================================
Demonstrates the minimal Pallas programming model:
  - kernel body (Refs, no return)
  - BlockSpec  (which tile each program instance owns)
  - grid       (how many instances)
  - interpret mode for CPU debugging before GPU compilation

Install: pip install -U "jax[cuda12]"
Run:     python3 pallas_add_gpu.py
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

print(f"Devices: {jax.devices()}")
print()

# ── Configuration ─────────────────────────────────────────────────────────────

N          = 1 << 16    # 65 536 elements — must be a multiple of BLOCK_SIZE
BLOCK_SIZE = 256        # elements each program instance handles
                        # power-of-2, ≥ 32 (warp size); 256 is a safe default

# ── Kernel ────────────────────────────────────────────────────────────────────
#
# One program instance runs this function for one BLOCK_SIZE slice.
# x_ref / y_ref / o_ref are *not* full arrays — each program sees only its
# tile, pre-selected by BlockSpec.  Reading with [...] loads all elements of
# that tile into registers; writing to o_ref[...] stores them back to HBM.
#
# No return statement — side-effects on o_ref are the output mechanism.

def add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]

# ── pallas_call wrapper ───────────────────────────────────────────────────────
#
# BlockSpec(block_shape, index_map):
#   block_shape = (BLOCK_SIZE,)         how large is each tile?
#   index_map   = lambda i: (i,)        program i → tile i
#                                       tile i covers elements [i*BLOCK, (i+1)*BLOCK)
#
# grid = (N // BLOCK_SIZE,)             total number of program instances

block_spec = pl.BlockSpec((BLOCK_SIZE,), lambda i: (i,))

def pallas_add(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[block_spec, block_spec],
        out_specs=block_spec,
        grid=(N // BLOCK_SIZE,),
    )(x, y)

# ── Data ──────────────────────────────────────────────────────────────────────

key = jax.random.PRNGKey(42)
x = jax.random.normal(key,               (N,), dtype=jnp.float32)
y = jax.random.normal(jax.random.fold_in(key, 1), (N,), dtype=jnp.float32)
reference = x + y    # jnp ground truth

# ── Step 1: verify logic with interpret=True (runs on CPU, no compilation) ───
#
# interpret=True replaces the Triton/Mosaic compiler with a JAX interpreter.
# Use this whenever you're debugging a kernel — same Python code, no GPU needed.
# If interpret=True and GPU give different answers, the compiled path has a bug.

out_interpret = pl.pallas_call(
    add_kernel,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    in_specs=[block_spec, block_spec],
    out_specs=block_spec,
    grid=(N // BLOCK_SIZE,),
    interpret=True, debug=True
)(x, y)

err_interpret = float(jnp.max(jnp.abs(out_interpret - reference)))
print(f"interpret mode  max |error| vs jnp.add : {err_interpret:.2e}  {'✓' if err_interpret == 0.0 else '✗'}")

# ── Step 2: compile and run on GPU ───────────────────────────────────────────
#
# First call triggers Triton compilation (slow — a few seconds).
# Subsequent calls hit the compiled kernel cache and are fast.

out_gpu = pallas_add(x, y)
jax.block_until_ready(out_gpu)

err_gpu = float(jnp.max(jnp.abs(out_gpu - reference)))
print(f"GPU mode        max |error| vs jnp.add : {err_gpu:.2e}  {'✓' if err_gpu == 0.0 else '✗'}")
print()

# ── Step 3: annotate what happened ───────────────────────────────────────────
#
# grid = (N // BLOCK_SIZE,) = 256 program instances
# each instance:
#   reads  BLOCK_SIZE float32s from x  →  256 × 4 = 1 024 bytes
#   reads  BLOCK_SIZE float32s from y  →  1 024 bytes
#   writes BLOCK_SIZE float32s to  o   →  1 024 bytes
#   total HBM traffic per instance: 3 072 bytes
#   total HBM traffic: 256 × 3 072 = 786 432 bytes = 0.75 MB
#   (for N=65536; scales linearly with N)

total_bytes = N * 3 * x.itemsize
print(f"Problem size    : N = {N:,} float32 elements")
print(f"Grid            : {N // BLOCK_SIZE} program instances × {BLOCK_SIZE} elements each")
print(f"HBM traffic     : {total_bytes / 1e6:.2f} MB  (read x, read y, write o)")
print()
print("Kernel is memory-bandwidth bound — compute cost is negligible.")
print("Pallas and jnp.add will be identical in throughput for this op.")
