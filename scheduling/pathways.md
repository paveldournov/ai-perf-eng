---
type: Tool
title: Pathways on Cloud
description: Google's orchestration runtime for large-scale multi-host JAX training and inference.
tags: [scheduling, google, jax, orchestration, tpu]
resource: https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro
timestamp: 2026-05-30T10:50:39-07:00
---

# Pathways on Cloud

← [Back to Scheduling](index.md)

---

**Source:** Google Cloud AI Hypercomputer docs — [Pathways intro](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro)

Pathways is Google's ML runtime for orchestrating large-scale JAX workloads across multiple TPU slices. It enables a single JAX client to treat thousands of accelerators as one unified device, supporting training, inference, and interactive workloads with automatic recovery.

---

## The Core Problem: Multi-Controller vs. Single-Controller

Standard JAX uses a **multi-controller** model: each process sees only the devices local to it, and SPMD collective ops coordinate across processes. This works well for symmetric all-dense training but breaks down for:

- Workloads spanning multiple TPU slices (cross-slice collectives require explicit coordination)
- Sparse/MoE models with asymmetric compute paths
- Interactive or RPC-style inference where a CPU client drives incremental TPU execution

Pathways replaces this with a **single-controller** model: one JAX client holds a unified view of all accelerators across all slices, and dispatches computation centrally.

| | Pathways (single-controller) | JAX default (multi-controller) |
|---|---|---|
| Control plane | Centralized client process | One process per host |
| Device view | Unified across all slices | Local to each process |
| Cross-slice ops | Transparent | Requires explicit SPMD coordination |
| Sparse/asymmetric | Supported natively | Difficult |
| Programming model | Standard JAX | SPMD-focused |

---

## Component Architecture

```
                        ┌─────────────────────────────┐
                        │     User Python Process      │
                        │  JAX client + IFRT Proxy     │
                        │       Client (CPU)           │
                        └────────────┬────────────────┘
                                     │ gRPC
                        ┌────────────▼────────────────┐
                        │     IFRT Proxy Server        │
                        │         (CPU)                │
                        └────────────┬────────────────┘
                                     │
                        ┌────────────▼────────────────┐
                        │    Pathways Client (IFRT)   │
                        │   Unified device view (CPU) │
                        └──────┬─────────────┬────────┘
                               │             │
               ┌───────────────▼─┐     ┌─────▼──────────────┐
               │ Resource Manager│     │  Pathways Workers   │
               │  (CPU-only)     │     │  (on TPU VMs)       │
               │  - allocation   │     │  - execute compiled │
               │  - health mon.  │     │    XLA programs     │
               │  - scheduling   │     │  - return results   │
               └─────────────────┘     └─────────────────────┘
```

### Components

**Pathways Resource Manager** — Central control plane. Manages accelerator allocation, health monitoring, job scheduling, and error coordination. CPU-only; runs as a dedicated pod.

**Pathways Client** — IFRT implementation that presents a unified device view to the JAX application. Coordinates with the Resource Manager for program placement. CPU-only.

**IFRT Proxy Client / Server** — Open-source runtime layer that decouples user code from the Pathways infrastructure. The proxy server (gRPC, port 29000) forwards requests from the user process to the Pathways client, enabling portability and client/server separation.

**Pathways Worker** — Processes running on TPU VMs. Receive compiled XLA executables from the client, execute them on the TPU hardware, and return results via the IFRT proxy layer. These are the only components that consume accelerator resources.

**Sidecar Server** — Optional gRPC server co-located with the worker pods. Executes user Python code directly on accelerator VMs, reducing data-transfer latency for latency-sensitive inference workloads.

---

## Kubernetes Integration: PathwaysJob CRD

Pathways workloads on GKE are defined via the **PathwaysJob** Custom Resource Definition, which wraps the JobSet API. It abstracts individual pod specs and manages the full lifecycle.

Key fields:

| Field | Purpose |
|-------|---------|
| `pathwaysVersion` | JAX/Pathways version (e.g. `jax-0.5.3`) |
| `workers` | TPU machine type, topology, slice count |
| `maxRestarts` | Job-level restart threshold on failure |
| `maxSliceRestarts` | Per-slice restart limit (resilient training) |
| `elasticSlices` | Max unavailable slices before job is unhealthy |
| `pathwaysDir` | GCS bucket for XLA compilation artifacts |
| `deploymentMode` | CPU pods on dedicated node vs. co-located with workers |
| `customComponents` | Override proxy server, resource manager, sidecar images |

### Container Images (per version tag)

| Component | Image |
|-----------|-------|
| Resource Manager / Worker | `us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:jax-<ver>` |
| IFRT Proxy Server | `us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:jax-<ver>` |

Worker pods require the `PATHWAYS_HEAD`, `MEGASCALE_NUM_SLICES`, `MEGASCALE_SLICE_ID`, and `MEGASCALE_COORDINATOR_ADDRESS` environment variables to locate the control plane and identify their slice.

---

## Supported Workload Patterns

| Pattern | Pathways mechanism |
|---------|-------------------|
| Large-scale dense training | Single client drives all-slice SPMD |
| MoE / sparse training | Asymmetric dispatch across slices |
| Multihost inference | Unified KV cache and weight sharding across slices |
| Resilient training | Automatic slice restart via `maxSliceRestarts`; `elasticSlices` for degraded-mode operation |
| Interactive / RPC inference | Sidecar server for low-latency client ↔ TPU round-trips |

---

## Performance Considerations

- **Compilation caching:** XLA programs are compiled once and cached to the `pathwaysDir` GCS bucket; subsequent restarts skip recompilation.
- **Cross-slice latency:** ICI fabric (see [Boardfly](../hardware/tpu/boardfly.md)) handles intra-pod all-reduce; cross-pod traffic routes over the data-center network via DCN — the primary latency boundary.
- **CPU bottleneck:** The Resource Manager and Pathways Client are CPU-only; for large clusters, sizing the CPU node correctly is important to avoid dispatch latency.
- **Sidecar vs. proxy:** For inference, co-locating user Python on the accelerator VM via the sidecar avoids an extra gRPC hop and reduces TTFT.

---

## See Also

- [TPU v6e (Trillium)](../hardware/tpu/tpu_v6e.md) — primary Pathways target for inference
- [TPU v8t / v8i](../hardware/tpu/tpu_v8.md) — next-gen training/inference targets
- [Boardfly interconnect](../hardware/tpu/boardfly.md) — ICI topology that Pathways dispatches across
- [Kueue](kueue.md) — Kubernetes job admission that sits above Pathways for cluster-level quota
- [Pallas kernels](../workloads/pallas_kernels.md) — writing custom kernels for the workers Pathways dispatches
