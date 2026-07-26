---
type: Index
title: Training Workloads
description: The training stage of the LLM lifecycle — the compute-heavy forward+backward regime and the distributed communication it depends on.
tags: [training, distributed-training, collectives, parallelism]
timestamp: 2026-07-26T00:00:00-07:00
---

# Training Workloads

← [Workloads Index](../index.md)

The **training** stage of the workload lifecycle: building the model by repeated
forward + backward passes over large datasets. Compared with
[inference](../inference/index.md), training is batch-based, compute-heavy
(backward pass ≈ 2× the forward), and dominated by weight/activation memory and
inter-accelerator communication.

This subsection is an entry point; the deep material currently lives in shared
and modeling pages linked below.

---

## Key topics

- **Compute accounting** — the `6ND` pre-training FLOP rule and full-lifecycle
  cost model are in [LLM inference model → total compute cost](../../modeling/llm_inference.md#total-compute-cost-accounting).
- **Parallelism** — data / tensor / pipeline / expert strategies and their
  memory/communication tradeoffs: [parallelism](../../modeling/parallelism.md).
- **Utilization** — [MFU / achieved vs. peak FLOPS](../../modeling/mfu.md).

Training relies on **collectives** for inter-accelerator communication, but those
are documented separately as [shared operators & kernels](../index.md#shared-operators--kernels)
([all-reduce / all-gather](../collective_ops.md),
[collective algorithms on TPU & GPU clusters](../collective_algorithms.md)) since
they are used across the whole lifecycle, not just training.

---

## See Also

- [Post-training workloads](../post-training/index.md) — fine-tuning, distillation, quantization
- [Inference workloads](../inference/index.md) — serving the trained model
- [Fault tolerance](../../simulation/fault_tolerance.md) — resilience for long training runs
- [References — Large Scale](../../references/index.md) — Megatron-LM, Ultra Scale Playbook, scaling books
