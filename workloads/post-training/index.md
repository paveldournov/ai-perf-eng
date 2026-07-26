---
type: Index
title: Post-Training Workloads
description: Adapting and compressing a trained model before serving — quantization, fine-tuning, and distillation.
tags: [post-training, quantization, fine-tuning, distillation, lora]
timestamp: 2026-07-26T00:00:00-07:00
---

# Post-Training Workloads

← [Workloads Index](../index.md)

Everything that happens to a model *after* pre-training and *before* (or between)
serving: adapting it to a task and shrinking it for efficient
[inference](../inference/index.md). This sits between the
[training](../training/index.md) and [inference](../inference/index.md) stages of
the lifecycle.

---

## Pages

| Page | Covers |
|------|--------|
| [Model preparation](model-preparation.md) | **Quantization** (formats, AWQ/SmoothQuant/GPTQ, KV-cache quant), **fine-tuning** (LoRA/QLoRA and frameworks), and **distillation** (teacher/student, response/feature/CoT) |

Rule of thumb from the handbook:

> Distillation makes the model smaller · Quantization makes it lighter ·
> [Inference optimizations](../inference/optimization.md) make serving more efficient.

---

## See Also

- [Inference workloads](../inference/index.md) — serving the prepared model
- [Model interaction](../inference/model-interaction.md#prompt-engineering) — prompt engineering / RAG as alternatives to fine-tuning
- [Memory capacity model](../../modeling/memory_capacity.md) — how precision changes the footprint
- [MoE efficiency](../moe.md) — sparse models: active vs. total parameters
