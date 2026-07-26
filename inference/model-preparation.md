---
type: Method
title: Model Preparation
description: Preparing a model for efficient inference — quantization (formats and methods), fine-tuning (LoRA/QLoRA and frameworks), and distillation (teacher/student).
tags: [quantization, fine-tuning, lora, qlora, distillation, awq, gptq, smoothquant]
resource: https://handbook.modular.com/model-preparation/
timestamp: 2026-07-26T00:00:00-07:00
---

# Model Preparation

← [LLM Inference Serving Index](index.md)

Digest of the handbook's [Model preparation](https://handbook.modular.com/model-preparation/)
chapter — shrinking or adapting a model *before* serving so inference is cheaper
and better-fit. Complements the runtime [serving optimizations](optimization.md):

> Distillation makes the model smaller · Quantization makes it lighter ·
> [Inference optimizations](optimization.md) make serving more efficient.

---

## Quantization

Convert weights (and optionally activations / KV cache) from high precision
(FP32/FP16) to lower precision (FP8/INT8/INT4/…). Three benefits: smaller
footprint, less data movement (helps the memory-bound decode phase), and faster
compute where hardware supports the format (H100: BF16/FP16 ≈ 1,979 TFLOPS vs
FP8/INT8 ≈ 3,958 — a clean 2× from halving bit width).

| Format | Size vs FP32 | Accuracy drop | Typical use |
|--------|--------------|---------------|-------------|
| FP32 | 100% | none | training |
| FP16 | 50% | minimal | training & inference (standard) |
| FP8 | 25% | low | training & inference (emerging; needs Hopper+) |
| INT8 | 25% | low | inference (good all-round) |
| INT4 | 12.5% | moderate | inference (needs GPTQ/AWQ) |
| INT2 | 6.25% | high | rare/experimental |

- **What to quantize:** weights (most common, stable), activations (trickier,
  more accuracy loss), and the **KV cache** at runtime (separate from weight
  quantization; must preserve key/query similarity — quantizing weights alone
  does **not** shrink KV cache per token).
- **Methods:** **AWQ** (protect the ~1% salient weights via activation stats;
  great for edge/low-bit), **SmoothQuant** (training-free W8A8; shifts
  quantization difficulty from activations to weights), **GPTQ** (fast PTQ to
  3–4 bit at scale; e.g. quantize OPT-175B in ~4 GPU-hours, ~3.25× over FP16).
- **vs pruning:** quantization reduces bits per weight; pruning removes weights.
  Often combined (train → prune → fine-tune → quantize → deploy). Quantization is
  usually easier in production thanks to hardware low-precision support.
- Most teams don't implement quantization themselves — start from a pre-quantized
  model on [Hugging Face](getting-started.md#choosing-the-right-model). Use it
  when GPU memory is limited, latency/cost must drop, or higher concurrency is
  needed; avoid it when maximum accuracy is required or the model is already
  small. Sizing: see [GPU memory](getting-started.md#sizing-gpu-memory) and
  [memory capacity](../modeling/memory_capacity.md).

## Fine-tuning

Continue training a pre-trained model on task-specific data to improve domain
expertise, instruction-following, or safety/alignment. It changes weights — unlike
[prompt engineering](model-interaction.md#prompt-engineering),
[function calling](model-interaction.md#function-calling), or RAG — so use it when
the desired behavior is stable, recurring, and not cleanly handled by prompts or
retrieval.

- **Efficient methods:** **LoRA** / **QLoRA** update small adapters instead of all
  weights, enabling fine-tuning on a single consumer GPU (full fine-tuning needs
  stronger hardware).
- **Frameworks:** Axolotl (YAML-config, HF-based), Unsloth (Triton-kernel
  optimized — ~2× faster, up to 80% less memory), Torchtune (PyTorch-native),
  LLaMA Factory (100+ models, Web UI).
- **Practical loop:** build an eval set → improve prompts/retrieval/tools →
  measure → fine-tune only for systematic remaining failures. Avoid fine-tuning
  when info is missing/changing (use RAG), when you lack an eval set, or when the
  deployment path (base model/provider) is uncertain. Quality of data matters more
  than quantity (often a few thousand good examples).
- **Hosted fine-tuning** is convenient but ties you to a provider's base-model
  lifecycle — treat a fine-tuned hosted model as a maintained production artifact.

## Distillation

Train a smaller **student** model to replicate a larger **teacher**, producing an
entirely new, smaller/faster/cheaper model that keeps much of the teacher's
capability (unlike quantization, which reduces precision of the *same* model — the
two are complementary and can be combined).

- **How:** the student trains on the teacher's **soft labels** (full next-token
  distribution — richer than a single hard label), typically combining
  distillation loss with standard cross-entropy.
- **Types:** *response* (match output probabilities; most common, scalable),
  *feature* (mimic intermediate representations; needs teacher internals),
  *chain-of-thought* (train on teacher reasoning traces — how DeepSeek-R1's
  distilled 1.5B–70B variants gained reasoning).
- **Inference impact:** lower latency, less memory, higher throughput, lower cost
  — before any runtime optimization. Use when you need a smaller model, a quantized
  large model still doesn't fit, or you want to transfer a specific capability;
  note some licenses prohibit training on model outputs.

---

## See Also

- [Planning a deployment](getting-started.md) — model choice, weight formats, GPU sizing
- [Memory capacity model](../modeling/memory_capacity.md) — how precision changes the footprint
- [Kernel optimization](kernel-optimization.md) — low-precision tensor-core throughput
- [MoE workloads](../workloads/moe.md) — sparse models and active-vs-total parameters
