---
type: Index
title: Post-Training Workloads
description: What happens after pre-training — behavioral post-training (SFT, preference optimization, RL/RLVR) and model preparation (quantization, fine-tuning, compression distillation).
tags: [post-training, alignment, sft, dpo, kto, rlhf, rlvr, grpo, quantization, fine-tuning, distillation, lora]
timestamp: 2026-08-22T00:00:00-07:00
---

# Post-Training Workloads

← [Workloads Index](../index.md)

Everything that happens to a model *after* pre-training and *before* (or between)
serving. This sits between the [training](../training/index.md) and
[inference](../inference/index.md) stages of the lifecycle, and splits into two
distinct concerns that share the name:

- **Behavioral** — shaping *what the model does*: SFT, preference optimization,
  RL, verifiable rewards, environments.
- **Efficiency** — adapting and shrinking the model for cheap serving:
  quantization, LoRA, compression distillation.

---

## Pages

| Page | Covers |
|------|--------|
| [Behavioral post-training](behavioral-post-training.md) | The stack that turns a base model into a deployable policy: **SFT**, **offline preference optimization** (DPO/IPO/SimPO/ORPO/KTO), **online RL** (REINFORCE/PPO/GRPO), **RLVR and environments**, **on-policy distillation**, and **world adaptation** / mecha-nudges |
| [Model preparation](model-preparation.md) | **Quantization** (formats, AWQ/SmoothQuant/GPTQ, KV-cache quant), **fine-tuning** (LoRA/QLoRA and frameworks), and **distillation** (teacher/student, response/feature/CoT) |

Rule of thumb from the handbook:

> Distillation makes the model smaller · Quantization makes it lighter ·
> [Inference optimizations](../inference/optimization.md) make serving more efficient.

---

## See Also

- [Training workloads](../training/index.md) — the fwd/bwd compute profile SFT and RL updates inherit
- [Inference workloads](../inference/index.md) — serving the prepared model; also the rollout half of online RL
- [Model interaction](../inference/model-interaction.md#prompt-engineering) — prompt engineering / RAG as alternatives to fine-tuning
- [Memory capacity model](../../modeling/memory_capacity.md) — how precision changes the footprint
- [MoE efficiency](../moe.md) — sparse models: active vs. total parameters

---

## Attribution

[Behavioral post-training](behavioral-post-training.md) digests **"Post-Training
LLMs"** by **Kawin Ethayarajh** (University of Chicago, Booth), presented at the
*AI and Economics Summer Institute 2026*
([slides](https://kawine.github.io/assets/aiesi_post-training_public.pdf)).
Summarized and restructured for this knowledge base, with systems commentary added.

[Model preparation](model-preparation.md) is **adapted and summarized** from the
**LLM Inference Handbook** ([handbook.modular.com](https://handbook.modular.com/))
by **Modular**, used under the
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
Source: [`modular/llm-inference-handbook`](https://github.com/modular/llm-inference-handbook)
(`docs/`). **Changes were made** — the original chapter was condensed into a summary
and cross-linked into this knowledge base. This attribution does not imply endorsement
by Modular.
