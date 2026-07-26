---
type: Concept
title: Model Interaction
description: How applications drive an LLM — inference parameters (sampling, length, penalties), structured outputs and constrained decoding, function calling, MCP, API compatibility, and prompt engineering.
tags: [inference-parameters, sampling, structured-outputs, constrained-decoding, function-calling, mcp, prompt-engineering]
resource: https://handbook.modular.com/model-interaction/
timestamp: 2026-07-26T00:00:00-07:00
---

# Model Interaction

← [LLM Inference Serving Index](index.md)

Digest of the handbook's [Model interaction](https://handbook.modular.com/model-interaction/)
chapter — sending a request and shaping the response. These are request-time
settings and protocols; they don't change model weights, but several have direct
serving implications (scheduling, KV-cache pressure, cost).

---

## Inference parameters

Request-time controls over decoding. Common ones:

| Parameter | Controls |
|-----------|----------|
| `temperature` | Randomness — lower = sharper/stable, higher = flatter/creative; `0` ≈ greedy |
| `top_p` (nucleus) | Smallest token set covering cumulative prob mass; adapts to model uncertainty |
| `top_k` | Keep only the k most likely tokens; simple but doesn't adapt to distribution shape |
| `max_tokens` / `min_tokens` | Bound (or floor) output length — a key latency & cost control |
| `stop` / `stop_token_ids` | End generation at a delimiter (not a substitute for schema enforcement) |
| `presence_penalty` / `frequency_penalty` / `repetition_penalty` | Discourage repetition (blunt instruments) |
| `seed` | Improve reproducibility (never a hard guarantee across version/kernel/batch changes) |
| `logprobs` | Token probabilities for debugging/scoring (larger responses) |
| `n` / `best_of` | Multiple candidates (multiplies cost) |
| `logit_bias`, `bad_words`, `allowed_token_ids` | Advanced token-level control for software-consumed output |

<a id="sampling-parameters"></a>
**Sampling** (see [basics](basics.md#context-window--sampling)): tune
`temperature` first; use `top_p` to bound the long tail adaptively; use `top_k`
when a framework recommends a hard cap. Don't change all at once during eval.
Field names/behavior vary across providers — treat them as part of your
evaluation surface, not portable guarantees.

**Serving impact:** larger `max_tokens` and exploratory sampling raise latency,
GPU utilization, KV-cache pressure, and tail latency; long generations hold
request state and KV cache longer. Handbook starting points (evaluate on your
own workload):

| Use case | Temperature | Top-p |
|----------|-------------|-------|
| Classification / extraction | 0.0–0.2 | 1.0 |
| RAG / factual QA | 0.1–0.5 | 0.9–1.0 |
| General chat | 0.5–0.8 | 0.9–1.0 |
| Code generation | 0.0–0.3 | 1.0 |
| Brainstorming / creative | 0.7–1.3 | 0.9–0.95 |

## Structured outputs

Responses in a machine-readable format (JSON, XML, regex-defined) so downstream
systems can parse them directly. Three ways to obtain them:

1. **Provider-native** (OpenAI/Anthropic/Google schema or JSON mode) — easiest;
   trades vendor lock-in, possible truncation, uneven enforcement.
2. **Re-prompting** (e.g. Instructor) — validate, and re-prompt on failure until
   valid; flexible, works with any model, but costs extra calls/latency.
3. **Constrained decoding** (structured generation) — modify logits in real time
   so only schema-valid tokens can be sampled. Fast, reliable, no retries; used
   by Outlines, Guidance, XGrammar, and integrated in vLLM and SGLang. Best for
   self-hosted open models; can even *improve* task accuracy.

## Function calling

Lets the model request external **tools** when a task needs outside data or
actions. Mechanically it's still next-token prediction guided by function
signatures in the prompt; the model emits a structured call, your code runs it,
and the result is fed back. Distinctions: *structured outputs* control response
**format**; *function calling* controls **when to act**; **tools** are the
actions; **agents** are systems that use function calling in a reasoning loop.

## Model Context Protocol (MCP)

An open standard for connecting AI assistants to external services/data via a
client-server architecture: **host** (the AI app, e.g. an IDE or chat client) →
**clients** (1:1 links) → **servers** (connectors exposing databases, APIs,
files, repos) over the **MCP protocol** transport.

## API compatibility

Most self-hosted engines (vLLM, SGLang, MAX) expose **OpenAI-compatible** (and
increasingly **Anthropic-compatible**) HTTP endpoints, so application code uses
the same client interface regardless of the backend — the key to keeping a
[serving stack runtime-agnostic](getting-started.md#choosing-the-right-inference-framework)
and portable across serverless and self-hosted deployments.

## Prompt engineering

Adjusting *how* you ask to get better answers — fast, cheap, no training, and the
first lever to pull before [fine-tuning](model-preparation.md#fine-tuning). It
has limits (long prompts get messy; behavior can drift), so the practical loop is
prompt/retrieval/tooling first, and fine-tune only for systematic failures worth
baking into the model.

---

## See Also

- [Inference basics](basics.md) — sampling and the decode loop these settings steer
- [Model preparation](model-preparation.md) — when to fine-tune instead of prompt
- [Infrastructure & operations](infrastructure-ops.md) — RAG / multi-model pipelines
- [Inference optimization](optimization.md) — how length & sampling affect scheduling
