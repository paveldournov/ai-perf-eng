---
type: Concept
title: Speculative Decoding
description: Draft-and-verify inference acceleration — why it works, the lossless acceptance rule, draft-strategy taxonomy, tree attention, and a speedup model.
tags: [inference, speculative-decoding, draft-model, tree-attention, decode, latency]
timestamp: 2026-06-30T00:00:00-07:00
---

# Speculative Decoding

← [Modeling Index](index.md)

Speculative decoding accelerates autoregressive LLM **decode** without changing the output
distribution. A cheap *drafter* proposes several future tokens; the expensive *target* model
**verifies them all in a single forward pass** and accepts the longest prefix consistent with
its own distribution. Because verifying `k` tokens costs almost the same as generating one,
each target pass can emit multiple tokens — turning a sequential bottleneck into a batched one.

---

## Why It Works: Decode Is Memory-Bound

A normal decode step is [memory-bandwidth-bound](llm_inference.md#decode-latency-model-per-token):
the GPU loads the entire weight set from HBM to produce **one** token, leaving the tensor cores
nearly idle. Scoring `k` candidate tokens in one pass loads those same weights **once** and uses
the spare compute to check all `k` positions. So the marginal cost of verifying extra tokens is
small until you re-enter the compute-bound regime.

> Speculative decoding trades **idle FLOPs** (plentiful at low batch) for **fewer HBM weight
> loads** (the actual bottleneck). It helps most exactly where decode is most wasteful:
> low batch size, latency-sensitive serving.

---

## The Draft → Verify → Accept Loop

1. **Draft.** A cheap mechanism proposes a candidate continuation of `γ` tokens (or a *tree* of
   candidates — see below).
2. **Verify.** The target runs **one** forward pass over prompt + all draft positions, producing
   its true next-token distribution at each position in parallel (using a causal/tree mask so
   position `i` only sees positions before it).
3. **Accept.** Walk the draft left-to-right; accept each token under the acceptance rule. On the
   first rejection, discard the rest, **correct** that position from the target's distribution,
   and loop. The target's free "bonus" token after the last accepted position means a step
   accepting `n` drafts emits `n+1` tokens.

---

## Losslessness: the Acceptance Rule

Speculative decoding is **exact** — the generated sequence is distributed identically to plain
target sampling (Leviathan et al., 2023; Chen et al., 2023). For draft prob `q(x)` and target
prob `p(x)` at a position:

```
accept x with probability  min(1, p(x) / q(x))
on reject, resample from    norm(max(0, p(x) − q(x)))
```

This **modified rejection sampling** provably yields samples from `p`. For **greedy** decoding it
reduces to a simple rule: accept the draft token iff it equals the target's argmax. Either way,
the target — not the drafter — decides the output; the drafter only affects *speed*, never
*quality*. (Lossy variants exist, e.g. relaxed acceptance thresholds, but trade exactness for
higher acceptance.)

---

## Speedup Model

Two quantities govern the win:

- **Acceptance rate `α`** — probability a drafted token is accepted (how well the drafter mimics
  the target on this workload).
- **Cost ratio `c`** — drafter cost per token relative to a target step (`c ≪ 1` for a good drafter).

For a linear chain of `γ` draft tokens, the expected number of tokens produced per target pass is

```
E[tokens] = (1 − α^(γ+1)) / (1 − α)
```

and the wall-clock speedup over plain decoding is roughly

```
speedup ≈ E[tokens] / (1 + c·γ)
```

**Reading it:** speedup rises with `α` and saturates in `γ` — past some draft length the
`α^(γ+1)` term vanishes and extra drafts just add cost. High-`α` workloads (code, math, templated
text) speculate far better than open-ended chat. This is why reported speedups span ~2× (chat) to
~10× (math/code).

---

## Draft-Strategy Taxonomy

The drafter is the design space. All share the same lossless verifier.

| Strategy | Drafter | Pros | Cons |
|---|---|---|---|
| **Independent draft model** | A small separate LLM (e.g. 1B drafting for 70B) | Simple; strong drafts | Must train/host a second model; vocab must match |
| **Self-speculative / early-exit** | The target itself, exiting at an early layer | No extra params | Modest acceptance; intrusive to the forward pass |
| **Medusa** | Extra **parallel heads** on the target predicting tokens `t+1, t+2, …` | One pass, no draft model | Heads are independent → weaker joint consistency |
| **EAGLE** | Lightweight head that drafts **autoregressively in feature space** on frozen target features | High acceptance; cheap | Sequential draft passes (latency per draft token) |
| **Lookahead / n-gram / prompt-lookup** | Retrieval of n-grams from context or a Jacobi fixed-point | Training-free | Acceptance is workload-dependent |
| **Block-diffusion heads** | Generate a whole block/tree in one pass | Parallel, fast | Branches scored independently → tree drifts from target |
| **Tree drafting** | Any of the above, but proposing a **tree** of candidates | Higher accepted length per pass | Needs tree attention + metadata management |

The frontier tension: **autoregressive** drafters (EAGLE) preserve quality but draft sequentially;
**one-pass** drafters (block-diffusion/Medusa) are fast but lose branch consistency. JetSpec
(below) targets exactly this gap.

---

## Tree Drafting and Tree Attention

A single linear draft chain wastes the verifier: if token 3 is wrong, tokens 4…γ are thrown away.
**Tree drafting** instead proposes multiple candidate continuations branching from shared prefixes,
so the target verifies many alternative paths in one pass and accepts the best root-to-leaf path.

This requires a **tree-causal attention mask**: each tree node attends to the original prefix and
its **ancestors** in the tree, but **not** to sibling branches or descendants. That mask is what
lets the whole tree be packed into one sequence and scored in a single forward pass while each
branch keeps correct autoregressive dependencies. It is a structured generalization of the causal
mask — see [Attention → tree-causal masking](../workloads/attention.md#tree-causal-masking).
Production implementations pair it with [paged attention](../workloads/attention.md#flash-attention)
so tree nodes share KV-cache pages with their common prefix.

---

## Worked Example: JetSpec (Parallel Tree Drafting)

**Source:** Hao AI Lab (2026), "JetSpec: Parallel Tree Drafting for Speculative Decoding."

JetSpec introduces **causal parallel tree drafting** — generating an entire candidate tree in one
draft pass *while preserving autoregressive factorization across branches*, resolving the
quality-vs-parallelism dilemma above.

- **Drafter:** a lightweight head trained on **frozen** target features (target untouched →
  plug-and-play) via forward-KL distillation over 16-token draft blocks.
- **Mechanism:** a **tree-causal mask** (node attends to prefix + ancestors only) used in *both*
  the one-pass draft and the one-pass verification, so branches stay target-consistent instead of
  drifting like independent block-diffusion scoring.
- **Lossless:** standard speculative-decoding acceptance — output matches the target.
- **Serving:** vLLM fork with custom **Triton + NVIDIA CuTe paged-FlashAttention** kernels and tree
  metadata management.

**Results (Qwen3-8B, budget 256, greedy):**

| Benchmark | Speedup |
|---|---|
| MATH-500 | **9.64×** |
| AIME25 | **8.78×** |
| HumanEval | **7.12×** |
| MT-Bench (open chat) | **4.58×** |

Evaluated on H100 and B200; ~**1000 TPS** on a single B200 (Qwen3-8B, MATH-500). JetSpec sustains
gains at higher token budgets where DDTree/DFlash saturate — the acceptance-rate-vs-budget curve
stays favorable because branch consistency is preserved. The high-`α` reasoning/code workloads
(MATH, AIME, HumanEval) speculate best; open-ended chat least — exactly what the
[speedup model](#speedup-model) predicts.

---

## When It Helps — and When It Doesn't

- **Best:** low batch size, latency-bound single-stream serving, high-acceptance workloads
  (code, math, structured output), greedy or low-temperature sampling.
- **Marginal/negative:** large batches already saturating the tensor cores — the extra
  verification FLOPs now **compete** with real work instead of using idle compute, so the
  memory-bound advantage shrinks. High-temperature, open-ended generation lowers `α`.
- **Cost:** drafter training/hosting, KV-cache and mask bookkeeping for trees, and added
  scheduling complexity in a batched server (variable accepted length per request per step).

---

## See Also

- [LLM inference model](llm_inference.md) — why decode is memory-bound (the premise)
- [Attention — tree-causal masking](../workloads/attention.md#tree-causal-masking) — the mask that makes tree verification one pass
- [MoE efficiency](../workloads/moe.md) — DeepSeek-V3 Multi-Token Prediction heads double as speculative drafters
- [Roofline model](roofline.md) — the compute-vs-BW framing behind "verify many for the price of one"
- [References — LLM Inference Efficiency](../references/index.md#llm-inference-efficiency)
