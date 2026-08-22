---
type: Concept
title: Behavioral Post-Training — SFT, Preference Optimization, RL, RLVR
description: The post-training stack that turns a base model into a deployable policy — SFT, offline preference optimization (DPO/KTO), online RL (REINFORCE/PPO/GRPO), verifiable rewards and environments, on-policy distillation, and world adaptation.
tags: [post-training, alignment, sft, dpo, kto, rlhf, rlvr, ppo, grpo, distillation, rl-environments]
resource: https://kawine.github.io/assets/aiesi_post-training_public.pdf
timestamp: 2026-08-22T00:00:00-07:00
---

# Behavioral Post-Training — SFT, Preference Optimization, RL, RLVR

← [Post-Training Workloads Index](index.md)

Digest of **"Post-Training LLMs"** — Kawin Ethayarajh (University of Chicago,
Booth), *AI and Economics Summer Institute 2026*, Chicago, Aug 6–11 2026
([slides](https://kawine.github.io/assets/aiesi_post-training_public.pdf)).

This page covers post-training in the **behavioral** sense — the training stages
that shape *what the model does*. It is the sibling of
[Model preparation](model-preparation.md), which covers post-training in the
**efficiency** sense (quantization, LoRA, compression distillation). Both happen
after pre-training; they answer different questions.

> *The model you use is not the model that was pretrained.*

---

## Why post-training exists

Pre-training learns a distribution over text: one generic objective over trillions
of tokens. The result is broad knowledge and capability **without a reliable user
interface** — a base model *continues* text, and a plausible continuation is not
necessarily a useful response. Post-training turns broad capability into a
**behavioral policy**: given context *x*, which responses *y* are probable.

Its goals, in the lecture's framing:

| Goal | What it buys |
|------|--------------|
| Interface | Follow instructions, use chat formats, call tools |
| Capability | Reason, code, browse, execute long-horizon tasks |
| Preference | Be helpful, concise, calibrated, stylistically appropriate |
| Safety | Refuse harmful requests without refusing harmless ones |
| Product | Obey latency, cost, reliability, and domain constraints |

Checkpoints along the way: **base** ("write a plausible continuation") →
**instruct** ("answer the user helpfully") → **reasoning** ("think deeply, try
multiple paths, call tools"). Post-training mostly changes the probability of
*already latent* behavior, and can build new behavioral routines — how much is
genuinely new remains debated.

### Lifecycle stages

| | Pre-training | Mid-training | Post-training |
|---|---|---|---|
| Data | Web-scale text, code, images | Targeted domain / capability corpora | Demos, preferences, rewards, environments |
| Typical scale | Trillions of tokens | Billions–trillions | Millions–billions |
| Signal | What text occurs? | How well-primed is the model for post-training? | What behavior is desired? |
| Output | Base model | Stronger base | Deployable model |

Boundaries are conventions, not hard lines — objectives and datasets blur across
stages. **Post-training ≠ alignment**: post-training is a question of *when in the
lifecycle*; alignment is a question of *aligned to what objective*. Most alignment
happens in post-training; not all post-training is alignment.

### The stack

1. **[SFT](#1-supervised-fine-tuning-sft--imitate)** — imitate
2. **[Offline preference optimization](#2-offline-preference-optimization--compare)** — compare
3. **[Online post-training](#3-online-post-training--explore)** — explore
4. **[RLVR and environments](#4-rewards--environments--verify)** — verify
5. **[On-policy distillation](#5-on-policy-distillation--transfer)** — transfer
6. **[World adaptation](#6-world-adaptation--anticipate)** — anticipate

---

## 1. Supervised fine-tuning (SFT) — imitate

Teach by showing. Each record is a (prompt *x*, desired response *y\**) pair, and
a large part of SFT is **instruction-following**: SFT on many tasks each phrased
as an (instruction, follow-through) pair generalizes to *unseen* task types
(Wei et al. 2022, FLAN). Demonstrations are not limited to prose answers — code,
tool calls (`search({"query": "BLS CPI latest"})`), and reasoning all appear as
targets.

**Mechanically identical to pre-training**: next-token prediction, just on
demonstrations instead of broad corpora. The one systems-relevant difference is
the **loss mask** — loss is computed only on the assistant response, with context
tokens (system, user) masked out:

```
SYSTEM You are helpful. | USER Estimate the effect. | ASSISTANT The estimate is...
   0   0   0   0    0   |  0    0     0      0      |     1     1    1    1
```

For the compute profile of that forward/backward pass, see
[Training workloads](../training/index.md) — SFT is a training-shaped workload
at a much smaller token count.

### Where SFT data comes from

Four sources feed one curated mixture: **human experts** (write/edit
demonstrations), **existing data** (reformat tasks as instructions), **teacher
models** (synthetic responses), and the **current policy** (sample, score,
filter). Synthetic data can bootstrap instruction-following outright
(Wang et al. 2023, *Self-Instruct*).

Compared to pre-training, **data quality matters far more than quantity**: LIMA
(Zhou et al. 2023) found quality + diversity beat 16× more examples with no
measured gain from the extra data. This is not a universal scaling law, but it
inverts the pre-training instinct.

Modern demonstrations are **trajectories, not answers** — user request → plan →
tool call → observation → check → final answer with caveats. SFT can imitate an
entire workflow.

### What SFT cannot do

| Does | Does not |
|---|---|
| Raises probability of desired responses | Say which of *n* valid options is better |
| Teaches formats and behavioral routines | Say how costly a mistake is |
| Creates a stable user interface | Say what happens under its *own* mistakes |

It is fundamentally **imitative**, and demonstrations leave information on the
table — hence comparisons.

---

## 2. Offline preference optimization — compare

**Offline** = the policy trains on a fixed dataset: responses were *not* sampled
from the current policy, feedback is human or AI labels, and many gradient steps
run against that frozen data. Data can come from anywhere; no rollouts required.
Systems consequence: **no inference engine in the training loop** — this is a
pure training workload, unlike [§3](#3-online-post-training--explore).

One record is `[prompt x, winner y_w, loser y_l]`. Naturally-occurring data works:
the **Stanford Human Preferences (SHP)** dataset (Ethayarajh et al. 2022) turned
Reddit upvote patterns into 385K pairwise comparisons over 18 subject areas, with
*collective* rather than single-annotator judgments — the only academic dataset
used to post-train Meta's Llama 2. (Reddit's subsequent data-licensing deals
totalled ~$203M in aggregate over 2–3 year terms, per its S-1.)

### From RLHF to DPO

Canonically preferences are assumed to arise from a **Bradley–Terry** model of
human utility. Classic **RLHF** (Ouyang et al. 2022) is two learned models and
two stages: preferences → reward model (scalar score) → optimize the policy to
maximize expected reward.

**DPO** (Rafailov et al. 2023) removes the explicit reward-model stage — it
optimizes the policy directly on preferences, and in theory minimizing the DPO
loss recovers the same optimal model. It optimizes a **relative likelihood
margin**: preferred response's relative likelihood up, rejected down, against a
frozen reference policy (π_ref = π_θ at the start), with β controlling drift.
The implicit reward is the log-ratio to the reference.

Most offline pairwise objectives share that structure:

| Objective | Variation |
|---|---|
| **DPO** | Logistic relative margin |
| **IPO** (Azar et al. 2024) | Squared target margin |
| **SimPO** (Meng et al. 2024) | Length-normalized; **no reference model** |
| **ORPO** (Hong et al. 2024) | SFT plus an odds-ratio margin |

Dropping the reference model (SimPO, ORPO) is also a memory win — a reference
policy is a second full copy of the weights resident during training.

They also share pathologies. **Likelihood displacement** (Razin et al. 2025): the
*relative* margin can widen while the absolute log-likelihood of the chosen
response falls — both probabilities drop, the preferred one just drops less.

### KTO — outcomes instead of pairs

**KTO** (Ethayarajh et al. 2024) judges each response separately as
👍 desirable / 👎 undesirable — no pairing required. Its loss embeds a
**prospect-theoretic utility function** with a reference point separating gains
from losses: desirable → log-likelihood up, undesirable → down, against a
policy-wide reference point *z₀*.

That absolute anchor is what paired objectives lack, and it changes the pipeline:

- **Paired (DPO):** base → **SFT** → DPO. SFT must come first — it makes preferred
  responses probable in absolute terms; DPO only manages the relative margin,
  even if both probabilities fall.
- **KTO:** base → KTO. Desirable labels push likelihood up, undesirable push down.
  **No SFT prerequisite.**

### Feedback format determines what can be learned

| | SFT | DPO | KTO |
|---|---|---|---|
| Record | prompt + target | prompt + winner + loser | prompt + response + label |
| Signal | exact target | relative winner | desirable / undesirable |
| Absolute anchor | yes | **no** | yes |
| SFT warm start | — | required | not required |

All three are **offline**: none of them enable exploration.

---

## 3. Online post-training — explore

**Online** = the current policy generates fresh responses; the policy explores
*and* learns, and the training distribution moves with the model. The loop:

```
prompts → current policy → rollouts → reward → update → (new policy → new rollouts → …)
```

The reward can come from anywhere. **This is the workload shape that makes RL
post-training a systems problem**: every step interleaves *inference*
(rollout generation — decode-bound, see
[LLM inference model](../../modeling/llm_inference.md)) with *training*
(gradient updates — compute-bound), plus weight synchronization between the
two. Offline objectives ([§2](#2-offline-preference-optimization--compare)) have
no such coupling.

### The objective

Maximize expected reward while staying close to a reference model — either as a
KL **penalty** added to the reward, or as a **constrained optimization** (maximize
reward subject to a divergence budget). In practice heuristics often stand in for
an explicit constraint.

### REINFORCE → PPO → GRPO

**REINFORCE** (Williams 1992) turns reward into a gradient via the **advantage**
*A(x,y)* relative to a baseline: above baseline → increase log-likelihood, at
baseline → little or no update, below → decrease it.

**PPO** = REINFORCE + a learned value estimator (**critic**) + a conservative
update rule. The importance ratio measures how far the policy moved after the
rollout; the advantage says whether the action beat the baseline; **clipping** is
a *one-sided brake* — careful about rewarding big winners, but still punishing
big losers.

**GRPO** (Shao et al. 2024, *DeepSeekMath*) keeps PPO's clipped update but drops
the critic, replacing it with a **within-prompt group baseline** (mean and std
over a group of rollouts for the same prompt). No value network means **less
memory and better low-level optimization for throughput** — at the cost of
needing a group of samples per prompt.

| | REINFORCE | PPO | GRPO |
|---|---|---|---|
| Baseline | none / simple | learned critic | group mean + std |
| Extra learned model | no | **yes (critic)** | no |
| Update constraint | none | PPO clipping | PPO clipping |
| Rollouts per prompt | one or more | one or more | **a group** |
| Main tradeoff | simple; high variance | stable; model-heavy | critic-free; **sample-heavy** |

That last row is the performance-engineering tradeoff in one line: PPO buys
stability with *memory* (a second trained network), GRPO buys it back with
*rollout throughput* — which is why GRPO-style training pushes hard on inference
serving inside the training loop.

GRPO variants target different optimization pathologies: **Dr. GRPO** (Liu et al.
2025) removes length and reward-std normalization; **DAPO** (Yu et al. 2025) adds
decoupled clipping, dynamic sampling, length-aware rewards; **GSPO** (Zheng et al.
2025) moves importance ratios and clipping to the *sequence* level; also SAPO
(Gao et al. 2025).

### Credit assignment

In practice most rewards are **sequence-level** — token-level credit assignment
largely does not work well. The same outcome weight touches every token, because
feedback is **sparse** (reward arrives only at the end), the **horizon is long**
(many choices precede the outcome), and **variance is high** (good and bad steps
move together).

---

## 4. Rewards & environments — verify

> Post-training is a **principal–agent problem with an extremely capable agent**.
> We can only optimize what we can see.

What we *want* (correctness, helpfulness, safety, long-run value) vs. what we
*measure* (human feedback, unit tests, self-evaluation, observed outcomes) vs.
what is *learned* (behavior that increases the measured reward).

### Verifiability and grindability

Reward signals differ on two axes — **verifiability** (how easy is it to check
correctness?) and **grindability** (how fast and repeatable is feedback?).
Coding and formal proofs sit in the easy/fast corner; CS papers and sales are
verifiable-ish but lumpy; econ papers and clinical care are hard to check *and*
slow.

| | Programmatic | Learned verifier | Human / real world |
|---|---|---|---|
| Marginal cost | ~zero | low | high |
| Latency | milliseconds | seconds | days to years |
| Attempts | millions | many | few |
| Main risk | missing tests | proxy hacking | noise, liability, … |

**A cheap verifier turns inference compute into training data.** That single
sentence is why RLVR reshaped the post-training compute budget: rollout
generation, not gradient computation, becomes the thing you scale.

### RLVR

**RL with verifiable rewards** replaces the classic learned reward model with a
checker: reward comes from a program, rule, or simulator; a **binary pass/fail is
enough** to rank many sampled responses; and the RL core is unchanged —
REINFORCE, PPO, GRPO all accept these rewards (Lambert et al. 2024, *Tülu 3*).
The loop is one hard prompt → many rollouts → cheap verifier tests every attempt
→ reinforce what passed (DeepSeek-AI 2025, *DeepSeek-R1*). Programming is high
ROI with positive spillover to other capabilities.

**RLVR also scales more predictably than RLHF.** With a verifiable reward, early
fits predict the later 100,000-GPU-hour run; with a learned proxy reward, more
sampled responses help but quickly plateau (Khatri et al. 2026, *Scaling RL
Compute for LLMs*; Hou et al. 2024, *Does RLHF Scale?*). For a
capacity-planning discipline, that predictability is the whole ballgame — it is
the RL analogue of a [scaling law](../../references/index.md#performance-modeling)
you can budget against.

### Overoptimization

Optimization pressure exposes every gap between proxy reward and goal
(Gao, Schulman & Hilton 2022): **early**, the proxy genuinely improves behavior;
**later**, the policy discovers the proxy's blind spots; **too far**, measured
reward keeps rising while actual quality falls. There is a best stopping point,
and it is before the proxy's maximum.

**Process rewards** (Lightman et al. 2023, *Let's Verify Step by Step*) promise
better credit assignment — labeling plan/retrieve/reason/act/result individually
rather than one label on the outcome — but remain uncommon at frontier scale as
far as is publicly known: one label gives ambiguous blame, many labels require
stronger assumptions. Modern systems **combine** rewards rather than choosing
one: verifier (tests, exact answers, constraints) + rubric (checklist for soft
attributes) + judge (learned model scoring open-ended quality) + human (audits,
escalations, real outcomes).

### Environments

An **RL environment is a sandbox: task + data + interface + tests.** Example: an
Excel clone loaded with `sales.xlsx`, a task ("create a summary sheet with
quarterly revenue by region, save the file"), an interface (click, type,
formulas, sheets), and tests (correct totals, right quarters, formulas used, file
opens) producing a scalar reward.

A useful environment must supply: **realistic state** (files, apps, people, latent
facts), **tools** (APIs with realistic permissions and failure modes), a **task
generator** (fresh problems at the policy's current frontier), a **simulator**
(world and user responses to actions), a **verifier** (checks that resist
shortcuts and partial compliance), and **reproducibility** (reproducible starting
states, contained side effects).

**The bottleneck is shifting from examples to environments:**

| Era | Data discipline |
|---|---|
| Pre-training | Collect any and all text; quantity > quality |
| SFT + preferences | Curate examples and judgments; quality > quantity |
| Agentic RL | **Design realistic environments and grind away at them** |

Systems-wise this means the training cluster now hosts a fleet of stateful,
side-effecting sandboxes alongside the accelerators — a scheduling and isolation
problem more than a FLOPs problem (see [Scheduling](../../scheduling/index.md)).

---

## 5. On-policy distillation — transfer

**Offline distillation** = SFT of a student on a teacher's trajectories
(Hinton et al. 2015). Its flaw: a student that only imitates never learns to
**recover from its own mistakes**. Training data follows the teacher's path;
at deployment one small student error produces an unseen prefix and the error
compounds (Agarwal et al. 2024).

**On-policy distillation (OPD)** asks the teacher about the *student's* mistakes:

1. **Student rollout** — generate the trajectory the student would actually produce.
2. **Teacher score** — evaluate every next-token distribution along that trajectory.
3. **Student update** — move toward the teacher at the states the student visits.

Formally the objective makes student and teacher distributions identical: the
student chooses which prefixes matter (**on-policy states**), the correction is a
DPO/KTO-like log-ratio now applied at **every token** (**dense reward**), and it
simplifies to a **KL divergence per prefix**.

A wrong rollout still carries signal. Given "ice cubes are added to a hot frying
pan; how many remain after three minutes?", a student that writes *"assume the
cubes do not melt"* and then computes 4 + 5 + 11 = 20 is predictable once it
adopts the wrong premise. **RLVR says "wrong"; OPD identifies *where*.**

| | RLVR | OPD |
|---|---|---|
| Signal per rollout | O(1) — reward arrives once, at the end | **O(T)** — teacher scores every visited prefix |
| Variance | high | lower (many token-level corrections) |
| Search | must rediscover the strategy | copies a discovered strategy |

Once RL has *found* a policy, OPD can copy it far more cheaply: in the Thinking
Machines Lab experiment (Lu & Thinking Machines Lab 2025), **7–10× fewer gradient
steps and 50–100× less compute**.

**OPSD** (Zhao et al. 2026) makes the student its own teacher by giving the
teacher role privileged information: same weights, but the teacher sees the
verified solution *y\** while the student sees only the problem *x*.

That asymmetry is why distillation is a key part of **parallelizing
post-training**: a shared base model → multiple experts trained by expensive,
sparse RL search (math, code, agents, safety) → one unified model via **dense**
on-policy distillation transfer. Expensive search happens in parallel and once;
the transfer is cheap.

---

## 6. World adaptation — anticipate

Post-training usually treats the environment as **fixed**: train on a distribution
of tasks and environments, deploy the policy, evaluate behavior as if the
surrounding world were exogenous. Once agents matter, that assumption breaks:

1. **Agent decisions** — models search, rank, recommend, purchase, allocate attention.
2. **Economic payoffs** — those decisions change sales, visibility, prices, access.
3. **Environment adapts** — people redesign content, interfaces, and signals in response.
4. **Behavior shifts** — the same model now faces a different decision environment.

This is a **feedback loop, not ordinary covariate shift**.

### Mecha-nudges

A **mecha-nudge** (Frey & Ethayarajh 2026) is a transformation of the environment
that (a) systematically changes AI-agent behavior by increasing
**machine-usable information**, while (b) not materially degrading the
environment for humans — it preserves **human-usable information**.

Example: the same product listing rewritten from prose ("A lovely handmade mug
for coffee or tea. About twelve ounces. Safe in the dishwasher.") into structured
attributes (`MATERIAL rare stoneware · CAPACITY 12 oz · CARE dishwasher safe`).
Unchanged for humans, easier for an agent to parse — and possibly seeded with
terms agents over-index on (e.g. *rare*).

Formally: *X* is the shared decision environment and *τ* the transformation;
*Y_M* is the machine decision with *I_M* measuring machine-usable information;
*Y_H* is the human decision with *ε* the tolerated information loss. *Y_M* and
*Y_H* are **constructed** — they are the decisions we want to study.

| Mechanism | What changes? | Primary target | Human constraint? |
|---|---|---|---|
| **Mecha-nudge** | Environment only | AI behavior | **Yes — by definition** |
| Prompt injection | Model instructions + choice set | AI behavior | No requirement |
| Traditional SEO | Human- and machine-readable context | Human traffic / search rank | Not defining |
| Adversarial example | Input surface | Model failure | Not defining |

### Evidence: Etsy

Etsy is a natural setting for observing environments adapting to AI. Post-ChatGPT
listings carry more machine-usable information about agentic curation (>40% of the
maximum possible increase), robust across prompts, labels, model families,
controls, and placebo settings — and stronger where using AI is less taboo.

Mecha-nudged listings are associated with more commercial success, **but only in
the post-ChatGPT era**. Accounting for seller fixed effects, per one bit of extra
machine-usable information:

| Listing group | Change in # reviews |
|---|---|
| Agent-selected · pre-ChatGPT | −18.1%\*\*\* |
| Agent-selected · **post-ChatGPT** | **+43.5%**\*\*\* |
| Agent-rejected · pre-ChatGPT | −12.5%\*\*\* |
| Agent-rejected · post-ChatGPT | −5.5%\*\*\* |

### Implications for post-training

| Risk | Post-training response |
|---|---|
| Strategic manipulation | Train against adversarial environment designers |
| Human–machine tradeoffs | Multi-objective rewards for agent *and* human welfare |
| Continual adaptation | Monitor deployment data and refresh the policy |

Open questions the lecture closes on: **identification** (how to causally identify
mecha-nudges in deployed systems), **equilibrium** (how agents and environments
co-evolve after repeated adaptation), **welfare** (who gains and loses), and
**governance** (how to train and regulate agents facing mecha-nudges).

> Post-training must prepare models for a world that reacts to them.

---

## Performance-engineering takeaways

Reading the stack as a systems workload rather than an ML recipe:

| Stage | Workload shape | What it stresses |
|---|---|---|
| SFT | Training (fwd+bwd), loss-masked | Same as [training](../training/index.md); small token count, quality-bound not compute-bound |
| DPO / IPO / ORPO | Training, **+ reference model resident** | Memory: two weight copies (SimPO/ORPO drop the reference) |
| KTO | Training, unpaired records | Same memory profile; cheaper data collection |
| PPO | **Inference + training interleaved**, + critic | Memory: policy + reference + critic + optimizer state |
| GRPO | Inference + training, group rollouts | **Rollout throughput** — decode-bound generation dominates |
| RLVR | Inference + verifier fleet | Rollout throughput *and* verifier/sandbox capacity |
| Agentic RL | Inference + stateful environments | Environment scheduling, isolation, reproducibility |
| On-policy distillation | Student rollout + teacher forward per token | Two models resident; 7–10× fewer steps than RL search |

The through-line: as post-training moves right along this table, the bottleneck
migrates from **gradient FLOPs** to **rollout generation** and then to
**environment capacity**. Modern RL post-training clusters look less like a
training job and more like a co-located training + serving + sandbox system.

---

## See Also

- [Model preparation](model-preparation.md) — post-training in the *efficiency* sense (quantization, LoRA, compression distillation)
- [Training workloads](../training/index.md) — the forward/backward compute profile SFT and the update steps inherit
- [LLM inference analytical model](../../modeling/llm_inference.md) — the decode-bound cost of rollout generation
- [Scheduling](../../scheduling/index.md) — co-locating rollout serving, training, and environment sandboxes
- [MFU](../../modeling/mfu.md) — the efficiency metric for the training half of the loop
- [References](../../references/index.md) — full citation list
