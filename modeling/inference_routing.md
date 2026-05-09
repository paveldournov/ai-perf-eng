# LLM Inference Routing

← [Modeling Index](index.md)

---

## Why HTTP Load Balancers Break for LLM Serving

Traditional load balancers (round-robin, least-connections, consistent hashing) assume **stateless, interchangeable backends and independent requests**. LLM inference violates every one of these assumptions.

### The Four Failure Modes

| Assumption violated | What it means for routing |
|---|---|
| **Stateless backends** | GPU pods retain KV caches after a response; the same prompt on a cached pod takes milliseconds, on a cold pod takes seconds (~80× difference) |
| **Interchangeable backends** | Prefill is compute-bound; decode is memory-bandwidth-bound — a pod optimized for one phase is wrong for the other |
| **Independent requests** | Multi-turn conversations share a prefix (system prompt + history); routing turn N+1 away from turn N's pod forces full KV recomputation |
| **Single-backend dispatch** | Disaggregated prefill/decode requires coordinating two pods per request; HTTP L7 picks exactly one |

---

## KV Cache State and Prefix-Aware Routing

After a response completes, an inference engine typically **retains the KV cache** for that session in GPU HBM. A subsequent request that shares a prefix with a cached session can skip prefill for the shared tokens entirely.

**Latency impact of cache state:**
- Cache hit on prefix: serve KV from HBM, skip prefill FLOPs → latency in milliseconds
- Cache miss: recompute all prefix tokens → seconds of prefill at full compute cost

Routing correctly requires knowing, in microseconds and under concurrent updates, "which of N pods has these blocks cached?" A simple hashmap with a mutex is insufficient at this query rate.

Prefix-aware routing uses **block-level hashing** (binary search over cumulative block hashes) and sharded bitmaps to answer cache-affinity queries at serving latency.

---

## Hardware Specialization: Prefill vs. Decode

| Phase | Compute profile | Bottleneck | Optimization target |
|---|---|---|---|
| Prefill | Dense matmul over all input tokens in parallel | FLOP-bound (high arithmetic intensity) | Maximize FLOP utilization |
| Decode | One token at a time, autoregressive | HBM bandwidth-bound (low arithmetic intensity) | Maximize memory bandwidth |

A pod configured for prefill (e.g., large tile sizes, compute-heavy scheduling) underperforms on decode and vice versa. **Disaggregated serving** deploys separate pod pools for each phase:

1. A prefill pod processes the prompt and builds the KV cache
2. The KV cache is transferred to a decode pod
3. The decode pod generates the output tokens

This requires the router to make **two coordinated routing decisions per request**, not one — a capability outside standard HTTP load balancers.

---

## Conversation Continuity

Multi-turn workloads dominate production LLM traffic. Each turn N+1 shares a growing prefix with turn N (system prompt + prior turns). **Session affinity** is therefore not a convenience — it is a first-order latency requirement.

When a session is evicted from a pod's KV cache (pod restart, rebalancing, or naive round-robin routing), the engine must re-prefill the entire conversation history from scratch. At typical conversation lengths this dominates response time.

Achieving session affinity requires the router to track active session → pod bindings and to incorporate cache-hit probability into scoring, not just connection counts.

---

## Three-Layer Router Architecture (Modular approach)

Modular's analysis decomposes an LLM-aware router into three cooperating layers:

### Data Layer
Tracks LLM-specific state (which blocks are cached on which pods) at microsecond read latency. Answers: "which pods have the largest prefix overlap with this request?" Must handle concurrent updates and pod churn.

### Decision Layer
Expresses routing policies as **composable plugins**: filters → scorers → picker. Each plugin is a typed, testable component; policies are assembled from primitives rather than forked per deployment pattern.

Example pipeline:
```
filter(healthy pods)
→ score(prefix cache overlap)
→ score(current load, tiebreak)
→ score(circuit-breaker state)
→ pick(best)
```

### Execution Layer
Orchestrates **multi-step request flows**. Single-backend dispatch (standard serving) is the degenerate case (one pod, one step). Disaggregated prefill/decode is (prefill pod → KV transfer → decode pod), handled without writing new HTTP handlers for each topology.

---

## Design Principle

> A new deployment pattern should become a **profile you assemble** from composable primitives, not a strategy you fork.

This means: filters, scorers, and execution steps are independently testable and reusable; the router validates compositions at build time rather than under live traffic.

---

## See Also

- [LLM inference model](llm_inference.md) — KV cache sizing, prefill/decode latency models
- [Modular blog series (Part 1)](https://www.modular.com/blog/why-llm-inference-needs-a-new-kind-of-router-part-1) — source; Part 2 covers data structures, Part 3 covers decision/execution layers
- [PagedAttention / vLLM](../references/index.md) — block-level KV cache management that enables prefix sharing
