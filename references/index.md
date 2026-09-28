---
type: Index
title: References
description: Curated papers, tools, datasets, and external resources for the knowledge base.
tags: [references, papers, citations]
timestamp: 2026-08-23T00:00:00-07:00
---

# References

← [Back to README](../README.md)

---

## Foundational Papers

### Performance Modeling
- **Roofline Model** — Williams, Waterman, Patterson (2009). "Roofline: An Insightful Visual Performance Model for Multicore Architectures." *CACM*. [dl.acm.org/doi/10.1145/1498765.1498785](https://dl.acm.org/doi/10.1145/1498765.1498785)
- **Scaling Laws** — Kaplan et al. (2020). "Scaling Laws for Neural Language Models." *arXiv:2001.08361*. [arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361)
- **Chinchilla** — Hoffmann et al. (2022). "Training Compute-Optimal Large Language Models." *arXiv:2203.15556*. [arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556)
- **Scaling laws (2026 update)** — Weng, L. (2026). *lilianweng.github.io*. [lilianweng.github.io/posts/2026-06-24-scaling-laws/](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/) (updated survey of LLM scaling laws past the Kaplan/Chinchilla era — the current picture of compute-optimal scaling and its limits)

### LLM Inference Efficiency
- **FlashAttention** — Dao et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention." *NeurIPS 2022*. [arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)
- **FlashAttention-2** — Dao (2023). "FlashAttention-2: Faster Attention with Better Parallelism." *ICLR 2024*. [arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691)
- **PagedAttention / vLLM** — Kwon et al. (2023). "Efficient Memory Management for Large Language Model Serving." *SOSP 2023*. [arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
- **Continuous batching** — Yu et al. (2022). "Orca: A Distributed Serving System for Transformer-Based Generative Models." *OSDI 2022*. [www.usenix.org/conference/osdi22/presentation/yu](https://www.usenix.org/conference/osdi22/presentation/yu)
- **LLM Inference Routing (Part 1)** — Modular (2025). "Why LLM Inference Needs a New Kind of Router." *Modular Blog*. [www.modular.com/blog/why-llm-inference-needs-a-new-kind-of-router-part-1](https://www.modular.com/blog/why-llm-inference-needs-a-new-kind-of-router-part-1)
- **Speculative decoding** — Leviathan, Kalman, Matias (2023). "Fast Inference from Transformers via Speculative Decoding." *ICML 2023*. [arxiv.org/abs/2211.17192](https://arxiv.org/abs/2211.17192) + Chen et al. (2023). "Accelerating LLM Decoding with Speculative Sampling." [arxiv.org/abs/2302.01318](https://arxiv.org/abs/2302.01318) (lossless draft-and-verify acceptance rule)
- **EAGLE / Medusa** — Li et al. (2024). "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty." [arxiv.org/abs/2401.15077](https://arxiv.org/abs/2401.15077) + Cai et al. (2024). "Medusa: Simple LLM Inference Acceleration via Multiple Decoding Heads." [arxiv.org/abs/2401.10774](https://arxiv.org/abs/2401.10774)
- **JetSpec (parallel tree drafting)** — Hao AI Lab (2026). "JetSpec: Parallel Tree Drafting for Speculative Decoding." Causal parallel tree drafting + tree-causal masking; up to ~9.6× on Qwen3-8B. [haoailab.com/blogs/parallel-tree-decoding/](https://haoailab.com/blogs/parallel-tree-decoding/) (see [Speculative Decoding](../modeling/speculative_decoding.md))
- **FlashAttention-3** — Shah et al. (2024). "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." *arXiv:2407.08608*. [arxiv.org/abs/2407.08608](https://arxiv.org/abs/2407.08608) (Hopper async + FP8/BF16; see [Kernel optimization](../workloads/inference/kernel-optimization.md#flashattention))
- **FlashAttention-4** — (2026). "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling." *arXiv:2603.05451*. [arxiv.org/abs/2603.05451](https://arxiv.org/abs/2603.05451) (CuTeDSL; tuned for Blackwell/B200)
- **SARATHI (chunked prefill)** — Agrawal et al. (2023). "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills." *arXiv:2308.16369*. [arxiv.org/abs/2308.16369](https://arxiv.org/abs/2308.16369) (decode-maximal batching; see [Inference optimization](../workloads/inference/optimization.md#chunked-prefill))
- **DistServe (PD disaggregation)** — Zhong et al. (2024). "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving." *OSDI 2024 / arXiv:2401.09670*. [arxiv.org/abs/2401.09670](https://arxiv.org/abs/2401.09670) (see [Inference optimization](../workloads/inference/optimization.md#prefill-decode-disaggregation))
- **Phase-decoupled power control (PD serving)** — Kim et al. (2026). "Phase-Decoupled, Model-Calibrated Power Control for Disaggregated LLM Serving." *arXiv:2609.11133*. [arxiv.org/abs/2609.11133v1](https://arxiv.org/abs/2609.11133v1) (prefill lane under an SM-clock window with a latency-guaranteeing floor; decode lane under a power cap auto-calibrated just above the measured throughput/latency cliff; +20.4% tokens/J at +3.5% mean e2e on 8× B200 serving Qwen3-Coder-480B FP8 — a Pareto improvement over NVIDIA Max-Q; claims scoped to MoE serving; see [Inference optimization](../workloads/inference/optimization.md#prefill-decode-disaggregation))
- **LLM Inference Handbook** — Modular (2026). Open handbook covering LLM inference end to end. [handbook.modular.com](https://handbook.modular.com/) (from inference basics through serving, scheduling, and kernel optimization — the kernel-optimization track's GPU architecture fundamentals chapter is a compact SM/memory-hierarchy/occupancy refresher for kernel work)
- **Understanding Transformers and Attention Mechanisms** — Serret, M.F. (2026). "An Introduction for Applied Mathematicians." *arXiv:2604.00965*. [arxiv.org/abs/2604.00965](https://arxiv.org/abs/2604.00965) (vector-level walkthrough of multi-head attention and transformer variants, closing with the efficiency toolbox — KV caching, grouped-query attention, latent attention; a math-first on-ramp to the attention-efficiency entries)
- **Quantization and Fast Inference** — Kalyanarangan, V. (2026). *Manning*. [manning.com/books/quantization-and-fast-inference](https://www.manning.com/books/quantization-and-fast-inference) (book-length treatment of quantization for fast LLM inference — formats, calibration, deployment practice)
- **Recurrent Looped Transformer (RLT)** — (2026). Conceptual review on alphaXiv. [alphaxiv.org/abs/2609.recurrent-looped-transformer](https://www.alphaxiv.org/abs/2609.recurrent-looped-transformer) (causal encoder builds reusable global KV memory; recurrent decoder carries its final hidden state and layerwise sliding-window cache across prompt and response tokens — unbounded latent depth at constant per-token work; hardware co-design separates parallel encoder work from recurrent decode; sharp parity length-generalization results, but no LM/RL/serving measurements yet — treat as speculative)
- **Nemotron 3 Ultra** — NVIDIA Research (2026). Nemotron-3-Ultra family. [research.nvidia.com/labs/nemotron/Nemotron-3-Ultra/](https://research.nvidia.com/labs/nemotron/Nemotron-3-Ultra/) (NVIDIA's efficiency-oriented Nemotron release; watch for the architecture/efficiency techniques it bakes in)

### Hardware Architecture
- **A100 Architecture** — NVIDIA (2020). NVIDIA A100 Tensor Core GPU Architecture Whitepaper. [images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)
- **H100 Architecture** — NVIDIA (2022). NVIDIA H100 Tensor Core GPU Architecture Whitepaper. [resources.nvidia.com/en-us-tensor-core](https://resources.nvidia.com/en-us-tensor-core)
- **TPUv4** — Jouppi et al. (2023). "TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning." *ISCA 2023*. [arxiv.org/abs/2304.01433](https://arxiv.org/abs/2304.01433)
- **TPU v6e (Trillium)** — Google Cloud (2024). "Introducing Trillium, sixth-generation TPUs." Blog + official docs. [cloud.google.com/tpu/docs/v6e](https://cloud.google.com/tpu/docs/v6e)
- **TPU v8t / v8i** — Google Cloud (2025). "TPU 8t and TPU 8i Technical Deep Dive." [cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive)
- **Boardfly (TPU v8i interconnect)** — Google (2025). Hierarchical high-radix inference network; see [Boardfly notes](../hardware/tpu/boardfly.md).
- **Dragonfly topology** — Kim et al. (2008). "Technology-Driven, Highly-Scalable Dragonfly Topology." *ISCA 2008*. [research.google.com/pubs/archive/34926.pdf](https://research.google.com/pubs/archive/34926.pdf) (foundational design Boardfly draws from)
- **TPU v7x (third-party specs)** — LMSYS (2026). Specs reported via a serving benchmark: ~4.6 PFLOP/s fp8, 7.38 TB/s HBM, 1.2 TB/s ICI; see [TPU v7x notes](../hardware/tpu/tpu_v7x.md). [www.lmsys.org/blog/2026-06-17-ling-2-6-tpu/](https://www.lmsys.org/blog/2026-06-17-ling-2-6-tpu/)
- **Apple Neural Engine** — Bryngelson (2026). "Apple Neural Engine: Architecture, Programming, and Performance." *arXiv:2606.22283*. [arxiv.org/abs/2606.22283](https://arxiv.org/abs/2606.22283) — Reverse-engineered account of Apple's fixed-function fp16 NPU (A11–A18, M1–M5); fp16 datapath + wide accumulator, 2 MB working-set roofline, direct dispatch below Core ML. See [ANE notes](../hardware/apple/ane.md).
- **Marvell AI data-center connectivity portfolio** — Marvell (2026). 256-lane PCIe 6.0 scale-up switching, CXL memory expansion/pooling/compression and near-memory acceleration, Photonic Fabric™ multi-rack memory sharing, Teralynx® T100 102 Tbps Ethernet, RELIANT™ interconnect telemetry; AI Infra Summit 2026. [businesswire.com](https://www.businesswire.com/news/home/20260909516427/en/Marvell-to-Showcase-End-to-End-AI-Data-Center-Connectivity-Portfolio-at-AI-Infra-Summit-2026)

### AI Accelerator Architectures (cross-vendor)
- **AI Chip Architectures** — Peake, J. (2026). Survey of the six architectures in real deployment (NVIDIA GPU, Google TPU, AMD Instinct, Cerebras WSE, AWS Trainium, Groq LPU), each read through philosophy / architecture / scaling / software, with per-chip and per-rack comparison tables. [www.jacobpeake.com/ai-chip-architectures](https://www.jacobpeake.com/ai-chip-architectures) — digested at [AI chip architectures](../hardware/architectures.md)
- **A New Golden Age for Computer Architecture** — Hennessy & Patterson (2018/2019). Turing Lecture, ISCA 2018. The domain-specific-architecture argument, with TPU v1 as the worked example (29× CPU throughput at 80× better energy efficiency); predicted "a Cambrian explosion of novel computer architectures." *CACM* 62(2). [dl.acm.org/doi/10.1145/3282307](https://dl.acm.org/doi/10.1145/3282307)
- **Groq TSP** — Abts et al. (2020). "Think Fast: A Tensor Streaming Processor (TSP) for Accelerating Deep Learning Workloads." *ISCA 2020*. [dl.acm.org/doi/10.1109/ISCA45697.2020.00023](https://dl.acm.org/doi/10.1109/ISCA45697.2020.00023) (functional slices; deterministic, compiler-scheduled datapath)
- **Groq software-scheduled network** — Abts et al. (2022). "A Software-defined Tensor Streaming Multiprocessor for Large-scale Machine Learning." *ISCA 2022*. [dl.acm.org/doi/10.1145/3470496.3527405](https://dl.acm.org/doi/10.1145/3470496.3527405) (scheduled, not routed: a compiled Dragonfly across thousands of chips)
- **Cerebras WSE** — Lie, S. (2023/2024). "Cerebras Architecture Deep Dive" / WSE-3 disclosures, *Hot Chips*. Wafer-scale integration: 84 stitched reticle fields, 900,000 dataflow cores, 44 GB on-wafer SRAM, weight streaming from MemoryX. [cerebras.ai/product-chip](https://www.cerebras.ai/product-chip)
- **AWS Trainium / NeuronCore** — AWS Neuron documentation. NeuronCore architecture (128×128 Tensor Engine, Vector/Scalar/GPSIMD engines, SBUF + PSUM scratchpads, CC-Cores) and the NKI tile-level kernel language. [awsdocs-neuron.readthedocs-hosted.com](https://awsdocs-neuron.readthedocs-hosted.com/)
- **OCP Microscaling (MX) formats** — Open Compute Project (2023). "OCP Microscaling Formats (MX) Specification v1.0." Block-scaled FP8/FP6/FP4 shared across Blackwell, MI355X, and TPU v8. [www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- **Broadcom custom-XPU business (2026)** — The Next Platform. Custom XPU + AI networking revenue forecast $2.6B (F2022) → $230B (F2028); six XPU customers (Google/TPU, Anthropic, OpenAI "Jalapeno", Meta/MTIA, +2); Tomahawk 6/Ultra Ethernet for scale-out and low-latency scale-up coherent interconnect, Tomahawk 7 (204.8 Tb/s, 400G SerDes) taped out; ~$12B/GW XPU capex vs $18B/GW Grace-Hopper, $40B/GW Vera-Rubin. [nextplatform.com](https://www.nextplatform.com/connect/2026/09/10/broadcom-rides-rocketing-trend-for-custom-ai-accelerators/5295681)
- **AI Hardware Accelerators for LLMs: Architectures and the Memory Wall** — Patel, S., Singh, R. (2026). *arXiv:2608.28048*. [arxiv.org/abs/2608.28048](https://arxiv.org/abs/2608.28048) (survey of GPUs, TPUs, Trainium, Groq, Cerebras, FPGAs, processing-in-memory/near-memory, neuromorphic and photonic accelerators read through transformer compute structure and roofline analysis; central claim: decode is bandwidth-bound, the KV cache can rival the weights in size, and data movement dominates energy — the memory system has become the computer)
- **Vera Rubin NVL72 agentic inference: 67× better performance per dollar** — SemiAnalysis (2026). [newsletter.semianalysis.com/p/vera-rubin-nvl72-agentic-inference](https://newsletter.semianalysis.com/p/vera-rubin-nvl72-agentic-inference) (Rubin NVL72 rack economics for agentic inference workloads — the perf-per-dollar framing for the hardware-economics side of the KB)
- **Meta compute buildout** — SemiAnalysis (2026). [open.substack.com/pub/semianalysis/p/meta-compute-everyone-wants-to-be](https://open.substack.com/pub/semianalysis/p/meta-compute-everyone-wants-to-be?r=ccw1b&utm_medium=ios) (SemiAnalysis on Meta's compute strategy and scale — infrastructure context for the accelerator-demand story)

### Mixture-of-Experts Systems
- **MoE-CAP** — Jiang, Fu, Mai, Ponti et al. (2024). "MoE-CAP: Benchmarking Cost, Accuracy and Performance of Sparse Mixture-of-Experts Systems." *arXiv:2412.07067*. [arxiv.org/abs/2412.07067](https://arxiv.org/abs/2412.07067) (defines S-MFU / S-MBU; see [MoE efficiency](../workloads/moe.md))
- **DeepSeek-V3** — DeepSeek-AI (2024). "DeepSeek-V3 Technical Report." *arXiv:2412.19437*. [arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437) (671B/37B MoE; auxiliary-loss-free load balancing; Multi-Token Prediction; ~2.788M H800 GPU-hours)
- **MegaScale-MoE** — ByteDance (2025). "MegaScale-MoE: Large-Scale Communication-Efficient Training of MoE Models in Production." *arXiv:2505.11432*. [arxiv.org/abs/2505.11432](https://arxiv.org/abs/2505.11432) (352B on 1,440 Hopper GPUs; 1.41M tok/s; 1.88× over Megatron-LM)
- **Scalable MoE with Megatron Core** — NVIDIA et al. (2026). "Heterogeneous Parallelism Mappings (MoE Parallel Folding)." *arXiv:2504.14960 / 2603.07685*. [arxiv.org/abs/2504.14960](https://arxiv.org/abs/2504.14960)
- **DeepEP / Hybrid-EP / NCCL EP** — NVIDIA & DeepSeek-AI. Device-initiated (IBGDA/TMA) expert-parallel communication. [github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) ; NCCL EP *arXiv:2603.13606* — [arxiv.org/abs/2603.13606](https://arxiv.org/abs/2603.13606)
- **Piper** — ORNL / Frontier (2026). "Piper: Efficient Large-Scale MoE Training via Resource Modeling and Pipelined Hybrid Parallelism." *arXiv:2605.05049*. [arxiv.org/abs/2605.05049](https://arxiv.org/abs/2605.05049) ; [github.com/rednote-ai/Piper](https://github.com/rednote-ai/Piper)
- **DisagMoE** — Zeng et al., UC Berkeley & Microsoft Research (2026). "DisagMoE: Computation-Communication Overlapped MoE Training via Disaggregated AF-Pipe Parallelism." *arXiv:2605.11005*. [arxiv.org/abs/2605.11005](https://arxiv.org/abs/2605.11005)
- **SGLang-JAX / Fused MoE V2 (Ling-2.6-1T on TPU)** — LMSYS (2026). "Optimizing Ling-2.6-1T on TPU with SGLang-JAX." Comm/compute-overlapped fp8 MoE kernel; in-kernel shared expert; fp8 activation quant for all-to-all; MLA+GLA hybrid backbone on TPU v7x. [www.lmsys.org/blog/2026-06-17-ling-2-6-tpu/](https://www.lmsys.org/blog/2026-06-17-ling-2-6-tpu/) (see [MoE case study](../workloads/moe.md#serving-case-study-fused-moe-v2-on-tpu-ling-26-1t))
- **Analytical MoE overlap (wave-quantized)** — Liu, Cui, Pericas (2026). "Analytical Resource Management for Fine-grained MoE Computation-Communication Overlap." *arXiv:2609.07536*. [arxiv.org/abs/2609.07536v1](https://arxiv.org/abs/2609.07536v1) (wave-quantized analytical model picks the comm-CTA count and SM resource partition at launch — no candidate execution, profiling, or kernel recompilation; 2.53× geo-mean on GEMM2+GatherRS over COMET on 4× A100; integrated into COMET/FLUX; see [MoE efficiency](../workloads/moe.md))

### Post-Training & Alignment
- **InstructGPT / RLHF** — Ouyang et al. (2022). "Training Language Models to Follow Instructions with Human Feedback." *NeurIPS 2022*. [arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)
- **FLAN (instruction tuning)** — Wei et al. (2022). "Finetuned Language Models Are Zero-Shot Learners." *ICLR 2022*. [arxiv.org/abs/2109.01652](https://arxiv.org/abs/2109.01652)
- **Self-Instruct** — Wang et al. (2023). "Self-Instruct: Aligning Language Models with Self-Generated Instructions." *ACL 2023*. [arxiv.org/abs/2212.10560](https://arxiv.org/abs/2212.10560)
- **LIMA** — Zhou et al. (2023). "LIMA: Less Is More for Alignment." *NeurIPS 2023*. [arxiv.org/abs/2305.11206](https://arxiv.org/abs/2305.11206) (quality + diversity beat 16× more SFT examples)
- **DPO** — Rafailov et al. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." *NeurIPS 2023*. [arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)
- **IPO / SimPO / ORPO** — Azar et al. (2024), [arxiv.org/abs/2310.12036](https://arxiv.org/abs/2310.12036); Meng et al. (2024), "SimPO," [arxiv.org/abs/2405.14734](https://arxiv.org/abs/2405.14734); Hong et al. (2024), "ORPO," [arxiv.org/abs/2403.07691](https://arxiv.org/abs/2403.07691)
- **KTO** — Ethayarajh et al. (2024). "KTO: Model Alignment as Prospect Theoretic Optimization." *ICML 2024*. [arxiv.org/abs/2402.01306](https://arxiv.org/abs/2402.01306) (unpaired desirable/undesirable labels; no SFT prerequisite)
- **SHP (Stanford Human Preferences)** — Ethayarajh et al. (2022). 385K collective pairwise comparisons from Reddit across 18 subject areas; used to post-train Llama 2. [huggingface.co/datasets/stanfordnlp/SHP](https://huggingface.co/datasets/stanfordnlp/SHP)
- **Likelihood displacement in DPO** — Razin et al. (2025). "Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization." *ICLR 2025*. [arxiv.org/abs/2410.08847](https://arxiv.org/abs/2410.08847)
- **REINFORCE** — Williams (1992). "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning." *Machine Learning* 8. [link.springer.com/article/10.1007/BF00992696](https://link.springer.com/article/10.1007/BF00992696)
- **PPO** — Schulman et al. (2017). "Proximal Policy Optimization Algorithms." *arXiv:1707.06347*. [arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)
- **GRPO / DeepSeekMath** — Shao et al. (2024). "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." *arXiv:2402.03300*. [arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300) (critic-free group baseline — less memory, better throughput)
- **GRPO variants** — Liu et al. (2025), "Dr. GRPO," [arxiv.org/abs/2503.20783](https://arxiv.org/abs/2503.20783); Yu et al. (2025), "DAPO," [arxiv.org/abs/2503.14476](https://arxiv.org/abs/2503.14476); Zheng et al. (2025), "GSPO," [arxiv.org/abs/2507.18071](https://arxiv.org/abs/2507.18071)
- **Tülu 3 (RLVR)** — Lambert et al. (2024). "Tülu 3: Pushing Frontiers in Open Language Model Post-Training." *arXiv:2411.15124*. [arxiv.org/abs/2411.15124](https://arxiv.org/abs/2411.15124)
- **DeepSeek-R1** — DeepSeek-AI (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." *arXiv:2501.12948*. [arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)
- **Reward model overoptimization** — Gao, Schulman & Hilton (2022). "Scaling Laws for Reward Model Overoptimization." *arXiv:2210.10760*. [arxiv.org/abs/2210.10760](https://arxiv.org/abs/2210.10760) (measured reward rises while true quality falls)
- **Process supervision** — Lightman et al. (2023). "Let's Verify Step by Step." *arXiv:2305.20050*. [arxiv.org/abs/2305.20050](https://arxiv.org/abs/2305.20050)
- **Does RLHF Scale?** — Hou et al. (2024). *arXiv:2412.06000*. [arxiv.org/abs/2412.06000](https://arxiv.org/abs/2412.06000) (learned proxy rewards plateau with more samples; contrast RLVR)
- **On-policy distillation** — Agarwal et al. (2024). "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes." *ICLR 2024*. [arxiv.org/abs/2306.13649](https://arxiv.org/abs/2306.13649) + Thinking Machines Lab (2025), "On-Policy Distillation," [thinkingmachines.ai/blog/on-policy-distillation/](https://thinkingmachines.ai/blog/on-policy-distillation/) (7–10× fewer gradient steps, 50–100× less compute than rediscovering the policy by RL)
- **Knowledge distillation** — Hinton, Vinyals & Dean (2015). "Distilling the Knowledge in a Neural Network." *arXiv:1503.02531*. [arxiv.org/abs/1503.02531](https://arxiv.org/abs/1503.02531)
- **DeepSeek Elastic Compute (DSec)** — DeepSeek-AI (2026). "A Sandbox Infrastructure for Effective Agentic Training at Scale." *arXiv:2609.22978*. [arxiv.org/abs/2609.22978](https://arxiv.org/abs/2609.22978) (production sandbox platform co-designed with the RL framework: FnCall/container/microVM/VM backends behind a unified SDK, on-demand image loading from 3FS, rollout execution decoupled from preemptible GPU training, mitigations for reward hacking; one unit spans ~160 nodes serving ~3M sandboxes/day with 380k concurrent and 5k creations/s — the infra behind agentic RL at DeepSeek scale)

See [Behavioral post-training](../workloads/post-training/behavioral-post-training.md) for
how these fit together as a stack, and for the workload shape each stage implies.

### GPU Kernel Engineering
- **LeetGPU** — GPU programming practice platform. [leetgpu.com](https://leetgpu.com/) (LeetCode-style CUDA kernel challenges with an automated judge — deliberate practice for kernel optimization)
- **GPU-Puzzles** — Rush, S. "Solve puzzles. Learn CUDA." [github.com/srush/gpu-puzzles](https://github.com/srush/gpu-puzzles) (puzzles from map/reduce up to attention and FlashAttention in a Numba-CUDA style — the canonical first rung of the GPU-programming ladder)
- **How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog** — Böhm, S. [siboehm.com/articles/22/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM) (step-by-step worklog taking a naive CUDA matmul to near-cuBLAS performance — tiling, shared memory, vectorization, double buffering; the single best read on the optimization arc)
- **Develop High-Performance GPU Kernels in C++ with NVIDIA CUDA Tile** — NVIDIA Technical Blog (2026). [developer.nvidia.com/blog/develop-high-performance-gpu-kernels-in-cpp-with-nvidia-cuda-tile/](https://developer.nvidia.com/blog/develop-high-performance-gpu-kernels-in-cpp-with-nvidia-cuda-tile/) (CUDA Tile tiled abstractions for portable high-performance GPU kernels in C++; part of the tile-level kernel-language trend alongside CuTe/Pallas)
- **CuTeDSL at Perplexity** — Perplexity Research (2026). [research.perplexity.ai/articles/cutedsl-at-perplexity](https://research.perplexity.ai/articles/cutedsl-at-perplexity) (production notes on CuTeDSL — NVIDIA's Python-embedded DSL for tensor-core kernels — from Perplexity's kernel work)
- **Statically finding races in CuTe kernels / proving absence of deadlocks (SPIN)** — metaworld.me (2026). [metaworld.me/blog/public/Statically-finding-races-in-CUTE-kernels-or-Proving-absences-of-Deadlocks](https://metaworld.me/blog/public/Statically-finding-races-in-CUTE-kernels-or-Proving-absences-of-Deadlocks) (applying the SPIN model checker to statically find races and prove absence of deadlocks in CuTe kernels — verification for async, barrier-heavy kernel code)
- **Auto-Improving Kernel** — jyotilakra92. [github.com/jyotilakra92/auto-improving-kernel](https://github.com/jyotilakra92/auto-improving-kernel) (repo exploring LLM-agent loops that iteratively profile and optimize GPU kernels)
- **HAWKEYE: Hardware-Aware GPU Kernel Optimization with Minimal Supervision** — (2026). [alphaxiv.org/abs/2608.hawkeye-hardware-aware-gpu-kernel-optimization](https://www.alphaxiv.org/abs/2608.hawkeye-hardware-aware-gpu-kernel-optimization) (gives kernel-writing agents a 10-strategy × architecture taxonomy where each cell ships an expert kernel, a profiling counter that verifies the optimization fired, and a composition recipe; agent performance keeps improving over 100 turns where doc/example baselines plateau; 1.22× expert Triton FLA on Blackwell linear attention, and the composition structure — not syntax — accounts for a 35% gap vs. ablated baselines on MI350)
- **Bend** — Higher Order Company. Massively parallel GPU language. [bend-lang.com](https://bend-lang.com/) (parallel-first language compiling to GPU/CPU — an alternative programming model for GPU compute)
- **What happens when a GPU reads memory** — Doubleword (2026). [blog.doubleword.ai/what-happens-when-a-gpu-reads-memory](https://blog.doubleword.ai/what-happens-when-a-gpu-reads-memory) (the full path of a GPU memory read — coalescing, caches, HBM — from a memory-hierarchy-first angle)
- **What happens when you run a CUDA kernel** — Finn, F. [fergusfinn.com/blog/what-happens-when-you-run-a-gpu-kernel/](https://fergusfinn.com/blog/what-happens-when-you-run-a-gpu-kernel/) (launch → grid/block/warp scheduling → execution; complements the memory-read piece above)
- **What is a CUDA Device Architecture?** — Modal GPU Glossary. [modal.com/gpu-glossary/device-hardware/cuda-device-architecture](https://modal.com/gpu-glossary/device-hardware/cuda-device-architecture) (glossary entry on SM generations and compute capability — handy reference when reading kernel code)

### ML Compilers & TPU
- **A Primer to ML Compilers** — Ramesh, A. [aramesh10.github.io/ml-compilers-primer/](https://aramesh10.github.io/ml-compilers-primer/index.html) (gentle, example-driven intro to ML compilers — graph IRs, lowering, codegen — feeding into the XLA/MLIR material)
- **Programming the TPU: What Its Open-Source Compiler Already Tells You** — Ramesh, A. (hiraditya). [hiraditya.github.io/posts/tpu-through-open-source/](https://hiraditya.github.io/posts/tpu-through-open-source/) (what TPU's open-source compiler stack reveals about the hardware — compiler archaeology as a window into the accelerator; see [XLA notes](../workloads/xla_compiler.md))
- **How to use Google microbenchmarks for evaluating TPU performance** — Google Developers Blog (2026). [developers.googleblog.com/how-to-use-google-microbenchmarks-for-evaluating-tpu-performance/](https://developers.googleblog.com/how-to-use-google-microbenchmarks-for-evaluating-tpu-performance/) (practical guide to microbenchmarking TPU kernels — methodology for the characterization track)
- **Unlocking TPU performance: Deep kernel profiling with XProf** — Google Open Source Blog (2026). [opensource.googleblog.com/2026/06/unlocking-tpu-performance-deep-kernel-profiling-with-xprof.html](https://opensource.googleblog.com/2026/06/unlocking-tpu-performance-deep-kernel-profiling-with-xprof.html) (deep kernel profiling on TPU with XProf — trace → roofline → optimization loop)
- **Stop Training Blind: Scaling AI with the new OpenTelemetry-based TPU AI Telemetry Collector agent** — Google Developers (2026). [discuss.google.dev/t/stop-training-blind-scaling-ai-with-the-new-opentelemetry-based-tpu-ai-telemetry-collector-agent/375210](https://discuss.google.dev/t/stop-training-blind-scaling-ai-with-the-new-opentelemetry-based-tpu-ai-telemetry-collector-agent/375210) (OTel-based telemetry collector for TPU training — observability for large-scale training performance)
- **Making TPU Model Performance Auto-optimization work with other Agents** — Vlasenko, A. (2026). [vlasenkoalexey.github.io/2026/06/making-tpu-auto-optimization-work-with-other-agents/](https://vlasenkoalexey.github.io/2026/06/making-tpu-auto-optimization-work-with-other-agents/) (agent-driven TPU performance auto-optimization composed with other agents — the agentic tuning loop applied to TPU workloads)

### Efficiency Techniques
- **Less is More: Scaling Dynamics with Sparsity** — Unconventional AI (2026). [unconv.ai/blog/less-is-more-scaling-dynamics-with-sparsity/](https://unconv.ai/blog/less-is-more-scaling-dynamics-with-sparsity/) (how sparsity changes scaling dynamics — when and why sparse training/inference wins over dense at scale)
- **Sparser, Faster, Lighter Transformer Language Models** — *distill.pub*. [distill.pub](http://distill.pub/) (Distill's feature on sparsity and efficiency techniques for lighter transformer LMs; link resolves to the publication home)
- **LLMs-from-scratch: differential self-attention (ch04/09_dsa)** — Raschka, S. [github.com/rasbt/LLMs-from-scratch/tree/main/ch04/09_dsa](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/09_dsa) (runnable notebook/code for differential self-attention — an attention variant that cancels common-mode noise — in the LLMs-from-scratch series)
- **AI Model Co-Design: Hardware-Friendly LLM Design** — NVIDIA Technical Blog (2026). [developer.nvidia.com/blog/ai-model-co-design-hardware-friendly-llm-design/](https://developer.nvidia.com/blog/ai-model-co-design-hardware-friendly-llm-design/) (designing LLM architectures with hardware constraints in mind — the co-design loop between model architecture and accelerator)

### Courses & Resource Collections
- **Hands-On Deep Learning (15-773)** — MIT OpenCourseWare, Spring 2024. [ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/video_galleries/lecture-videos/](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/video_galleries/lecture-videos/) (full MIT lecture-video series on hands-on deep learning and LLMs)
- **GPU Perf Engineering Resources** — wafer-ai. [github.com/wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) (curated collection of GPU performance-engineering resources — blogs, papers, tools)

### Analytical Modeling
- **Understanding Latency Hiding on GPUs** — Volkov, V. (2016). PhD dissertation, UC Berkeley. UCB/EECS-2016-143. [www2.eecs.berkeley.edu/Pubs/TechRpts/2016/Archive/EECS-2016-143.pdf](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/Archive/EECS-2016-143.pdf) (analytical study of GPU multithreading as a latency-hiding mechanism; prior GPU performance models mispredict throughput by up to 1.7× on synthetic workloads — the pitfalls are in occupancy/concurrency and Little's-law reasoning; derives the warp count needed to hide latency as a function of arithmetic intensity)
- **LLM Inference Math** — Sheng et al.; Kipply (2022). "Transformer Inference Arithmetic." (blog post) [kipp.ly/transformer-inference-arithmetic/](https://kipp.ly/transformer-inference-arithmetic/)
- **Megatron-LM** — Narayanan et al. (2021). "Efficient Large-Scale Language Model Training on GPU Clusters." *SC 2021*. [arxiv.org/abs/2104.04473](https://arxiv.org/abs/2104.04473)


### Large Scale
- **Ultra Scale Playbook** Training LLM on GPU clusters ([huggingface.co/spaces/nanotron/ultrascale-playbook?section=high-level_overview](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=high-level_overview))
- **Scaling LLMs on TPUs** — JAX-ML (2024). "Scaling Book: a guide to LLM scaling on TPU/JAX." ([github.com/jax-ml/scaling-book](https://github.com/jax-ml/scaling-book))
- **Every μs Matters: Achieving Near Speed-of-Light Latency in GPU Collectives** — (2026). *arXiv:2607.16100v1*. [arxiv.org/html/2607.16100v1](https://arxiv.org/html/2607.16100v1) (pushing GPU collective latency toward the physical limit — latency rather than bandwidth as the frontier for fine-grained distributed inference; see [collective ops](../workloads/collective_ops.md))
- **Pipeline Schedule Tutor** — Yang, E. Interactive tutor for pipeline parallelism. [ezyang.github.io/pipeline-parallelism-tutor/](https://ezyang.github.io/pipeline-parallelism-tutor/#level=first-steps) (hands-on walkthrough of pipeline schedules — bubbles, 1F1B, interleaved — for distributed training)
- **Pathways on Cloud** — Google Cloud docs + AI-Hypercomputer/pathways-utils. [docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro) ; [github.com/AI-Hypercomputer/pathways-utils](https://github.com/AI-Hypercomputer/pathways-utils) (Google's Pathways orchestration for large training on AI Hypercomputer; see [Pathways notes](../scheduling/pathways.md))

### Production Reliability & Fault Tolerance
- **TorchPass** — Clockwork (2025). "TorchPass: Workload Fault Tolerance." Software-based live GPU migration and network path failover for distributed AI training. [clockwork.io/blog/torchpass-workload-fault-tolerance/](https://clockwork.io/blog/torchpass-workload-fault-tolerance/)

### Simulation — Distributed Systems
- **ASTRA-sim** — Rashidi et al. (2020). "ASTRA-sim: Enabling SW/HW Co-Design Exploration for Distributed DL Training Platforms." *ISPASS 2020.* [astra-sim.github.io/](https://astra-sim.github.io/)
- **ASTRA-sim 2.0** — Won et al. (2023). "Modeling Hierarchical Networks and Disaggregated Systems for Large-model Training at Scale." *ISPASS 2023.* [arxiv.org/abs/2303.14006](https://arxiv.org/abs/2303.14006)
- **Chakra** — Won et al. (2023). "Chakra: Advancing Performance Benchmarking and Co-design using Standardized Execution Traces." *arXiv:2305.14516.* [arxiv.org/abs/2305.14516](https://arxiv.org/abs/2305.14516)
- **Heterogeneous LLM Training Sim** — (2025). "Simulating LLM Training Workloads for Heterogeneous Compute and Network Infrastructure." *arXiv:2508.05370.* [arxiv.org/abs/2508.05370](https://arxiv.org/abs/2508.05370)

### Simulation — Dataflow Mappers
- **Timeloop** — Parashar et al. (2019). "Timeloop: A Systematic Approach to DNN Accelerator Evaluation." *ISPASS 2019.* [timeloop.csail.mit.edu/](https://timeloop.csail.mit.edu/)
- **MAESTRO** — Kwon et al. (2020). "MAESTRO: A Data-Centric Approach to Understand Reuse, Performance, and Hardware Cost of DNN Mappings." *IEEE Micro 2020.* [github.com/maestro-project/maestro](https://github.com/maestro-project/maestro)

### Simulation — Cycle-Accurate GPU
- **Accel-Sim** — Khairy et al. (2020). "Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling." *ISCA 2020.* [accel-sim.github.io/](https://accel-sim.github.io/)
- **MGPUSim** — Sun et al. (2019). "MGPUSim: Enabling Multi-GPU Performance Modeling and Optimization." *ISCA 2019.* [github.com/sarchlab/mgpusim](https://github.com/sarchlab/mgpusim)
- **SCALE-sim v3** — (2025). "A modular cycle-accurate systolic accelerator simulator for end-to-end system analysis." *arXiv:2504.15377.* [arxiv.org/abs/2504.15377](https://arxiv.org/abs/2504.15377)

### Simulation — LLM Inference Analysis
- **LLM Inference Unveiled** — Yuan et al. (2024). "LLM Inference Unveiled: Survey and Roofline Model Insights." *arXiv:2402.16363.* [arxiv.org/abs/2402.16363](https://arxiv.org/abs/2402.16363) (companion tool: LLM-Viewer)
- **Roofline-Driven ML Method** — Imai (2024). "Predicting LLM Inference Latency: A Roofline-Driven ML Method." *NeurIPS 2024 MLforSystems Workshop.* [mlforsystems.org/assets/papers/neurips2024/paper28.pdf](https://mlforsystems.org/assets/papers/neurips2024/paper28.pdf)
- **Hardware-Agnostic Analytical Modeling** — (2025). "Forecasting LLM Inference Performance via Hardware-Agnostic Analytical Modeling." *arXiv:2508.00904.* [arxiv.org/abs/2508.00904](https://arxiv.org/abs/2508.00904)
- **LLM Inference on GPUs Characterization** — (2024). "A Systematic Characterization of LLM Inference on GPUs." *arXiv:2512.01644.* [arxiv.org/abs/2512.01644](https://arxiv.org/abs/2512.01644)

---

## Tools & Frameworks

| Tool              | Purpose                             | Source                              |
| ----------------- | ----------------------------------- | ----------------------------------- |
| Nsight Compute    | NVIDIA GPU kernel profiler          | [developer.nvidia.com/nsight-compute](https://developer.nvidia.com/nsight-compute) |
| Nsight Systems    | System-level timeline profiler      | [developer.nvidia.com/nsight-systems](https://developer.nvidia.com/nsight-systems) |
| PyTorch Profiler  | Python-level + CUDA trace           | [pytorch.org](https://pytorch.org)  |
| ROCm profiler     | AMD GPU profiler                    | [rocm.docs.amd.com](https://rocm.docs.amd.com) |
| NCCL tests        | Collective communication benchmarks | [github.com/NVIDIA/nccl-tests](https://github.com/NVIDIA/nccl-tests) |
| TransformerEngine | FP8 training library                | [github.com/NVIDIA/TransformerEngine](https://github.com/NVIDIA/TransformerEngine) |
| vLLM              | Inference serving, paged attention  | [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) |
| ASTRA-sim         | Distributed training simulator      | [github.com/astra-sim/astra-sim](https://github.com/astra-sim/astra-sim) |
| Chakra            | Standardized ML workload traces     | [github.com/mlcommons/chakra](https://github.com/mlcommons/chakra) |
| Timeloop          | DNN accelerator dataflow mapper     | [timeloop.csail.mit.edu](https://timeloop.csail.mit.edu) |
| MAESTRO           | Dataflow analytical cost model      | [github.com/maestro-project/maestro](https://github.com/maestro-project/maestro) |
| Accel-Sim         | Cycle-accurate NVIDIA GPU sim       | [accel-sim.github.io](https://accel-sim.github.io) |
| MGPUSim           | Cycle-accurate AMD GPU sim          | [github.com/sarchlab/mgpusim](https://github.com/sarchlab/mgpusim) |
| SCALE-sim v3      | Cycle-accurate systolic NPU sim     | [arxiv.org/abs/2504.15377](https://arxiv.org/abs/2504.15377) |
| LLM-Viewer        | Per-layer LLM roofline analysis     | [github.com/hahnyuan/LLM-Viewer](https://github.com/hahnyuan/LLM-Viewer) |
| LLMRoofline       | Cross-HW LLM roofline comparison    | [github.com/feifeibear/LLMRoofline](https://github.com/feifeibear/LLMRoofline) |

---

## Podcasts & Talks

- **Kawin Ethayarajh — "Post-Training LLMs"** (2026). *AI and Economics Summer Institute 2026*, Chicago, Aug 6–11. Lecture covering the full behavioral post-training stack: SFT, offline preference optimization (DPO/KTO), online RL (REINFORCE/PPO/GRPO), RLVR and environments, on-policy distillation, and world adaptation / mecha-nudges. [kawine.github.io/assets/aiesi_post-training_public.pdf](https://kawine.github.io/assets/aiesi_post-training_public.pdf) — digested at [Behavioral post-training](../workloads/post-training/behavioral-post-training.md)
- **Reiner Pope on Dwarkesh Podcast** (2026). "The math behind how LLMs are trained and served." Covers inference latency arithmetic, batch size analysis, MoE rack layout, pipeline parallelism, total compute cost accounting, and Chinchilla over-training. [open.spotify.com/episode/0lQEgY6q0BczmP4oTUht1p](https://open.spotify.com/episode/0lQEgY6q0BczmP4oTUht1p) — Flashcard companion: [reiner-flashcards.vercel.app/](https://reiner-flashcards.vercel.app/)

---

## Useful Online Resources

- **AI Chip Architectures** — Jacob Peake (2026). Long-form cross-vendor survey of AI accelerator architectures: NVIDIA, TPU, AMD, Cerebras, Trainium, Groq — philosophy, microarchitecture, scale-up/scale-out fabrics, software stacks, and comparison tables. [www.jacobpeake.com/ai-chip-architectures](https://www.jacobpeake.com/ai-chip-architectures) — digested at [AI chip architectures](../hardware/architectures.md)
- **LLM Inference Handbook** — Modular (2026). "A practical guide for understanding, optimizing, scaling, and operating LLM inference systems." [handbook.modular.com](https://handbook.modular.com/) — source: [github.com/modular/llm-inference-handbook](https://github.com/modular/llm-inference-handbook) (`docs/` used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); adapted/summarized with changes). Digested across the [inference](../workloads/inference/index.md) and [post-training](../workloads/post-training/index.md) workloads subsections.
- Kipply's "Transformer Inference Arithmetic" blog — [kipp.ly/transformer-inference-arithmetic/](https://kipp.ly/transformer-inference-arithmetic/)
- Tim Dettmers' GPU blog (quantization, memory) — [timdettmers.com/](https://timdettmers.com/)
- Horace He's "Making Deep Learning Go Brrrr From First Principles" — [horace.io/brrr_intro.html](https://horace.io/brrr_intro.html)
- Adam Mainz — "GPUs have two speeds" (2026). Plain-English roofline explainer: the two ceilings (compute vs bandwidth), arithmetic intensity, and the ridge point, via a chef/runner analogy and vector-add-vs-matmul worked examples. [x.com/MainzOnX](https://x.com/MainzOnX/status/2077757143592186262) (see [Roofline model](../modeling/roofline.md))
- NVIDIA DLPerf benchmarks — [developer.nvidia.com/deep-learning-performance-training-inference](https://developer.nvidia.com/deep-learning-performance-training-inference)

---

## See Also

- [Hardware specs](../hardware/index.md)
- [Modeling](../modeling/index.md)
- [Inference workloads](../workloads/inference/index.md)

---

### Social bookmarks (self-email sweep 2026-09-27)
Thin-content social posts saved by the user; recorded as a grouped list, not individual entries.
- GPU optimization kernels — @shubh6200 (2026-09-26). [x.com/shubh6200/status/2103734500555661476](https://x.com/shubh6200/status/2103734500555661476)
- RL post — @lu__jasper (2026-09-18). [x.com/lu__jasper/status/2100601963813507268](https://x.com/lu__jasper/status/2100601963813507268)
- Apple neural engine — @anemll (2026-09-11). [x.com/anemll/status/2098438312017236149](https://x.com/anemll/status/2098438312017236149)
- Hot chips 2026 — @jasonschips (2026-09-02). [x.com/jasonschips/status/2095084434185924625](https://x.com/jasonschips/status/2095084434185924625)
- Inference and GPU performance — @akshay_pachaar (2026-08-16). [x.com/akshay_pachaar/status/2088645148590874869](https://x.com/akshay_pachaar/status/2088645148590874869)
- Profiling sgLang — @jino_rohit (2026-08-08). [x.com/jino_rohit/status/2085947942339563598](https://x.com/jino_rohit/status/2085947942339563598)
- Chips the world needs — @patricktoulme (2026-08-04). [x.com/patricktoulme/status/2084684706134561109](https://x.com/patricktoulme/status/2084684706134561109)
- From GPT2 to Kimi — @waterloo_intern (2026-07-29). [x.com/waterloo_intern/status/2081762065392541951](https://x.com/waterloo_intern/status/2081762065392541951)
- SIMD — @davidcrawshaw (2026-07-25). [x.com/davidcrawshaw/status/2080296820379402498](https://x.com/davidcrawshaw/status/2080296820379402498)
- TPU kernels deep dive — @wafer_ai (2026-07-18). [x.com/wafer_ai/status/2074958366150197499](https://x.com/wafer_ai/status/2074958366150197499)
- Flops and bw GPU — @mainzonx (2026-07-16). [x.com/mainzonx/status/2077757143592186262](https://x.com/mainzonx/status/2077757143592186262)
- Inference compute analysis — @aryatschand (2026-07-15). [x.com/aryatschand/status/2077439453258563625](https://x.com/aryatschand/status/2077439453258563625)
- LLM for apple silicon — @old_sound (2026-07-14). [x.com/old_sound/status/2076932819008242037](https://x.com/old_sound/status/2076932819008242037)
- Kernel optimization TPU — @aryatschand (2026-07-10). [x.com/aryatschand/status/2069848205974737181](https://x.com/aryatschand/status/2069848205974737181)
- TPU internal — @grigoryevko (2026-07-10). [x.com/grigoryevko/status/2069876132980076601](https://x.com/grigoryevko/status/2069876132980076601)
- Post by wafer on X — @wafer_ai (2026-07-09). [x.com/wafer_ai/status/2074654134972895551](https://x.com/wafer_ai/status/2074654134972895551)
- Post by Bastian Hagedorn on X — @hagedornbastian (2026-07-07). [x.com/hagedornbastian/status/2074509770342445375](https://x.com/hagedornbastian/status/2074509770342445375)
- Grok sglang — @h100envy (2026-07-06). [x.com/h100envy/status/2070852290878009586](https://x.com/h100envy/status/2070852290878009586)
- KV caching explained — @_avichawla (2026-06-27). [x.com/_avichawla/status/2070828078247604480](https://x.com/_avichawla/status/2070828078247604480)
- Inference engines post — @theahmadosman (2026-06-27). [x.com/theahmadosman/status/2057183854444843202](https://x.com/theahmadosman/status/2057183854444843202)
- Optimizing Ling 26-1T on TPU (SGLang) — LinkedIn (2026-06-26). [linkedin.com/posts/our-new-blog-optimizing-ling-26-1t-on-share-7473060090284146688-gJav](https://www.linkedin.com/posts/our-new-blog-optimizing-ling-26-1t-on-share-7473060090284146688-gJav/?utm_source=share&utm_medium=member_ios&rcm=ACoAAA)
- Post by Swati Gupta on X — @hrswatigupta (2026-06-08, two posts). [x.com/hrswatigupta/status/2063953304108065023](https://x.com/hrswatigupta/status/2063953304108065023) ; [x.com/hrswatigupta/status/2060304091356791118](https://x.com/hrswatigupta/status/2060304091356791118)
- LLM math tweet — @theahmadosman (2026-05-21). [x.com/theahmadosman/status/2057183854444843202](https://x.com/theahmadosman/status/2057183854444843202)
- Post by Patrick C Toulme on X — @patricktoulme (2026-05-16). [x.com/patricktoulme/status/2055709800986780028](https://x.com/patricktoulme/status/2055709800986780028)
- Post by Colin Breck on X — @breckcs (2026-05-10). [x.com/breckcs/status/2048432277299036219](https://x.com/breckcs/status/2048432277299036219)
- Post on Kernels — @theahmadosman (2026-04-26). [x.com/theahmadosman/status/2048234466519236818](https://x.com/theahmadosman/status/2048234466519236818)
