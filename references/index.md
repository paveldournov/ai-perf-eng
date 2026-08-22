---
type: Index
title: References
description: Curated papers, tools, datasets, and external resources for the knowledge base.
tags: [references, papers, citations]
timestamp: 2026-08-22T00:00:00-07:00
---

# References

← [Back to README](../README.md)

---

## Foundational Papers

### Performance Modeling
- **Roofline Model** — Williams, Waterman, Patterson (2009). "Roofline: An Insightful Visual Performance Model for Multicore Architectures." *CACM*. [dl.acm.org/doi/10.1145/1498765.1498785](https://dl.acm.org/doi/10.1145/1498765.1498785)
- **Scaling Laws** — Kaplan et al. (2020). "Scaling Laws for Neural Language Models." *arXiv:2001.08361*. [arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361)
- **Chinchilla** — Hoffmann et al. (2022). "Training Compute-Optimal Large Language Models." *arXiv:2203.15556*. [arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556)

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

### Mixture-of-Experts Systems
- **MoE-CAP** — Jiang, Fu, Mai, Ponti et al. (2024). "MoE-CAP: Benchmarking Cost, Accuracy and Performance of Sparse Mixture-of-Experts Systems." *arXiv:2412.07067*. [arxiv.org/abs/2412.07067](https://arxiv.org/abs/2412.07067) (defines S-MFU / S-MBU; see [MoE efficiency](../workloads/moe.md))
- **DeepSeek-V3** — DeepSeek-AI (2024). "DeepSeek-V3 Technical Report." *arXiv:2412.19437*. [arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437) (671B/37B MoE; auxiliary-loss-free load balancing; Multi-Token Prediction; ~2.788M H800 GPU-hours)
- **MegaScale-MoE** — ByteDance (2025). "MegaScale-MoE: Large-Scale Communication-Efficient Training of MoE Models in Production." *arXiv:2505.11432*. [arxiv.org/abs/2505.11432](https://arxiv.org/abs/2505.11432) (352B on 1,440 Hopper GPUs; 1.41M tok/s; 1.88× over Megatron-LM)
- **Scalable MoE with Megatron Core** — NVIDIA et al. (2026). "Heterogeneous Parallelism Mappings (MoE Parallel Folding)." *arXiv:2504.14960 / 2603.07685*. [arxiv.org/abs/2504.14960](https://arxiv.org/abs/2504.14960)
- **DeepEP / Hybrid-EP / NCCL EP** — NVIDIA & DeepSeek-AI. Device-initiated (IBGDA/TMA) expert-parallel communication. [github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) ; NCCL EP *arXiv:2603.13606* — [arxiv.org/abs/2603.13606](https://arxiv.org/abs/2603.13606)
- **Piper** — ORNL / Frontier (2026). "Piper: Efficient Large-Scale MoE Training via Resource Modeling and Pipelined Hybrid Parallelism." *arXiv:2605.05049*. [arxiv.org/abs/2605.05049](https://arxiv.org/abs/2605.05049) ; [github.com/rednote-ai/Piper](https://github.com/rednote-ai/Piper)
- **DisagMoE** — Zeng et al., UC Berkeley & Microsoft Research (2026). "DisagMoE: Computation-Communication Overlapped MoE Training via Disaggregated AF-Pipe Parallelism." *arXiv:2605.11005*. [arxiv.org/abs/2605.11005](https://arxiv.org/abs/2605.11005)
- **SGLang-JAX / Fused MoE V2 (Ling-2.6-1T on TPU)** — LMSYS (2026). "Optimizing Ling-2.6-1T on TPU with SGLang-JAX." Comm/compute-overlapped fp8 MoE kernel; in-kernel shared expert; fp8 activation quant for all-to-all; MLA+GLA hybrid backbone on TPU v7x. [www.lmsys.org/blog/2026-06-17-ling-2-6-tpu/](https://www.lmsys.org/blog/2026-06-17-ling-2-6-tpu/) (see [MoE case study](../workloads/moe.md#serving-case-study-fused-moe-v2-on-tpu-ling-26-1t))

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

See [Behavioral post-training](../workloads/post-training/behavioral-post-training.md) for
how these fit together as a stack, and for the workload shape each stage implies.

### Analytical Modeling
- **LLM Inference Math** — Sheng et al.; Kipply (2022). "Transformer Inference Arithmetic." (blog post) [kipp.ly/transformer-inference-arithmetic/](https://kipp.ly/transformer-inference-arithmetic/)
- **Megatron-LM** — Narayanan et al. (2021). "Efficient Large-Scale Language Model Training on GPU Clusters." *SC 2021*. [arxiv.org/abs/2104.04473](https://arxiv.org/abs/2104.04473)


### Large Scale
- **Ultra Scale Playbook** Training LLM on GPU clusters ([huggingface.co/spaces/nanotron/ultrascale-playbook?section=high-level_overview](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=high-level_overview))
- **Scaling LLMs on TPUs** — JAX-ML (2024). "Scaling Book: a guide to LLM scaling on TPU/JAX." ([github.com/jax-ml/scaling-book](https://github.com/jax-ml/scaling-book))

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
