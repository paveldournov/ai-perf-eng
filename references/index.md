---
type: Index
title: References
description: Curated papers, tools, datasets, and external resources for the knowledge base.
tags: [references, papers, citations]
timestamp: 2026-05-08T23:08:42-07:00
---

# References

← [Back to README](../README.md)

---

## Foundational Papers

### Performance Modeling
- **Roofline Model** — Williams, Waterman, Patterson (2009). "Roofline: An Insightful Visual Performance Model for Multicore Architectures." *CACM*.
- **Scaling Laws** — Kaplan et al. (2020). "Scaling Laws for Neural Language Models." *arXiv:2001.08361*.
- **Chinchilla** — Hoffmann et al. (2022). "Training Compute-Optimal Large Language Models." *arXiv:2203.15556*.

### LLM Inference Efficiency
- **FlashAttention** — Dao et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention." *NeurIPS 2022*.
- **FlashAttention-2** — Dao (2023). "FlashAttention-2: Faster Attention with Better Parallelism." *ICLR 2024*.
- **PagedAttention / vLLM** — Kwon et al. (2023). "Efficient Memory Management for Large Language Model Serving." *SOSP 2023*.
- **Continuous batching** — Yu et al. (2022). "Orca: A Distributed Serving System for Transformer-Based Generative Models." *OSDI 2022*.
- **LLM Inference Routing (Part 1)** — Modular (2025). "Why LLM Inference Needs a New Kind of Router." *Modular Blog*. https://www.modular.com/blog/why-llm-inference-needs-a-new-kind-of-router-part-1

### Hardware Architecture
- **A100 Architecture** — NVIDIA (2020). NVIDIA A100 Tensor Core GPU Architecture Whitepaper.
- **H100 Architecture** — NVIDIA (2022). NVIDIA H100 Tensor Core GPU Architecture Whitepaper.
- **TPUv4** — Jouppi et al. (2023). "TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning." *ISCA 2023*. arXiv:2304.01433
- **TPU v6e (Trillium)** — Google Cloud (2024). "Introducing Trillium, sixth-generation TPUs." Blog + official docs. https://cloud.google.com/tpu/docs/v6e
- **TPU v8t / v8i** — Google Cloud (2025). "TPU 8t and TPU 8i Technical Deep Dive." https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive
- **Boardfly (TPU v8i interconnect)** — Google (2025). Hierarchical high-radix inference network; see [Boardfly notes](../hardware/tpu/boardfly.md).
- **Dragonfly topology** — Kim et al. (2008). "Technology-Driven, Highly-Scalable Dragonfly Topology." *ISCA 2008*. https://research.google.com/pubs/archive/34926.pdf (foundational design Boardfly draws from)

### Mixture-of-Experts Systems
- **MoE-CAP** — Jiang, Fu, Mai, Ponti et al. (2024). "MoE-CAP: Benchmarking Cost, Accuracy and Performance of Sparse Mixture-of-Experts Systems." *arXiv:2412.07067*. (defines S-MFU / S-MBU; see [MoE efficiency](../workloads/moe.md))
- **DeepSeek-V3** — DeepSeek-AI (2024). "DeepSeek-V3 Technical Report." *arXiv:2412.19437*. (671B/37B MoE; auxiliary-loss-free load balancing; Multi-Token Prediction; ~2.788M H800 GPU-hours)
- **MegaScale-MoE** — ByteDance (2025). "MegaScale-MoE: Large-Scale Communication-Efficient Training of MoE Models in Production." *arXiv:2505.11432*. (352B on 1,440 Hopper GPUs; 1.41M tok/s; 1.88× over Megatron-LM)
- **Scalable MoE with Megatron Core** — NVIDIA et al. (2026). "Heterogeneous Parallelism Mappings (MoE Parallel Folding)." *arXiv:2504.14960 / 2603.07685*.
- **DeepEP / Hybrid-EP / NCCL EP** — NVIDIA & DeepSeek-AI. Device-initiated (IBGDA/TMA) expert-parallel communication. github.com/deepseek-ai/DeepEP ; NCCL EP *arXiv:2603.13606*.
- **Piper** — ORNL / Frontier (2026). "Piper: Efficient Large-Scale MoE Training via Resource Modeling and Pipelined Hybrid Parallelism." *arXiv:2605.05049*. github.com/rednote-ai/Piper
- **DisagMoE** — Zeng et al., UC Berkeley & Microsoft Research (2026). "DisagMoE: Computation-Communication Overlapped MoE Training via Disaggregated AF-Pipe Parallelism." *arXiv:2605.11005*.

### Analytical Modeling
- **LLM Inference Math** — Sheng et al.; Kipply (2023). "Transformer Inference Arithmetic." (blog post)
- **Megatron-LM** — Narayanan et al. (2021). "Efficient Large-Scale Language Model Training on GPU Clusters." *SC 2021*.


### Large Scale
- **Ultra Scale Playbook** Training LLM on GPU clusters (https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=high-level_overview)
- **Scaling LLMs on TPUs** — JAX-ML (2024). "Scaling Book: a guide to LLM scaling on TPU/JAX." (https://github.com/jax-ml/scaling-book)

### Production Reliability & Fault Tolerance
- **TorchPass** — Clockwork (2025). "TorchPass: Workload Fault Tolerance." Software-based live GPU migration and network path failover for distributed AI training. https://clockwork.io/blog/torchpass-workload-fault-tolerance/

### Simulation — Distributed Systems
- **ASTRA-sim** — Rashidi et al. (2020). "ASTRA-sim: Enabling SW/HW Co-Design Exploration for Distributed DL Training Platforms." *ISPASS 2020.*
- **ASTRA-sim 2.0** — Won et al. (2023). "Modeling Hierarchical Networks and Disaggregated Systems for Large-model Training at Scale." *ISPASS 2023.*
- **Chakra** — Won et al. (2023). "Chakra: Advancing Performance Benchmarking and Co-design using Standardized Execution Traces." *arXiv:2305.14516.*
- **Heterogeneous LLM Training Sim** — (2025). "Simulating LLM Training Workloads for Heterogeneous Compute and Network Infrastructure." *arXiv:2508.05370.*

### Simulation — Dataflow Mappers
- **Timeloop** — Parashar et al. (2019). "Timeloop: A Systematic Approach to DNN Accelerator Evaluation." *ISPASS 2019.*
- **MAESTRO** — Kwon et al. (2020). "MAESTRO: A Data-Centric Approach to Understand Reuse, Performance, and Hardware Cost of DNN Mappings." *IEEE Micro 2020.*

### Simulation — Cycle-Accurate GPU
- **Accel-Sim** — Khairy et al. (2020). "Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling." *ISCA 2020.*
- **MGPUSim** — Sun et al. (2019). "MGPUSim: Enabling Multi-GPU Performance Modeling and Optimization." *ISCA 2019.*
- **SCALE-sim v3** — (2025). "A modular cycle-accurate systolic accelerator simulator for end-to-end system analysis." *arXiv:2504.15377.*

### Simulation — LLM Inference Analysis
- **LLM Inference Unveiled** — Yuan et al. (2024). "LLM Inference Unveiled: Survey and Roofline Model Insights." *arXiv:2402.16363.* (companion tool: LLM-Viewer)
- **Roofline-Driven ML Method** — Imai (2024). "Predicting LLM Inference Latency: A Roofline-Driven ML Method." *NeurIPS 2024 MLforSystems Workshop.*
- **Hardware-Agnostic Analytical Modeling** — (2025). "Forecasting LLM Inference Performance via Hardware-Agnostic Analytical Modeling." *arXiv:2508.00904.*
- **LLM Inference on GPUs Characterization** — (2024). "A Systematic Characterization of LLM Inference on GPUs." *arXiv:2512.01644.*

---

## Tools & Frameworks

| Tool              | Purpose                             | Source                              |
| ----------------- | ----------------------------------- | ----------------------------------- |
| Nsight Compute    | NVIDIA GPU kernel profiler          | NVIDIA developer                    |
| Nsight Systems    | System-level timeline profiler      | NVIDIA developer                    |
| PyTorch Profiler  | Python-level + CUDA trace           | pytorch.org                         |
| ROCm profiler     | AMD GPU profiler                    | AMD ROCm                            |
| NCCL tests        | Collective communication benchmarks | github.com/NVIDIA/nccl-tests        |
| TransformerEngine | FP8 training library                | github.com/NVIDIA/TransformerEngine |
| vLLM              | Inference serving, paged attention  | github.com/vllm-project/vllm        |
| ASTRA-sim         | Distributed training simulator      | github.com/astra-sim/astra-sim      |
| Chakra            | Standardized ML workload traces     | github.com/mlcommons/chakra         |
| Timeloop          | DNN accelerator dataflow mapper     | timeloop.csail.mit.edu              |
| MAESTRO           | Dataflow analytical cost model      | github.com/maestro-project/maestro  |
| Accel-Sim         | Cycle-accurate NVIDIA GPU sim       | accel-sim.github.io                 |
| MGPUSim           | Cycle-accurate AMD GPU sim          | github.com/sarchlab/mgpusim         |
| SCALE-sim v3      | Cycle-accurate systolic NPU sim     | arxiv.org/abs/2504.15377            |
| LLM-Viewer        | Per-layer LLM roofline analysis     | github.com/hahnyuan/LLM-Viewer      |
| LLMRoofline       | Cross-HW LLM roofline comparison    | github.com/feifeibear/LLMRoofline   |

---

## Podcasts & Talks

- **Reiner Pope on Dwarkesh Podcast** (2024). "How LLMs are trained and deployed at scale." Covers inference latency arithmetic, batch size analysis, MoE rack layout, pipeline parallelism, total compute cost accounting, and Chinchilla over-training. Flashcard companion: https://reiner-flashcards.vercel.app/

---

## Useful Online Resources

- Kipply's "Transformer Inference Arithmetic" blog
- Tim Dettmers' GPU blog (quantization, memory)
- Harm de Vries' "Making Deep Learning Go Brrrr" series
- NVIDIA DLPerf benchmarks

---

## See Also

- [Hardware specs](../hardware/index.md)
- [Modeling](../modeling/index.md)
