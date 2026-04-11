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

### Hardware Architecture
- **A100 Architecture** — NVIDIA (2020). NVIDIA A100 Tensor Core GPU Architecture Whitepaper.
- **H100 Architecture** — NVIDIA (2022). NVIDIA H100 Tensor Core GPU Architecture Whitepaper.
- **TPUv4** — Jouppi et al. (2023). "TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning." *ISCA 2023*.

### Analytical Modeling
- **LLM Inference Math** — Sheng et al.; Kipply (2023). "Transformer Inference Arithmetic." (blog post)
- **Megatron-LM** — Narayanan et al. (2021). "Efficient Large-Scale Language Model Training on GPU Clusters." *SC 2021*.


### Large Scale
- **Ultra Scale Playbook** Training LLM on GPU clusters (https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=high-level_overview)
- **Scaling LLMs on TPUs** — JAX-ML (2024). "Scaling Book: a guide to LLM scaling on TPU/JAX." (https://github.com/jax-ml/scaling-book)

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

## Useful Online Resources

- Kipply's "Transformer Inference Arithmetic" blog
- Tim Dettmers' GPU blog (quantization, memory)
- Harm de Vries' "Making Deep Learning Go Brrrr" series
- NVIDIA DLPerf benchmarks

---

## See Also

- [Hardware specs](../hardware/index.md)
- [Modeling](../modeling/index.md)
