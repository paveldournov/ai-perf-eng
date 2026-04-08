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
