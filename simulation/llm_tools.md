---
type: Method
title: LLM-Specific Analysis Tools
description: Purpose-built tools for analyzing LLM inference performance with built-in model/hardware configs.
tags: [simulation, llm, inference, tools, analysis]
timestamp: 2026-04-11T16:59:22-07:00
---

# LLM-Specific Analysis Tools

← [Simulation Index](index.md)

These tools are purpose-built for analyzing LLM inference performance — faster and more accessible than general simulators, with built-in LLM model configs and hardware parameter databases.

---

## LLM-Viewer {#llm-viewer}

**What it is:** An open-source tool for per-layer roofline analysis of LLM inference. Supports both a web UI and CLI.

**Workflow:**
1. Specify the LLM (model config: layers, d_model, heads, etc.)
2. Specify the hardware (peak FLOPS, HBM bandwidth)
3. Tool generates per-layer analysis: compute ops, memory footprint, roofline position

**Analysis outputs per layer:**
- FLOPs and bytes moved
- Arithmetic intensity → roofline classification (compute / BW bound)
- Peak memory consumption (tracks activation dependencies)
- Performance bottleneck and headroom

**Sweeps supported:** batch size × sequence length × hardware — generates performance curves.

**Companion paper:** Yuan et al. (2024). "LLM Inference Unveiled: Survey and Roofline Model Insights." arXiv:2402.16363.

**Links:** https://github.com/hahnyuan/LLM-Viewer

---

## LLMRoofline {#llmroofline}

**What it is:** A lightweight tool to compare hardware platforms for LLM inference tasks using the roofline model. Focuses on cross-platform comparison.

**Use case:** Given a model and multiple target chips, quickly plot where each operation falls on each chip's roofline — useful for hardware selection.

**Links:** https://github.com/feifeibear/LLMRoofline

---

## LLM Inference Arithmetic (Kipply / hand-rolled)

For cases where a tool is overkill, the first-principles arithmetic in [modeling/llm_inference.md](../modeling/llm_inference.md) can be done in a spreadsheet or a short Python script:

```python
# Decode latency lower bound (BW-bound)
params = 70e9          # 70B model
bw = 3.35e12           # H100 HBM BW (bytes/s)
dtype_bytes = 2        # BF16

t_per_token = (2 * params * dtype_bytes) / bw
print(f"{t_per_token*1000:.2f} ms/token")   # ~41 ms/token at batch=1
```

---

## Comparison

| Tool | Input | Output | Best for |
|------|-------|--------|----------|
| LLM-Viewer | Model config + HW params | Per-layer roofline report | Detailed layer-level analysis |
| LLMRoofline | Model + multiple HW | Cross-HW comparison chart | HW selection |
| Manual arithmetic | Back-of-envelope | Single-number estimate | Quick feasibility check |
| Accel-Sim | CUDA trace | Cycle-accurate perf | Microarch-level detail |

---

## See Also

- [LLM inference analytical model](../modeling/llm_inference.md)
- [Roofline model](../modeling/roofline.md)
- [Characterization](../characterization/index.md)
