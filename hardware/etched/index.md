---
type: Index
title: Etched — Transformer-Specific Accelerators
description: Landing page for Etched's hardware — the Sohu transformer-only inference ASIC.
tags: [etched, sohu, asic, transformer, inference]
timestamp: 2026-07-25T00:00:00-07:00
---

# Etched — Transformer-Specific Accelerators

← [Hardware Index](../index.md)

Etched builds **transformer-only** inference silicon. Rather than a programmable
accelerator that can run any model, its chips **hardwire the transformer dataflow
into the die** — trading all generality for throughput and utilization on the one
architecture that dominates today's LLM serving. This is the datacenter-throughput
mirror of the [Apple ANE](../apple/ane.md)'s fixed-function, edge-efficiency bet.

---

## Chip Coverage

| Chip | Type | Primary Use |
|------|------|-------------|
| [Sohu](sohu.md) | Transformer-only inference ASIC (dual systolic + attention engines) | High-batch LLM inference |

---

## See Also

- [Hardware family overview](../index.md)
- [Roofline parameters by chip](../roofline_params.md)
- [Apple ANE](../apple/ane.md) — the other fixed-function accelerator covered here
