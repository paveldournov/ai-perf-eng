# CLAUDE.md — Project Instructions

This knowledge base follows the **Open Knowledge Format (OKF)**: a directory of
markdown files where the file path is the concept's identity, each file is one
concept, `index.md` provides progressive disclosure per folder, and concepts
link to each other with normal markdown links. See [OKF Conventions](#okf-conventions).

## OKF Conventions

### Frontmatter (required on every concept file)

Every `.md` file opens with YAML frontmatter. `type` is the only required field;
the rest are strongly encouraged.

```yaml
---
type: <one of the controlled vocabulary below>   # required
title: <human-readable title>
description: <one-line summary; used by agents for relevance>
tags: [<lowercase>, <kebab-case>, <facets>]
resource: <canonical external URL>               # omit for internal-only pages
timestamp: <ISO 8601, last meaningful update>
---
```

### `type` controlled vocabulary

| `type` | Use for |
|---|---|
| `Index` | Section landing / navigation page (every `index.md`, plus `README.md` and `GUIDE.md`) |
| `Hardware` | A specific accelerator, chip, board, or interconnect topology |
| `Concept` | An analytical model, formula, or principle you reason *with* (roofline, MFU, attention, GEMM) |
| `Method` | A methodology or simulation/benchmarking approach (analytical, cycle-accurate, dataflow mapper) |
| `Tool` | A concrete runnable software system or runtime (Kueue, Ray, Pathways, llm-d) |
| `Kernel` | A kernel-level code implementation (Pallas, Triton) |
| `Dataset` | Data artifacts: spec tables, profiling traces |
| `Reference` | A standalone page curating an external work or citation |

Keep the vocabulary closed — if a new source doesn't fit, prefer the closest
existing `type` over inventing one; only extend the vocabulary deliberately.

### Reserved filenames

- `index.md` — folder landing page (`type: Index`); links down to children.
- `log.md` — optional chronological change history for a folder.

## Ingesting a New Source

When the user brings a new source (paper, blog post, whitepaper, benchmark, hardware announcement), follow this process:

### 1. Decide where it lives

| What it is | Where it goes |
|---|---|
| Hardware specs / architecture | `hardware/<vendor>/` |
| Workload behavior, kernels | `workloads/` |
| Analytical models, roofline | `modeling/` |
| Simulators, tools | `simulation/` |
| Benchmarking methodology | `characterization/` |
| Citation / pointer only | `references/index.md` |

If there is enough depth for a standalone page (e.g. a deep-dive on a specific chip or simulator), create a new file. Otherwise add to an existing file or append to `references/index.md`.

### 2. Write the content

- **New file**: open with OKF [frontmatter](#frontmatter-required-on-every-concept-file) (`type` required), then the `# Title`, a `← Back to <parent>` link, `---`, and structured H2/H3 sections.
- **Existing file**: extend the relevant section; add a table row, bullet, or subsection as appropriate.
- **Reference-only**: add a bullet under the correct subsection in `references/index.md` — format: `**Title** — Author(s) (Year). "Title." *Venue*. URL`

### 3. Wire up cross-links

- Add a row or bullet to the parent section's `index.md`.
- Add "See also" links on closely related existing pages.
- Update `README.md` only when a brand-new top-level section is created.

### 4. Commit

Commit to `main` with a short imperative message: `Add <thing> to <section>`.
