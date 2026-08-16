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
- When a brand-new top-level section is created, add it to **both** `README.md`
  (the Sections table) **and** `_sidebar.md` (the website's nav). Ordinary new
  pages inside an existing section need no `_sidebar.md` change — they surface
  through that section's `index.md`.

### 4. Commit & publish

Commit to `main` with a short imperative message: `Add <thing> to <section>`,
then push. The knowledge base is served as a [docsify](https://docsify.js.org)
site (`index.html` at the repo root) that renders the markdown files live — there
is **no build step**, so a pushed page appears on the site automatically. See
[Website](#website) for details.

## Website

The knowledge base is published at
**https://paveldournov.github.io/ai-perf-eng/** via GitHub Pages (source:
`main` / root). It is a [docsify](https://docsify.js.org) single-page app —
`index.html` loads the markdown files client-side and renders them; there is no
static-site generator and nothing to build or regenerate.

Consequences for ingestion:

- **New pages just work.** Any `.md` file added under an existing section is
  live as soon as it is pushed to `main` — Pages redeploys automatically.
- **Relative markdown links** (`[GEMM](gemm.md)`, `../modeling/roofline.md`)
  resolve correctly because docsify runs with `relativePath: true`. Keep using
  normal relative links; do not rewrite them to `.html`.
- **Frontmatter** is stripped before rendering, so it never appears on the page.
- **Mermaid** fenced blocks (```` ```mermaid ````) render as diagrams.
- **Nav:** `_sidebar.md` lists only the top-level sections; update it *only* when
  adding a new top-level section (see step 3). Within-section navigation comes
  from each folder's `index.md`.
- **Per-page contents:** every page gets a right-hand table of contents built
  from its `##`/`###` headings, so use a normal heading hierarchy and don't
  hand-roll a contents list at the top of a page.
- **`.nojekyll`** at the repo root must stay — it stops GitHub Pages from running
  Jekyll, which would otherwise drop `_sidebar.md` (underscore-prefixed).

### Theme

`index.html` carries the whole design — the "Instrument" theme: dark-first, IBM
Plex Sans/Mono, teal accent, built on docsify's minimal `pure` base. All colour
lives in CSS custom properties at the top of the `<style>` block (`:root` is
dark, `html[data-theme="light"]` is light), so a palette change means editing
those two blocks and nothing else. Plugins in use: search, copy-code,
`docsify-plugin-toc` (right-hand TOC), `docsify-pagination` (prev/next), prism,
mermaid. Two behaviours are hand-rolled at the bottom of the file — tables get
wrapped in a scrolling `.table-wrap`, and the light/dark toggle suppresses CSS
transitions for one frame so custom-property colours repaint.
