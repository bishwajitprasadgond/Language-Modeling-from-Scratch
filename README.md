# Spring 2026 CS336 lectures

This repository contains the lecture materials for Stanford's **Language Modeling from Scratch (CS336)**, Spring 2026 (the 3rd offering of the course). It holds 17 lectures, the bibliography they cite, the figures they use, and an interactive study guide (`index.html`) that covers every lecture in detail.

- 9 lectures are **executable Python** (`lecture_XX.py`). You run them, and they produce a step-by-step trace of code, values and figures.
- 8 lectures are **slide decks** (`lecture_XX.pdf`).
- **`index.html`** is a single self-contained page with detailed notes, live charts, step-through diagrams, calculators and quizzes for all 17 lectures. Open it in a browser; it needs no build step.

---

## Contents

1. [Quick start: the interactive study guide](#quick-start-the-interactive-study-guide)
2. [Lecture index](#lecture-index)
3. [Course structure](#course-structure)
4. [Repository layout](#repository-layout)
5. [Executable lectures](#executable-lectures)
6. [Non-executable lectures](#non-executable-lectures)
7. [The interactive study guide in detail](#the-interactive-study-guide-in-detail)
8. [Rebuilding `index.html`](#rebuilding-indexhtml)
9. [Key formulas](#key-formulas)
10. [Dependencies](#dependencies)

---

## Quick start: the interactive study guide

Open `index.html` in any modern browser. Fonts and math rendering (MathJax) load from a CDN, so an internet connection gives the best result. All charts and diagrams are drawn by plain JavaScript embedded in the page.

If your browser blocks local files from loading images, serve the folder instead:

```powershell
python -m http.server 8765
```

Then visit `http://localhost:8765/index.html`.

| Control | What it does |
|---|---|
| Sidebar | One collapsible group per lecture. **Expand / Collapse** opens or folds them all; `≡` hides the sidebar. |
| Charts | Hover for exact values (lines show a crosshair). Click a legend chip to hide a series. **table** shows the numbers behind the chart. |
| Step-through diagrams | **prev / next / play** walk through an algorithm. Dimmed parts are inactive in the current step. |
| Calculators | Sliders recompute on every move. Log-scale sliders span several orders of magnitude. |
| Quizzes | Every lecture ends with self-check questions. **reveal all answers** opens them together. |
| Figures | Numbered per lecture (Figure 1.1, Figure 5.3, …). Click an image to zoom; `Esc` closes it. |
| Theme | `N` / `D` toggles light and dark mode. Charts use a palette tuned separately for each mode. |

---

## Lecture index

| # | Lecture | Source | Highlights in `index.html` |
|---|---|---|---|
| 1 | Course overview, the LM landscape, and tokenization | `lecture_01.py` | BPE step-through that runs real merges, tokenizer compression and trade-off calculators, FLOPs/Chinchilla calculator |
| 2 | Resource accounting: tensors, precision, FLOPs, memory | `lecture_02.py` | fp32/fp16/bf16/fp8 bit-layout explorer, training-time (6ND) and memory calculators, roofline, einsum steppers |
| 3 | LLM architecture and hyperparameters | `lecture_03.pdf` | Pre- vs post-norm, RMSNorm, SwiGLU, interactive RoPE rotation, hyperparameter ratios, model-config tables |
| 4 | Attention alternatives and mixtures of experts | `lecture_04.pdf` | Attention-mask heatmaps, linear attention and SSMs, MLA, MoE routing step-through, active-vs-total parameter and KV-cache calculators |
| 5 | GPUs | `lecture_05.pdf` | Memory hierarchy, roofline with ridge point, tile/wave quantization, online-softmax and FlashAttention steppers |
| 6 | Kernels and GPU programming: CUDA, Triton, benchmarking | `lecture_06.py` | Grid/block/thread indexing, occupancy, coalescing, kernel-fusion traffic, tiled-GEMM walk, real CUDA/Triton code |
| 7 | Parallelism: collectives, hardware, distributed training | `lecture_07.py` | All eight collectives on 2–8 ranks, ring all-reduce bandwidth, GPipe vs 1F1B schedules, DP/TP/PP communication |
| 8 | Parallelism basics at scale | `lecture_08.pdf` | ZeRO stages and FSDP memory, pipeline bubbles, tensor/sequence/expert parallelism, 3D recipes from real runs |
| 9 | Scaling laws: the basics | `lecture_09.pdf` | Power-law fitting, Kaplan vs Chinchilla, IsoFLOP curves, compute-optimal N and D calculator |
| 10 | Inference | `lecture_10.py` | Prefill vs decode intensity, KV-cache sizing (MHA/GQA/MQA/MLA/CLA), quantization, speculative decoding, PagedAttention |
| 11 | Scaling: case studies and details | `lecture_11.pdf` | muP and hyperparameter transfer, cosine vs WSD schedules, critical batch size, over-training |
| 12 | Evaluation | `lecture_12.py` | Perplexity to agent benchmarks, Elo / Bradley–Terry, pass@k, contamination detection, LMArena |
| 13 | Data I: where pretraining data comes from | `lecture_13.py` | Data-source mixes, crawling and HTML-to-text, copyright and licensing decision flow, curation pipeline |
| 14 | Data II: filtering, deduplication, mixing | `lecture_14.py` | Quality filters, Jaccard and MinHash lab, LSH S-curve, Bloom filters, mixing-ratio explorer |
| 15 | After pretraining: mid-training, SFT, RLHF, PPO, DPO | `lecture_15.pdf` | RLHF pipeline, Bradley–Terry reward model, DPO loss, KL-penalty trade-off |
| 16 | Post-training 2: RL from verifiable rewards (RLVR) | `lecture_16.pdf` | Policy gradient, PPO clipping, GRPO advantages, R1-Zero/R1 recipes, length and std biases, pass@k |
| 17 | Multimodal models | `lecture_17.py` | ViT patchify, CLIP/SigLIP contrastive loss lab, LLaVA and Qwen-VL architectures, visual-token counts |

The page also has a **roadmap** at the top and a **cheat sheet and glossary** at the end.

---

## Course structure

The course follows the life cycle of a language model:

```
Unit 1  Basics        L1 tokenization → L2 resource accounting → L3 architecture → L4 attention variants & MoE
Unit 2  Systems       L5 GPUs → L6 kernels → L7 collectives & code → L8 parallelism at scale
Unit 3  Scaling       L9 scaling laws → L11 case studies;  L10 inference;  L12 evaluation
Unit 4  Data          L13 sources & law → L14 filtering, dedup, mixing
Unit 5  Alignment     L15 SFT / RLHF / DPO → L16 RLVR → L17 multimodal
```

---

## Repository layout

| Path | What it is |
|---|---|
| `lecture_01.py` … `lecture_17.py` | Executable lectures 1, 2, 6, 7, 10, 12, 13, 14, 17 |
| `lecture_03.pdf` … `lecture_16.pdf` | Slide decks for lectures 3, 4, 5, 8, 9, 11, 15, 16 |
| `index.html` | The interactive study guide for all 17 lectures (one file) |
| `references.py` | Bibliography: every paper and model the lectures cite, with title, authors, date and URL |
| `lecture_util.py` | Link helpers (`article_link`, `post_link`, `video_link`, `get_local_url`) |
| `gpu_util.py` | `cuda_if_available()` and `get_max_memory_usage()` for the GPU lectures |
| `facts.py` | Hardware constants: H100 dense FLOP/s and HBM bandwidth |
| `modal_execute.py` | Runs a lecture on a cloud GPU through [Modal](https://modal.com) |
| `images/` | Figures used by the lectures and the study guide |
| `slides/` | Slide renders from the PDF lectures, embedded in `index.html` |
| `var/files/` | Cached copies of cited arXiv papers and remote images |
| `assets/` | The compiled edtrace trace viewer (front-end build output) |
| `package.json`, `package-lock.json` | Front-end tooling (Vite) for the trace viewer |
| `pyproject.toml`, `uv.lock` | Python dependencies, managed with `uv` |
| `.nojekyll` | Tells GitHub Pages to serve the folder as-is |

---

## Executable lectures

These are named `lecture_XX.py` and are written with [edtrace](https://github.com/percyliang/edtrace). Inside a lecture:

- `text("...")` is the narration, in Markdown.
- `image(...)` shows a figure and `link(...)` cites a reference from `references.py`.
- Ordinary Python is live code. Its values are recorded as the lecture runs, and `# @inspect x` displays the value of `x` at that line.

### Setup

```powershell
uv sync
```

```powershell
git clone https://github.com/percyliang/edtrace
```

### Compile a lecture

```powershell
python execute.py -m lecture_01
```

This generates `var/traces/lecture_01.json` and caches any images as appropriate. `execute.py` is the edtrace runner, so it becomes available after the setup step above.

### GPU lectures on Modal

Lectures 6 and 7 time and profile real CUDA and Triton kernels, so they need a GPU. Without one, run them on Modal:

```powershell
modal run modal_execute.py --module lecture_06
```

This builds a CUDA image, runs the lecture remotely, and copies the trace (plus the PTX and profiler outputs) back into `var/`.

### View a trace locally

Start a local server, then open `http://localhost:5173?trace=var/traces/sample.json`:

```powershell
npm run --prefix=edtrace/frontend dev
```

### Deploy to the main website

```powershell
npm run --prefix=edtrace/frontend build
```

Then commit the updated `assets/` and push:

```powershell
git add assets
```

```powershell
git commit -am "<some message>"
```

```powershell
git push
```

---

## Non-executable lectures

These are named `lecture_XX.pdf`: slide decks for lectures 3, 4, 5, 8, 9, 11, 15 and 16. Open them in any PDF viewer. Their content is also written up in `index.html`, with key slides rendered into `slides/` and embedded as numbered figures.

---

## The interactive study guide in detail

`index.html` is written from the lecture sources themselves:

- **For `.py` lectures**, the notes follow the lecture's own sections. They include the real code (with copy buttons), and the values the lecture computes or asserts. Calculators use the same formulas, so their default settings reproduce the lecture's numbers.
- **For `.pdf` lectures**, the notes cover every slide topic. Figures and tables that can't be recreated faithfully are embedded as slide images, not redrawn from guessed data.
- Each lecture has a TL;DR, numbered sections, interactive widgets, numbered figures, key takeaways, a papers-and-links list, and a quiz with worked answers.
- Where a source is ambiguous or inconsistent, the page says so in place, instead of quietly picking one reading. Examples are a slide value that contradicts its own table, or a typo in an exponent. Numbers derived for the guide, rather than taken from a lecture, are labelled as such.

Math is written in TeX and rendered by MathJax. Colours follow a categorical palette that was checked for colour-blind separation, with dedicated dark-mode values.

---

## Rebuilding `index.html`

The page is assembled from per-lecture HTML fragments by a small build script that lives outside this folder, in `../_build/`:

| File in `../_build/` | Role |
|---|---|
| `frag/NN_*.html` | One fragment per lecture, plus `00_intro.html` and `99_cheatsheet.html` |
| `viz.js` | Widget library: line, bar, scatter, matrix, flow, stepper and calc |
| `viz.css`, `base.css` | Page and widget styles |
| `build.py` | Concatenates the fragments into `index.html` and numbers every figure as `Figure <lecture>.<k>` |
| `check_js.py`, `smoke.js` | Checks: JavaScript syntax and duplicate ids; a headless run of every widget |
| `fix_lt.py` | Escapes a raw `<` inside math, which would otherwise break MathJax rendering |

To rebuild after editing a fragment, run this from `../_build/`:

```powershell
python build.py
```

The page from before this rewrite (a Software Testing study page) is kept as `index_software-testing_backup.html`.

---

## Key formulas

| Quantity | Formula |
|---|---|
| Matmul FLOPs, (m×k)·(k×n) | 2·m·k·n |
| Training compute, dense model | C ≈ 6·N·D (N parameters, D tokens) |
| Inference compute per token | ≈ 2·N (active parameters, for MoE) |
| Training memory with AdamW | ≈ 16 bytes per parameter, plus activations |
| Arithmetic intensity | FLOPs ÷ bytes moved; compute-bound when above peak FLOP/s ÷ bandwidth (≈ 295 on an H100) |
| Chinchilla loss | L(N, D) = E + A/N^α + B/D^β; compute-optimal D ≈ 20·N |
| KV cache | 2 × bytes × layers × KV heads × head dim × sequence length × batch |
| Ring all-reduce traffic per GPU | 2·(p − 1)/p · M |
| Pipeline bubble (GPipe) | (p − 1)/(m + p − 1) for p stages and m micro-batches |
| GRPO advantage | (rᵢ − mean(r)) / std(r), over a group of G samples |
| MinHash/LSH candidate probability | 1 − (1 − sʳ)ᵇ |

The cheat sheet in `index.html` has all of these with rendered math and live calculators.

---

## Dependencies

Python (see `pyproject.toml`, managed with `uv`): `edtrace`, `einops`, `mmh3`, `modal`, `tiktoken`. The GPU lectures also use `torch`, `numpy`, `sympy` and Triton; the Modal image installs these.

Front end: `vite`, used to build the edtrace viewer in `assets/`.

The study guide itself has no install step. It needs only a browser, plus a network connection for web fonts and MathJax.
