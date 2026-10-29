# Build a Reasoning Model From Scratch — the guide

An exhaustive, hands-on tutorial site for implementing reasoning models in
PyTorch: evaluation harnesses, inference-time scaling, GRPO reinforcement
learning, and distillation — built around Sebastian Raschka's free
eight-video course.

**Read it at:** https://vkdev01.github.io/build-reasoning-model-guide/

## What is in here

| Path | What it is |
| --- | --- |
| `docs/` | The tutorial, one Markdown page per chapter, with annotated code, worked examples, data tables and exercises |
| `docs/notebooks/*.ipynb` | Eight runnable notebooks, one per chapter, that drive the `reasoning_from_scratch` package |
| `mkdocs.yml` | MkDocs Material configuration |
| `requirements.txt` | Site build dependencies |

## Run the notebooks

```bash
git clone https://github.com/rasbt/reasoning-from-scratch.git
cd reasoning-from-scratch
uv sync                      # installs torch 2.10, tokenizers, sympy, the package
uv run jupyter lab
```

Then open the notebook you want from `docs/notebooks/` in this repo. The
notebooks import from the installed `reasoning_from_scratch` package, so they
must be run with that environment active.

## Build the site locally

```bash
pip install -r requirements.txt
mkdocs serve          # http://127.0.0.1:8000
mkdocs build --strict # what CI runs
```

## Attribution

Tutorial text and notebooks: MIT, © 2026 VK.

All taught code and all benchmark figures: Sebastian Raschka's
[reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch),
Apache-2.0. Every page carries its own source note. If you find this guide
useful, buy the book and star the original repo.
