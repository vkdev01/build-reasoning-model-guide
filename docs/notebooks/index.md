# Notebooks

Eight notebooks, one per chapter. They import the `reasoning_from_scratch`
package rather than re-implementing it, so they work as *drivers* — you change a
parameter, run a cell, and see what happens.

| Notebook | Chapter | Runs on |
| --- | --- | --- |
| [`01_generation_and_kv_cache.ipynb`](01_generation_and_kv_cache.ipynb) | 2 | CPU |
| [`02_verifier_and_baseline.ipynb`](02_verifier_and_baseline.ipynb) | 3 | CPU |
| [`03_inference_scaling.ipynb`](03_inference_scaling.ipynb) | 4 | CPU (slow) / GPU |
| [`04_self_refinement.ipynb`](04_self_refinement.ipynb) | 5 | CPU (slow) / GPU |
| [`05_grpo_training.ipynb`](05_grpo_training.ipynb) | 6 | GPU |
| [`06_reading_training_logs.ipynb`](06_reading_training_logs.ipynb) | 7 | CPU |
| [`07_distillation.ipynb`](07_distillation.ipynb) | 8 | GPU |
| [`08_capstone_eval.ipynb`](08_capstone_eval.ipynb) | Capstone | CPU / GPU |

Click any name to download it.

## Setup

The notebooks need the course package installed. They do **not** vendor it.

```bash
git clone https://github.com/rasbt/reasoning-from-scratch.git
cd reasoning-from-scratch
uv sync
uv run jupyter lab
```

Then copy the `.ipynb` you want into that directory, or point Jupyter at this
repo's `docs/notebooks/` folder and make sure your kernel is the `uv` environment.

## Conventions

Every notebook follows the same shape:

1. **Setup cell** — imports and device selection. Run first, always.
2. **Concept cells** — small, fast, no model needed where possible.
3. **Experiment cell** — the thing to change. Each has a `# TRY:` marker with
   suggestions.
4. **Predict-then-check** — a markdown cell asking you to write down an
   expectation before running. This is the part that makes it stick; do not skip
   it.
5. **Takeaway cell** — the number or fact you should be able to quote afterwards.

## Note on outputs

The notebooks ship with **no stored outputs**. They are templates, not transcripts:
running them is the point. Where a figure or a value is worth knowing in advance,
it is written into the surrounding markdown as "expected" with the reason, so you
can tell a broken environment from a genuine surprise.

## GPU notebooks

`05_grpo_training.ipynb` and `07_distillation.ipynb` are written to start small —
50 steps, 500 examples — so you can validate the loop before committing hours. The
first cell of each prints a wall-clock estimate for a full run on your hardware.
