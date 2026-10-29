# Setup

## Learning objectives

- Choose hardware appropriate to the chapters you intend to run.
- Install the course environment reproducibly.
- Verify it with a smoke test that exercises weights, tokenizer, and KV cache.

## Hardware

| Setup | Chapters 2-5 | Chapters 6-8 | Verdict |
| --- | --- | --- | --- |
| Laptop, CPU only | Workable, slow — full 500-question eval is 43.6 min on an M4 | Impractical | Do weeks 1-3 locally, rent for 4-6 |
| Apple Silicon (MPS) | Good — 37.5 min for 500 questions on an M4 GPU | Possible but hot; the author reported the M4 getting "too hot" on long evals | Fine for the first half |
| One rented GPU (A100 / H100 / L40S) | Fast — H100: 3.3 min batched for 500 questions | Yes | The sane choice for chapters 6-8 |
| Free Colab T4 | Yes — the weight file is 1.5 GB, fits in 16 GB with room for activations | Short runs only; session limits will cut a 500-step run | Good for a first GRPO taste |

The repo's own recommendation for cloud is [Lightning AI
Studio](https://lightning.ai/) — a persistent environment you can switch between
CPU and GPU without reinstalling — or Google Colab. Any provider where the
PyTorch install survives between sessions is worth more than one where it does
not.

## Install

The `uv` route is what the author uses, and it pins PyTorch 2.10.0 through a
committed lockfile. That pin is the difference between a working environment and
an afternoon of debugging CUDA.

```bash
git clone https://github.com/rasbt/reasoning-from-scratch.git
cd reasoning-from-scratch

uv python pin 3.13      # 3.10-3.14 all work; 3.13 only needed for the Chainlit chat UI
uv sync                 # installs torch 2.10.0 from uv.lock
uv run jupyter lab
```

If you would rather not use `uv`:

```bash
pip install -r https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/refs/heads/main/requirements.txt
```

That installs four things that matter:

| Package | Version | Why |
| --- | --- | --- |
| `torch` | `>=2.10,<3` | Everything |
| `tokenizers` | `>=0.21.2` | Qwen3 tokenizer, loaded from JSON |
| `sympy` | `>=1.14` | The verifier's symbolic equality check |
| `reasoning-from-scratch` | `>=0.1.2` | The package this guide imports |

!!! tip "Pin the Python version deliberately"

    The repo has no `.python-version` file. `uv python pin 3.13` creates one. If
    you skip it, `uv` picks whatever it likes and your environment drifts from
    the lockfile's expectations. Appendix G (the chat interface) genuinely needs
    3.13 because Chainlit does not support 3.14.

## Smoke test

This is chapter 2's core end to end: download weights, load them, generate with
a KV cache. If it prints a coherent paragraph, your environment is correct.

```python title="smoke_test.py"
from pathlib import Path
import torch
from reasoning_from_scratch.qwen3 import (
    Qwen3Model, QWEN_CONFIG_06_B, Qwen3Tokenizer, download_qwen3_small,
)
from reasoning_from_scratch.ch02 import get_device, generate_text_basic_stream_cache

device = get_device()          # cuda / mps / xpu / cpu; sets TF32 on CUDA
download_qwen3_small(kind="base", out_dir="qwen3")   # 1.5 GB, cached after first run

tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=Path("qwen3") / "tokenizer-base.json"
)
model = Qwen3Model(QWEN_CONFIG_06_B)
model.load_state_dict(torch.load(Path("qwen3") / "qwen3-0.6B-base.pth"))
model.to(device)

prompt = "Explain what a reasoning model is in two sentences."
ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)

for tok in generate_text_basic_stream_cache(
    model, ids, max_new_tokens=120, eos_token_id=tokenizer.eos_token_id
):
    print(tokenizer.decode([tok.item()]), end="", flush=True)
```

**Expected output:** a short, fluent, unremarkable paragraph. That is correct.
This is the *base* model, before any reasoning training: it does not think before
it answers, and it will often commit to a single unhedged guess.

If you get tokens like `<|im_start|>` in the output, you loaded the wrong
tokenizer — see [troubleshooting](../reference/troubleshooting.md).

## Directory layout, and which part you import

You will spend most of your time in the importable package, not the notebooks:

```text
reasoning-from-scratch/
├── reasoning_from_scratch/          <-- import this
│   ├── qwen3.py                     model, tokenizer, weight download, KVCache
│   ├── qwen3_optimized.py           GPU-optimised model variant
│   ├── ch02.py                      generation: greedy, streaming, KV cache
│   ├── ch03.py                      verifier + MATH-500 evaluation
│   ├── ch04.py                      temperature, top-p, self-consistency
│   ├── ch05.py                      scorers, critique/refine, best-of-N
│   ├── ch06.py                      GRPO: reward, advantages, training loop
│   ├── ch07.py                      log parsing, entropy, format reward
│   ├── ch08.py                      distillation data, training loop
│   ├── appendix_c.py                loading HF weights into the scratch model
│   └── utils.py                     downloads with backup URLs
├── ch02/ ... ch08/                  notebooks, bonus scripts, benchmark READMEs
└── chG/                             Chainlit chat interface
```

Each `chNN/` folder holds a `01_main-chapter-code/` notebook that rebuilds the
package functions cell by cell. Reading the notebook *and* the module is the
fastest way in: the notebook shows the derivation, the module shows the final
shape.

## If you only have a laptop

**Do locally:** chapters 2, 3, 4, 5. Chapter 2's generation demos are instant.
Chapter 3's verifier work is pure string and SymPy work — no model needed. For
evaluation use `--dataset_size 50` and accept the noise.

**Do on rented GPU:** chapters 6, 7, 8. A single-step `compute_grpo_loss` demo
will run on a CPU with low `max_new_tokens`, but a 500-step run needs a GPU.

**The shortcut:** every checkpoint from chapters 7 and 8 is published on Hugging
Face. If you run out of compute, download them and evaluate — you get the
results without the training. Do at least one training run yourself, though. The
point is the loop, not the file.

**Minimum viable version, 10 hours, one laptop:** chapter 2 smoke test, chapter
3 verifier plus a 50-question baseline, chapter 4 self-consistency on 20
questions, and chapter 6's `compute_grpo_loss` read until you can explain the
advantage normalisation line by line. That is the conceptual core, and it needs
no GPU.

## Checkpoint

- `smoke_test.py` prints a paragraph.
- Inside the environment you will use for the notebooks, `import reasoning_from_scratch` succeeds and `reasoning_from_scratch.__version__` prints a version string.
- You know which of your machines will run chapters 6-8, and roughly what it will cost.

Next: [Chapter 2 — Generation and KV caching](../chapters/ch02-generation.md).
