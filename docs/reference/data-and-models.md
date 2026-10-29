# Data & models

Every URL below was resolved and checked while writing this guide.

## Model weights

Two variants, both loaded by `download_qwen3_small`. Each weight file is
**1.50 GB**.

| Variant | Weight file | Tokenizer | MATH-500 |
| --- | --- | --- | ---: |
| base | `qwen3-0.6B-base.pth` | `tokenizer-base.json` | 15.6% |
| reasoning | `qwen3-0.6B-reasoning.pth` | `tokenizer-reasoning.json` | 48.2% |

```text
https://huggingface.co/rasbt/qwen3-from-scratch
```

```python
from reasoning_from_scratch.qwen3 import download_qwen3_small
download_qwen3_small(kind="base", out_dir="qwen3")       # or kind="reasoning"
```

A Backblaze backup URL is baked into the downloader, so a Hugging Face outage
does not stop you.

## Training checkpoints

Plain PyTorch `state_dict` files for the `reasoning_from_scratch` model class —
**not** Transformers checkpoints.

| Repo | Contents |
| --- | --- |
| `rasbt/qwen3-from-scratch-grpo-checkpoints` | `7_3_plus_tracking`, `7_4_plus_clip_ratio`, `7_5_plus_kl`, `7_6_plus_format_reward` |
| `rasbt/qwen3-from-scratch-distill-checkpoints` | `ch08_distill_deepseek_r1`, `ch08_distill_qwen3_235b_a22b` |

```python
from reasoning_from_scratch.qwen3 import (
    download_qwen3_grpo_checkpoints, download_qwen3_distill_checkpoints,
)

download_qwen3_grpo_checkpoints(grpo_type="clip_ratio", step="00050", out_dir="qwen3")
download_qwen3_distill_checkpoints(distill_type="deepseek_r1", step="06682", out_dir="qwen3")
```

## Datasets

### MATH-500 test set — 500 items

```text
https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/main/ch03/01_main-chapter-code/math500_test.json
```

| Field | Type | Example |
| --- | --- | --- |
| `problem` | str | "Convert the point $(0,3)$ ... to polar coordinates." |
| `solution` | str | Full worked solution with `\boxed{}` |
| `answer` | str | `\\left( 3, \\frac{\\pi}{2} \\right)` |
| `subject` | str | `Precalculus` |
| `level` | int | `2` (1-5) |
| `unique_id` | str | `test/precalculus/807.json` |

### Training set — 12,000 items

MATH full minus MATH-500, so there is no contamination.

```text
https://raw.githubusercontent.com/rasbt/math_full_minus_math500/refs/heads/main/math_full_minus_math500.json
```

Fields: `problem`, `level`, `type`, `solution`, `answer`, `unique_id`.

### Distillation data — 4 partitions × 12,000 items

```text
https://huggingface.co/datasets/rasbt/math_distill/resolve/main/data/deepseek-r1-math-train.json
https://huggingface.co/datasets/rasbt/math_distill/resolve/main/data/deepseek-r1-math500.json
https://huggingface.co/datasets/rasbt/math_distill/resolve/main/data/qwen3-235b-a22b-math-train.json
https://huggingface.co/datasets/rasbt/math_distill/resolve/main/data/qwen3-235b-a22b-math500.json
```

| Field | Content |
| --- | --- |
| `problem` | The question |
| `gtruth_answer` | Verified ground truth |
| `message_thinking` | Teacher's reasoning trace |
| `message_content` | Teacher's final answer |

```python
from reasoning_from_scratch.ch08 import load_distill_data
data = load_distill_data(partition="deepseek-r1-math-train")
```

## Loaders

| Function | Module | Returns |
| --- | --- | --- |
| `load_math500_test()` | `ch03` | 500 dicts |
| `load_math_train()` | `ch06` | 12,000 dicts |
| `load_distill_data(partition=...)` | `ch08` | 12,000 dicts |
| `download_qwen3_small(kind=...)` | `qwen3` | writes files to disk |
| `load_model_and_tokenizer(which_model=...)` | `ch03` | `(model, tokenizer)` |
| `load_reasoning_tokenizer()` | `ch08` | tokenizer only |

All three data loaders cache a local copy on first call and accept a
`local_path` argument if you want to control where it lands.

## Model configuration

```python
QWEN_CONFIG_06_B = {
    "vocab_size": 151_936, "context_length": 40_960, "emb_dim": 1024,
    "n_heads": 16, "n_layers": 28, "hidden_dim": 3072, "head_dim": 128,
    "qk_norm": True, "n_kv_groups": 8, "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}
```

## Environment

| Requirement | Version |
| --- | --- |
| Python | 3.10-3.14 (3.13 for the Chainlit UI) |
| `torch` | `>=2.10,<3` (lockfile pins 2.10.0) |
| `tokenizers` | `>=0.21.2` |
| `sympy` | `>=1.14` |
| `reasoning-from-scratch` | `>=0.1.2` (package version 0.2.0) |
