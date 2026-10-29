# Sources, attribution and licence

## Primary sources

| Source | What it is |
| --- | --- |
| [Building Reasoning Models from the Ground Up](https://magazine.sebastianraschka.com/p/building-reasoning-models) | The free course: 8 videos, ~11 hours, published Oct 2026 |
| [rasbt/reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch) | Code repository: package, notebooks, bonus scripts, benchmark READMEs, training logs |
| [Build a Reasoning Model (From Scratch)](https://sebastianraschka.com/reasoning-from-scratch/) | The companion book (Manning) |
| [rasbt/qwen3-from-scratch](https://huggingface.co/rasbt/qwen3-from-scratch) | Base and reasoning Qwen3-0.6B weights |
| [rasbt/math_distill](https://huggingface.co/datasets/rasbt/math_distill) | Distillation data from DeepSeek-R1 and Qwen3-235B |
| [Shivani Virdi's LinkedIn post](https://www.linkedin.com/in/shivani-virdi/) | The post that surfaced the course and prompted this guide |

## Licence

**This guide** — the tutorial text, page structure, exercises, and the eight
notebooks — is original work, MIT licensed, © 2026 VK.

**The taught code** — every function signature, implementation snippet and
benchmark figure — comes from `rasbt/reasoning-from-scratch`, Apache License 2.0,
© Sebastian Raschka. Snippets are reproduced or abridged for teaching, with
attribution on every page and a source note in each chapter's footer.

If this guide is useful to you, the thing to do is
[star the original repository](https://github.com/rasbt/reasoning-from-scratch) and
[buy the book](https://sebastianraschka.com/reasoning-from-scratch/).

## How the figures in this guide were checked

| Claim type | Method |
| --- | --- |
| Verifier behaviour (ch 3 test table) | Functions extracted from `ch03.py` and executed; every result observed |
| Dataset sizes and schemas | Files downloaded and inspected: 500 / 12,000 / 12,000 rows |
| Accuracy and runtime tables | Read from the repo's chapter READMEs |
| GRPO and distillation curves | Parsed from the CSVs in `ch07/02_logs/` and `ch08/03_logs/` |
| Teacher acceptance rate (90.6%) | Printed by `ch08_main.ipynb` when it runs the verifier over the distillation set |
| KV cache sizes, entropy figures | Computed from `QWEN_CONFIG_06_B` and ln(vocab_size) — see [Results](results.md) |

**Not verified by execution:** the GPU training runs and the full 500-question
evaluations. There is no GPU in the environment where this guide was written, and
those runs take hours. Training dynamics are read from the shipped logs, not
reproduced.

## A correction to the LinkedIn post

The post says "the examples run on a laptop". That is true for chapters 2-5 with
reduced dataset sizes, and the author says the same. It is not true for chapters
6-8: the author is explicit that a GPU makes the later experiments "much more
practical", and the GRPO logs show roughly 47 tokens/second, which is a GPU
number.
