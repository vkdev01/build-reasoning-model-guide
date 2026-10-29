# Results & benchmarks

All figures are the repository's own, measured on the full 500-question MATH-500
test set. Unless noted, runtimes are on an NVIDIA DGX Spark with sequential
generation.

## The full ladder

| # | Method | Model | Accuracy | Runtime |
| ---: | --- | --- | ---: | ---: |
| 1 | Baseline, greedy decoding | base | 15.6% | 10.1 min |
| 2 | Baseline, greedy decoding | reasoning | 48.2% | 182.1 min |
| 3 | Chain-of-thought prompting | base | 40.6% | 84.5 min |
| 4 | Temperature + top-p only | base | 17.8% | 30.7 min |
| 5 | Top-p + self-consistency (n=3) | base | 29.6% | 97.6 min |
| 6 | Top-p + self-consistency (n=5) | base | 27.8% | 116.8 min |
| 7 | Top-p + self-consistency (n=10) | base | 31.6% | 300.4 min |
| 8 | Top-p + CoT | base | 33.4% | 129.2 min |
| 9 | Self-consistency (n=3) + top-p + CoT | base | 42.2% | 211.6 min |
| 10 | Self-consistency (n=5) + top-p + CoT | base | 48.0% | 452.9 min |
| 11 | Self-consistency (n=10) + top-p + CoT | base | 52.0% | 862.6 min |
| 12 | Self-consistency (n=3) + top-p + CoT | reasoning | 55.2% | 544.4 min |
| 13 | Self-refinement, no scorer, 1 iter | base | 25.0% | 84.8 min |
| 14 | Self-refinement, no scorer, 2 iters | base | 22.0% | 165.4 min |
| 15 | Self-refinement, heuristic, 1 iter | base | 21.6% | 84.7 min |
| 16 | Self-refinement, no scorer, 1 iter | reasoning | 56.6% | 498.8 min |
| 17 | Self-refinement, heuristic, 1 iter | reasoning | 57.8% | 498.6 min |
| 18 | Best-of-N (n=3) + heuristic | base | 40.6% | 327.7 min |
| 19 | Best-of-N (n=3) + avg logprob | base | 43.2% | 330.2 min |
| 20 | Self-consistency (n=3) + avg logprob tie-break | base | 44.8% | 327.7 min |

!!! note "15.6% vs 15.2%"

    Chapter 3's README states the base model accuracy as 15.6% (78/500). The
    chapter 4 and 5 tables list the same baseline as 15.2%. Same run family,
    version drift between runs. Use 15.6% as the anchor.

## Runtime by hardware

`evaluate_math500.py`, 500 questions:

| Device | Base, sequential | Reasoning, sequential | Base, batched (bs=128) | Reasoning, batched |
| --- | ---: | ---: | ---: | ---: |
| Mac Mini M4, CPU | 43.6 min | — | 167.2 min | — |
| Mac Mini M4, GPU | 37.5 min | — | error | — |
| DGX Spark | 10.0 min | 182.2 min | 16.3 min | 119.3 min |
| H100 | 13.3 min | 185.4 min | 3.3 min | 14.6 min |

The author reported the M4 running "too hot" for the reasoning-model evaluations,
hence the dashes.

## GRPO training progression

From the CSVs in `ch07/02_logs/`:

| Variant | Step 1 | Peak | Step 500 |
| --- | ---: | ---: | ---: |
| Chapter 6, original, no KL | 15.6% | 46.2% @50 | 33.0% |
| 7_3, + tracking | 15.6% | 46.2% @50 | 33.0% |
| 7_4, + clipped ratio | 15.6% | 45.6% @100 | 41.6% |
| 7_5, + KL term | 15.6% | 40.8% @50 | 0.0% |
| 7_6, + format reward | 50.8% | 56.2% @50 | 35.0% |

7_3 is byte-identical to chapter 6 — adding instrumentation changed nothing. 7_6
starts from a different baseline, so its curve is not directly comparable.

## Distillation

DeepSeek-R1 teacher, 3 epochs, ~20,000 steps:

| Epoch | Step | Train loss | Val loss |
| ---: | ---: | ---: | ---: |
| 1 | 50 | 0.766 | 0.699 |
| 1 | 100 | 0.746 | 0.668 |
| 3 | 20,000 | 0.436 | 0.534 |
| 3 | 20,046 | 0.556 | 0.534 |

Teacher answer acceptance: **90.6% (10,871 / 12,000)** on the training partition.

## Derived figures used in this guide

These are computed here, from the model config and the logs, not taken from the
repo:

| Figure | Value | How |
| --- | --- | --- |
| KV cache per token | 112 KiB | 8 KV groups × 128 dims × 2 (K,V) × 28 layers × 2 bytes |
| KV cache at 1,024 tokens | 112 MiB | above × 1024 |
| KV cache at 40,960 (full context) | 4,480 MiB | above × 40,960 |
| Uniform-distribution entropy | 11.93 | ln(151,936) — matches the collapsed KL run's `entropy_avg` of 11.93 |
| Reasoning eval cost vs base | ~18× | 182.2 / 10.0 min |
| reward/length correlation, ch6 run | -0.74 | computed from `ch06_rlvr_grpo_original_no_kl_metrics.csv` |
| uniform entropy, ln(vocab) | 11.93 | matches the KL run's final `entropy_avg` of 11.928 |
