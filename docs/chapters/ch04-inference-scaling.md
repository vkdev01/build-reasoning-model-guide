# Chapter 4 · Inference-time scaling

!!! abstract "In one line"

    Get more accuracy without touching a single weight, by sampling many answers
    and voting — and learn why the marginal return flattens fast.

## Learning objectives

- Implement temperature scaling and top-p (nucleus) filtering from the logits up.
- Implement self-consistency: sample *n* answers, vote on the final value.
- Read the measured ladder and explain why n=10 is barely better than n=3.
- Decide when inference scaling is worth the compute and when it is not.

## The idea

The weights are frozen. You are only changing *how you draw tokens* and *how you
combine several draws*. Two knobs do the work:

```mermaid
flowchart LR
    P["prompt"] --> S["sample n answers<br/>temperature + top-p"]
    S --> X["extract final answer<br/>from each"]
    X --> V["majority vote"]
    V --> A["final answer"]
```

## Temperature

Temperature divides the logits before the softmax:

```python title="reasoning_from_scratch/ch04.py"
def scale_logits_by_temperature(logits, temperature):
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    return logits / temperature
```

Why it works, in one line: softmax is $$p_i = e^{z_i}/\\sum_j e^{z_j}$$, so
dividing all $$z$$ by $$T < 1$$ magnifies the gaps between them and concentrates
probability on the top token; $$T > 1$$ compresses the gaps and flattens the
distribution.

| Temperature | Effect | Typical use |
| --- | --- | --- |
| 0 (or `argmax`) | Deterministic, most likely token | Baselines, evaluation |
| 0.7 | Mild diversity | Self-refinement |
| 0.8-0.9 | Diverse but coherent | Self-consistency, GRPO rollouts |
| > 1.5 | Incoherent | Never, except to see it break |

## Top-p (nucleus) filtering

Keep the smallest set of tokens whose cumulative probability reaches `p`, then
renormalise. The implementation has one subtlety worth reading twice:

```python title="reasoning_from_scratch/ch04.py"
def top_p_filter(probas, top_p):
    if top_p is None or top_p >= 1.0:
        return probas

    sorted_probas, sorted_idx = torch.sort(probas, dim=1, descending=True)
    cumprobas = torch.cumsum(sorted_probas, dim=1)

    # Keep tokens where prefix cumulative mass (before token) is < top_p
    # Example: [0.5, 0.41, 0.09] with top_p=0.9 should keep the first two tokens
    prefix = cumprobas - sorted_probas
    keep = prefix < top_p
    keep[:, 0] = True                      # always keep at least one

    kept_sorted = torch.where(keep, sorted_probas, torch.zeros_like(sorted_probas))
    filtered = torch.zeros_like(probas).scatter(1, sorted_idx, kept_sorted)

    denom = torch.sum(filtered, dim=1, keepdim=True).clamp_min(1e-12)
    return filtered / denom
```

The comparison is on the cumulative mass **before** each token (`prefix`), not
after. That keeps the token that crosses the threshold, which is what makes
`[0.5, 0.41, 0.09]` with `top_p=0.9` keep two tokens rather than one. A naive
`cumprobas < top_p` would drop the second token and truncate too aggressively.

`keep[:, 0] = True` is the other detail: if `top_p` is tiny, nothing passes the
filter and you would divide by zero. Keeping the single most likely token is the
sane fallback.

## Self-consistency

Sample `n` answers at temperature, extract each final value, take the mode. The
repo's version is careful about ties, which matters more than you would think:

```python title="reasoning_from_scratch/ch04.py (abridged)"
def self_consistency_vote(model, tokenizer, prompt, device,
                          num_samples=10, temperature=0.8, top_p=0.9,
                          max_new_tokens=2048, seed=None, ...):
    short_answers = []
    for i in range(num_samples):
        if seed is not None:
            torch.manual_seed(seed + i + 1)         # reproducible per sample
        answer = generate_text_stream_concat_flex(
            ..., generate_func=generate_text_top_p_stream_cache,
            temperature=temperature, top_p=top_p,
        )
        short_answers.append(extract_final_candidate(answer, fallback="number_then_full"))

    counts = Counter(short_answers)
    mc = counts.most_common()
    top_freq = mc[0][1]
    majority_winners = [s for s, f in mc if f == top_freq]
    final_answer = mc[0][0] if len(majority_winners) == 1 else None

    return {"full_answers": ..., "short_answers": ..., "counts": dict(counts),
            "groups": ..., "majority_winners": majority_winners,
            "final_answer": final_answer}
```

`final_answer` is `None` on a tie, and `majority_winners` tells you who tied.
Returning `None` rather than silently picking the first is the honest choice —
and chapter 5 shows how to break the tie with a scorer.

### Use it

```python title="wk4_self_consistency.py"
from reasoning_from_scratch.ch03 import render_prompt, load_math500_test, grade_answer
from reasoning_from_scratch.ch04 import self_consistency_vote

math500 = load_math500_test()

for item in math500[:5]:
    res = self_consistency_vote(
        model, tokenizer, render_prompt(item["problem"]), device,
        num_samples=5, temperature=0.9, top_p=0.9, max_new_tokens=2048, seed=0,
    )
    print(f"votes={res['counts']}")
    print(f"winner={res['final_answer']!r}  ties={res['majority_winners']}")
    print(f"correct={grade_answer(str(res['final_answer']), item['answer'])}")
    print("---")
```

## The measured ladder

Full 500-question MATH-500, base model, from the repo's own table:

| Method | Accuracy | Runtime | Points per hour |
| --- | ---: | ---: | ---: |
| Greedy (baseline) | 15.6% | 10.1 min | — |
| Chain-of-thought prompt | 40.6% | 84.5 min | **+25.0 for +74 min** |
| Temperature + top-p only | 17.8% | 30.7 min | +2.2 for +20 min |
| Top-p + self-consistency (n=3) | 29.6% | 97.6 min | +11.8 for +67 min |
| Top-p + self-consistency (n=5) | 27.8% | 116.8 min | **worse than n=3** |
| Top-p + self-consistency (n=10) | 31.6% | 300.4 min | +2.0 for +204 min |
| Top-p + CoT | 33.4% | 129.2 min | — |
| Self-consistency (n=3) + CoT | 42.2% | 211.6 min | +8.8 over CoT |
| Self-consistency (n=5) + CoT | 48.0% | 452.9 min | +5.8 |
| Self-consistency (n=10) + CoT | 52.0% | 862.6 min | +4.0 |

Three findings that are worth internalising more than any single number:

1. **The prompt is worth more than the sampling.** CoT alone gives +25 points.
   Sampling alone gives +2. If you can only do one thing, change the prompt.
2. **Self-consistency is not monotonic.** n=5 (27.8%) scores *worse* than n=3
   (29.6%) without CoT. Voting over noisy, correlated samples can hurt. This is
   why you measure instead of assuming.
3. **Returns collapse.** From n=5 to n=10 with CoT: +4 points for +410 minutes.
   That is 100× the baseline's cost for a quarter of the CoT gain.

!!! warning "This is the argument for chapter 6"

    Inference scaling is a linear cost forever — every question, every user, every
    time. Training costs once and then runs at greedy speed. That is the entire
    economic case for GRPO, and this table is the evidence.

## Bonus material

`ch04/02_math500-inference-scaling-scripts/` contains standalone scripts for every
row: `cot_prompting_math500.py`, `self_consistency_math500.py`, a batched variant,
and `run_all_experiments_math500.sh` to reproduce the whole table.

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 --top_p 0.9 \
    --num_samples 3 --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
```

## Exercises

1. **Reproduce row 4 and row 7** on 50 questions. Do you see the same
   non-monotonicity between n=3 and n=5?
2. **Find the correlation.** For one question, look at `full_answers`. Are the
   five samples genuinely different reasoning paths, or the same path with
   cosmetic variation? Correlated samples explain why voting saturates.
3. **Break the tie.** When `final_answer is None`, the chapter returns nothing.
   Write a tie-breaker using the average log-probability of each candidate —
   then check your answer against chapter 5's `self_consistency_scorer_math500.py`.
4. **Cheap temperature sweep.** At what temperature does accuracy collapse? Is
   the curve sharp or gentle?

## Checkpoint

- You have run at least two rows of the ladder on your own hardware.
- You can explain why `prefix = cumprobas - sorted_probas` is not `cumprobas`.
- You can state the cost-per-point argument for training over sampling.

**Notebook:** [`03_inference_scaling.ipynb`](../notebooks/03_inference_scaling.ipynb) ·
**Source:** `reasoning_from_scratch/ch04.py` ·
**Video:** 4 of the course.
