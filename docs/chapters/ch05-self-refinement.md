# Chapter 5 · Self-refinement, scoring, and best-of-N

!!! abstract "In one line"

    Give the model a chance to critique and revise its own answer — and learn why
    that helps a strong model and hurts a weak one.

## Learning objectives

- Score candidate answers three ways: heuristic, average log-probability, and
  log-probability of the extracted answer only.
- Build a critique-then-revise loop and control it with a scorer.
- Distinguish self-refinement from best-of-N from self-consistency.
- Explain the base-vs-reasoning asymmetry in the measured results.

## Three ways to spend more compute

These are easy to confuse, so pin them down first:

| Method | Generates | Selects by | Chapter |
| --- | --- | --- | --- |
| Self-consistency | n independent answers | majority vote on final value | 4 |
| Best-of-N | n independent answers | a **scorer** ranks them | 5 |
| Self-refinement | 1 answer, then k revisions | a scorer decides whether to keep the revision | 5 |

Self-consistency needs no scorer, which is why it comes first. The moment you
want to rank candidates rather than count them, you need a scoring function.

## Scorer 1 — heuristic

No model call, just rules about the shape of the answer:

```python title="reasoning_from_scratch/ch05.py"
def heuristic_score(answer, prompt=None, brevity_bonus=500.0,
                    boxed_bonus=2.0, extract_bonus=1.0, fulltext_bonus=0.0):
    score = 0.0

    cand = extract_final_candidate(answer, fallback=None)
    if cand:
        score += boxed_bonus                      # +2.0 for a proper \boxed{}
    else:
        cand = extract_final_candidate(answer, fallback="number_only")
        if cand:
            score += extract_bonus                # +1.0 for a bare number
        else:
            cand = extract_final_candidate(answer, fallback="number_then_full")
            if cand:
                score += fulltext_bonus           # +0.0 - no credit

    score += 1.5 * math.exp(-len(answer) / brevity_bonus)   # brevity, decays with length
    return score
```

Read the defaults as a specification: **a boxed answer is worth twice a bare
number, and rambling is penalised** by a term that decays exponentially with
length. That is a crude but effective proxy for "did it follow instructions".

It costs nothing and it is surprisingly hard to beat. In the repo's table it
matches or beats average log-probability in some configurations.

## Scorer 2 and 3 — log-probability

Average log-probability asks "how confident was the model in what it said":

```python title="reasoning_from_scratch/ch05.py"
def avg_logprob_answer(model, tokenizer, prompt, answer, device="cpu"):
    prompt_ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(answer)
    full_ids = torch.tensor(prompt_ids + answer_ids, device=device)

    logits = model(full_ids.unsqueeze(0)).squeeze(0)
    logprobs = torch.log_softmax(logits, dim=-1)
    # ... average over the answer-token positions only ...
```

The important subtlety: you score **only the answer tokens**, not the prompt.
Otherwise a model that is merely confident about the question looks confident
about its answer.

The third variant, `logprob_extract`, scores only the extracted final answer
rather than the whole response. Intuition: a model can be fluent about its
reasoning and still be unsure of the number it boxes. Scoring just the box is
closer to what you care about.

## The refinement loop

Two prompts, and they are worth reading as prompt engineering rather than code:

```python title="reasoning_from_scratch/ch05.py"
def make_critique_prompt(raw_prompt, draft):
    return (
        "You are a meticulous reviewer. Identify logical errors, missing "
        "steps, or arithmetic mistakes. If the answer seems correct, "
        "say so briefly. Then propose a concise plan to fix issues.\n\n"
        f"Question:\n{raw_prompt}\n\n"
        f"Draft answer:\n{draft}\n\n"
        "Write a short critique and bullet-point fix plan "
        "(under ~120 words).\n"
        "Critique:"
    )

def make_refine_prompt(raw_prompt, draft, critique):
    return (
        "Revise the answer using the critique. Keep it concise and "
        "end with a final boxed result: \\boxed{ANSWER}\n\n"
        f"Question:\n{raw_prompt}\n\n"
        f"Previous answer:\n{draft}\n\n"
        f"Critique:\n{critique}\n\n"
        "Revised answer:"
    )
```

Note the word limit on the critique (~120 words) and the repeated instruction to
end with a boxed result. Both exist because a 0.6B model will happily spend 800
tokens critiquing and then forget to answer.

The loop itself:

```python title="self_refinement_loop (signature)"
self_refinement_loop(
    model, tokenizer, raw_prompt, device,
    iterations=2,              # how many critique->revise rounds
    max_response_tokens=2048,
    max_critique_tokens=256,
    score_fn=None,             # None = always accept the revision
    prompt_renderer=render_prompt,
    prompt_suffix="",
    verbose=False,
)
```

With `score_fn=None` you always keep the revision. With a scorer, you keep
whichever of draft and revision scores higher — which turns refinement from a
gamble into a monotone improvement *by construction* (you can always fall back
to the draft).

## Worked example

```python title="wk5_refinement.py"
from reasoning_from_scratch.ch05 import self_refinement_loop, heuristic_score

raw_prompt = "Half the value of $3x-9$ is $x+37$. What is the value of $x$?"

for score_fn, label in [(None, "always accept"), (heuristic_score, "scored")]:
    out = self_refinement_loop(
        model, tokenizer, raw_prompt, device,
        iterations=1, score_fn=score_fn, verbose=True,
    )
    print(f"\n=== {label} ===")
    print(out)
```

## The measured results

Base model, full MATH-500:

| Method | Scorer | Iterations | Accuracy | Runtime |
| --- | --- | ---: | ---: | ---: |
| Baseline (ch3) | — | — | 15.2% | 10.1 min |
| Self-refinement | none | 1 | **25.0%** | 84.8 min |
| Self-refinement | none | 2 | 22.0% | 165.4 min |
| Self-refinement | heuristic | 1 | 21.6% | 84.7 min |
| Self-refinement | logprob | 1 | 21.4% | 85.3 min |
| Self-refinement | logprob-extract | 1 | 20.4% | 85.0 min |

Reasoning model:

| Method | Scorer | Iterations | Accuracy | Runtime |
| --- | --- | ---: | ---: | ---: |
| Baseline (ch3) | — | — | 48.2% | 182.1 min |
| Self-refinement | none | 1 | **56.6%** | 498.8 min |
| Self-refinement | heuristic | 1 | **57.8%** | 498.6 min |
| Self-refinement | logprob | 1 | 48.4% | 499.7 min |

### Reading these honestly

**On the base model, refinement loses to a plain CoT prompt** (25.0% vs 40.6%).
Two iterations are worse than one (22.0% vs 25.0%). A weak model critiquing a
weak draft mostly talks itself out of correct answers — it has no reliable
signal about which of its two attempts is better, so each revision is close to a
coin flip and you pay 84 minutes for the privilege.

**On the reasoning model, refinement helps** (48.2% → 56.6%, and 57.8% with the
heuristic scorer). A model that can reason can also *review* reasoning. This is
the most useful asymmetry in the chapter: self-improvement loops are a
capability multiplier, not a capability substitute.

**Log-probability scoring underperforms the free heuristic** on the reasoning
model (48.4% vs 57.8%). A model's stated confidence is not well calibrated on
long reasoning traces — it can be confidently wrong. The dumb length-and-format
heuristic wins because it is measuring something real: instruction-following.

!!! tip "Tie-breaking self-consistency"

    `ch05/02_math500-more-inference-scaling-scripts/self_consistency_scorer_math500.py`
    uses these scorers to break the ties that chapter 4 returns as `None`. The
    result: self-consistency n=3 + CoT goes from 43.2% (majority vote) to 44.8%
    (average logprob tie-break). A small gain, but it costs no extra generation.

## Best-of-N

Same idea as self-consistency but select by score instead of count:

| Method | Accuracy | Runtime |
| --- | ---: | ---: |
| CoT baseline | 33.4% | 129.2 min |
| Best-of-N (n=3) + heuristic | 40.6% | 327.7 min |
| Best-of-N (n=3) + avg logprob | 43.2% | 330.2 min |

Best-of-N beats majority voting at the same n, because a scorer uses information
the vote throws away: *how* the answer was written, not just what it was.

## Exercises

1. **Make the heuristic yours.** Change `brevity_bonus` to 100 and 2000. Which
   direction does accuracy move? Why does the default 500 make sense for answers
   of a few hundred characters?
2. **Force the failure.** Run refinement with `score_fn=None` for 3 iterations on
   a question the base model gets right first time. Does it survive?
3. **Scorer ablation.** Build a scorer that only checks for `\boxed{}` and
   nothing else. Compare against the full heuristic.
4. **Predict before you run.** Before measuring, write down whether refinement
   should help or hurt the base model, and why. Then check.

## Checkpoint

- You can explain why `score_fn=None` is a gamble and `score_fn=heuristic_score`
  is monotone by construction.
- You can say why log-probability scoring underperforms a length heuristic on
  reasoning traces.
- You have run one refinement loop and looked at the critique text.

**Notebook:** [`04_self_refinement.ipynb`](../notebooks/04_self_refinement.ipynb) ·
**Source:** `reasoning_from_scratch/ch05.py` ·
**Video:** 5 of the course.
