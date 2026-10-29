# Capstone: point the pipeline at your own domain

The course uses maths because maths has an unambiguous verifier. Your job — if you
want this to be more than a tutorial — is to swap the verifier and keep the
pipeline.

## What stays the same

Everything except two functions:

| Component | Reusable? |
| --- | --- |
| Model, tokenizer, generation | Yes, unchanged |
| GRPO loop, advantages, clipping | Yes, unchanged |
| Distillation data format and training loop | Yes, unchanged |
| Prompt template (`render_prompt`) | Rewrite |
| Verifier (`grade_answer` / `reward_rlvr`) | Rewrite |

That is the point worth taking away: **the reasoning machinery is domain-agnostic.
Only the reward function is domain-specific.**

## Choosing a domain

You need three things: a task an LLM is bad at without thinking, a checkable
answer, and a source of problems. Good candidates:

| Domain | Verifier | Problems from |
| --- | --- | --- |
| Text-to-SQL | Execute both queries, compare result sets | Your own schema + question log |
| Unit / currency conversion | Deterministic arithmetic with tolerances | Generated combinations |
| Regex or grammar tasks | Run against a test corpus | Existing test suites |
| Scheduling / constraint puzzles | Constraint checker | Generated instances |
| Code golf / small functions | Unit tests in a sandbox | Exercism-style corpora |
| Data-cleaning transforms | Assert on output dataframe | Your own pipelines |

Avoid anything where "correct" needs a human. The moment you need an LLM judge,
you have lost the property that makes this cheap.

## The five steps

### 1. Write the verifier first

Before collecting any data, write `grade_answer` for your domain and test it on
twenty hand-checked cases — including five you expect to fail. If you cannot write
a verifier in an afternoon, pick a different domain.

```python title="my_verifier.py — the shape to copy"
def grade_answer(pred_text, gt_text):
    """Return True only if pred is provably equivalent to gt."""
    raise NotImplementedError
```

### 2. Build a small eval set

50-200 items with verified ground truth. This is your `math500_test.json`
equivalent, and it must be held out from all training data. Record the same fields
so the existing tooling works: `problem`, `answer`, plus whatever metadata helps
you slice results.

### 3. Measure the baseline

Adapt `evaluate_math500.py` to your file. You need that number before you change
anything, or you will not know whether training helped.

!!! tip "Sanity check"

    If your base model scores 0%, your prompt or your verifier is broken — not
    your model. Fix that before spending GPU hours.

### 4. Choose your lever

| If you have | Do |
| --- | --- |
| No GPU, decent teacher access | Distillation: generate traces, verify, train |
| GPU, cheap verifier | GRPO from the base model |
| Neither | Inference scaling only: sample and vote. Still a valid result |

### 5. Write it up

A short report with: your baseline, your method, your result, the failure modes you
hit, and one chart. The failure modes are the interesting part — anyone can report
a number that went up.

## What a good capstone looks like

- A repo where `python run_eval.py` reproduces your headline number.
- A verifier with tests.
- A training log with the metric columns from chapter 7.
- Two or three sentences explaining what your model learned *to do differently*,
  not just that accuracy improved.

## Extension: the chat interface

Appendix G in the original repo wires the checkpoint into Chainlit. It needs
Python 3.13 (Chainlit does not support 3.14).

```bash
cd chG/01_main-chapter-code
uv run chainlit run app.py
```

Point it at your own checkpoint and you have something to demo.
