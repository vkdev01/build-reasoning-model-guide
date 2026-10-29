# Troubleshooting

Every row below is a failure mode documented in the repo or forced by the code.

## Accuracy and evaluation

??? question "Accuracy is exactly 0.0%"

    The model is not emitting `\boxed{}`, so `extract_final_candidate` returns
    nothing and every answer is graded wrong — or, in training, every reward is 0.

    Print a raw generation and look at it. Fix the prompt (`render_prompt`) before
    you touch the verifier. A model that answers correctly but unboxed scores 0.0
    by design.

??? question "Accuracy is suspiciously low but outputs look right"

    Likely verifier false negatives. Take 50 outputs marked wrong and eyeball them.
    Known cause: nested tuples normalising asymmetrically, e.g.
    `grade_answer(r"(3, \frac{\pi}{2})", "(3, pi/2)")` returns `False` because the
    left side normalises to `3, (\pi)/(2)`, loses its leading bracket, and
    `split_into_parts` returns one part instead of two.

    See `ch03/03_advanced-parser` for a better LaTeX parser.

??? question "Accuracy is 100%"

    You are probably evaluating on the training set, or your verifier is returning
    True for everything. Check `grade_answer("38", "83")` returns `False`.

## Generation and devices

??? question "Batched evaluation crashes on a Mac"

    Some PyTorch ops used in efficient batched inference are not implemented for
    MPS. Either:

    ```bash
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run evaluate_math500_batched.py --batch_size 128
    ```

    or pass `--disable_efficient_mode`.

??? question "Output contains `<|im_start|>` and `<|im_end|>`"

    Wrong tokenizer. The base model uses `tokenizer-base.json`; the reasoning model
    uses `tokenizer-reasoning.json` with `apply_chat_template=True` and
    `add_thinking=True`. Mixing them produces exactly this.

??? question "`torch.compile` fails on Windows"

    Known platform issue. Use `ch02/04_torch-compile-windows`, or skip compilation
    and accept the speed hit.

??? question "Generation is very slow"

    Check three things: is TF32 enabled (automatic on CUDA via `get_device()`), did
    you enable `torch.compile`, and are you evaluating 500 questions when 50 would
    do while iterating?

## Training

??? question "`reward_avg` is pinned at 0.0"

    Either the model never produces `\boxed{}` (see above), or every problem is too
    hard for the current policy. Try a subset of easier problems (`level` 1-2 in the
    training data) to confirm the loop can learn at all.

??? question "`reward_avg` is pinned at 1.0"

    Every rollout succeeds, so every advantage is 0 and nothing is learned. Your
    problems are too easy, or your `num_rollouts` is too small to show variance.

??? question "Loss becomes a huge negative number"

    Normal in this loop — the loss is `-(advantage × logprob).sum()`, so a
    high-reward long answer can produce a large negative value. What matters is the
    trend and whether accuracy moves. In the repo's log, step 500 shows
    `loss = -916.8` with `reward_avg = 0.125`.

??? question "Accuracy peaks early then decays"

    Policy drift into verbose output. `avg_response_len` climbs from 6 to 762 tokens
    over 500 steps in the reference run, and in that log it is *negatively*
    correlated with reward (r ≈ -0.74) — the long outputs are mostly wrong, so this
    is degeneration rather than a successful reward hack. Mitigations, in order of
    effectiveness: clipped policy ratios (7_4), a token cap, a format reward (7_6).

??? question "`entropy_avg` climbs to ~11.9 and accuracy is 0"

    The policy has collapsed to a uniform distribution over the vocabulary —
    ln(151,936) = 11.93. In the repo's KL run, `kl_loss` equals total `loss`,
    meaning the KL term took over the objective entirely. Kill the run, reduce the
    KL coefficient, or drop the term.

??? question "`adv_std` is 0.0"

    Every rollout in the group got the same reward. No signal, no learning. Use more
    rollouts or harder problems.

??? question "Verifier returns `None` mid-training"

    Deliberate: `sympy_parser` rejects expressions over 2,000 characters so a
    degenerate policy cannot crash SymPy. If it fires constantly, your policy is
    rambling.

## Distillation

??? question "Validation loss never falls"

    Usually the tokenizer. Student weights are the **base** model; the tokenizer must
    be `load_reasoning_tokenizer()`. Also check that
    `tokenizer.encode(answer, chat_wrapped=False)` is used for the target — wrapping
    it nests template tokens inside the answer.

??? question "`load_state_dict` key mismatch"

    GRPO and distillation checkpoints on Hugging Face are plain PyTorch
    `state_dict`s, not Transformers checkpoints. Load into
    `Qwen3Model(QWEN_CONFIG_06_B)`, not `AutoModel`.

??? question "Train loss falls, validation turns upward"

    Memorising the teacher's phrasing. Stop earlier, use more data, or filter more
    aggressively by length.

## Environment

??? question "Chainlit will not install"

    No Python 3.14 support. Pin 3.13 for the chat UI only.

??? question "PyTorch version drift breaks the lockfile"

    ```bash
    uv lock --upgrade-package torch     # accept a newer 2.x
    git restore uv.lock && uv sync      # or go back to 2.10.0
    ```

??? question "`uv` complains about `exclude-newer`"

    `uv self update`. The repo's `pyproject.toml` requires a recent `uv`.
