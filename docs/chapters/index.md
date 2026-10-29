# Chapters 2-8

One page per chapter of the course, in the order the author teaches them. Every
page has the same anatomy: objectives, concepts, annotated code, a worked
example, exercises, and a checkpoint you can pass or fail.

| Chapter | Video | You build | Time |
| --- | --- | --- | --- |
| [2 · Generation & KV caching](ch02-generation.md) | 1-2 | Streaming generation with a KV cache | ~6 h |
| [3 · Evaluation & the verifier](ch03-verifier.md) | 3 | A math verifier and a MATH-500 baseline | ~8 h |
| [4 · Inference-time scaling](ch04-inference-scaling.md) | 4 | Sampling, top-p, self-consistency | ~5 h |
| [5 · Self-refinement](ch05-self-refinement.md) | 5 | Scorers, best-of-N, critique-and-revise | ~4 h |
| [6 · GRPO training](ch06-grpo.md) | 6 | RL with verifiable rewards | ~9 h |
| [7 · Reading a training run](ch07-reading-runs.md) | 7 | Clipping, KL, format reward, diagnosis | ~7 h |
| [8 · Distillation](ch08-distillation.md) | 8 | A student trained on teacher traces | ~8 h |

!!! tip "Watch before you read"

    Each video is a live coding session, which means you see the author hit an
    error and fix it. That is most of the value. Watch at 1.5×, then read the
    page here for the concepts the video assumes and the numbers it does not
    dwell on, then run the notebook.
