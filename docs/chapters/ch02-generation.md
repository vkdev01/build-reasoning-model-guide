# Chapter 2 · Generation and KV caching

!!! abstract "In one line"

    Load a real 0.6B transformer, generate text one token at a time, then make it
    fast with a KV cache — and understand exactly what that cache stores.

## Learning objectives

- Encode text to token IDs and decode back, using Qwen3's own tokenizer.
- Implement greedy decoding and a streaming generator.
- Explain what a KV cache stores, and compute its memory cost from the model config.
- Measure the speed-up from caching and from `torch.compile` on your own hardware.

## The model you are working with

Qwen3-0.6B is defined by one dictionary in `reasoning_from_scratch/qwen3.py`. Read
it before anything else, because every shape downstream follows from it:

```python title="reasoning_from_scratch/qwen3.py"
QWEN_CONFIG_06_B = {
    "vocab_size": 151_936,     # Vocabulary size
    "context_length": 40_960,  # Length originally used during training
    "emb_dim": 1024,           # Embedding dimension
    "n_heads": 16,             # Number of attention heads
    "n_layers": 28,            # Number of layers
    "hidden_dim": 3072,        # Size of intermediate dim in FeedForward
    "head_dim": 128,           # Size of the heads in GQA
    "qk_norm": True,           # Whether to normalize queries & keys in GQA
    "n_kv_groups": 8,          # Key-Value groups for GQA
    "rope_base": 1_000_000.0,  # The base in RoPE's "theta"
    "dtype": torch.bfloat16,   # Lower-precision dtype to reduce memory
}
```

Three numbers matter for the rest of this chapter:

- **`n_layers = 28`** — the cache holds one entry per layer, which is why the
  `KVCache` constructor takes `n_layers`.
- **`n_kv_groups = 8`** with **`n_heads = 16`** — this is *grouped-query
  attention*. Sixteen query heads share eight key/value heads, so the cache is
  half the size it would otherwise be. (Two query heads per KV head.)
- **`head_dim = 128`** — each head is a 128-dimensional vector.

## Tokens

```python title="tokens.py"
from pathlib import Path
from reasoning_from_scratch.qwen3 import Qwen3Tokenizer

tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=Path("qwen3") / "tokenizer-base.json"
)

ids = tokenizer.encode("Explain large language models.")
for i in ids:
    print(f"{i} --> {tokenizer.decode([i])!r}")

print(tokenizer.decode(ids))
```

Two things to notice, both of which bite people later:

1. **Subword tokenisation is not word tokenisation.** "models" may be one token
   or several. The model has never seen a "word" in its life.
2. **Special tokens exist.** `<|im_start|>` and `<|im_end|>` appear when the
   tokenizer is configured with a chat template. If you load the *reasoning*
   tokenizer when you meant the *base* one, you will see these in raw output and
   wonder why the model is broken. It is not; you are.

| Tokenizer | File | Chat template | Thinking |
| --- | --- | --- | --- |
| base | `tokenizer-base.json` | no | no |
| reasoning | `tokenizer-reasoning.json` | yes | yes (`add_thinking=True`) |

## From logits to the next token

A forward pass gives you a tensor of shape `(batch, seq_len, vocab_size)` —
one score per vocabulary entry, per position. You only need the last position:

```python title="reasoning_from_scratch/ch02.py · the essential three lines"
out = model(token_ids)[:, -1]                          # (batch, vocab_size)
next_token = torch.argmax(out, dim=-1, keepdim=True)   # greedy choice
token_ids = torch.cat([token_ids, next_token], dim=1)  # append and repeat
```

`argmax` is greedy decoding: always take the single most likely token. It is
deterministic, which is why the chapter 3 baseline is reproducible, and it is
the opposite of what chapter 4 does on purpose.

## Why the naive loop is wasteful

Without a cache, generating token *n* re-runs the entire prefix through all 28
layers. The attention work is therefore quadratic in sequence length while the
useful work is linear:

| Tokens generated | Forward passes | Attention work (relative) |
| ---: | ---: | ---: |
| 10 | 10 | ~55 |
| 100 | 100 | ~5,050 |
| 1,000 | 1,000 | ~500,500 |

Every one of those passes recomputes keys and values for tokens that have not
changed. They cannot change — attention is causal, so a token's key and value
depend only on the tokens before it. That is the observation the cache exploits.

## What the KV cache stores

For every layer, you keep the **keys and values** computed so far. The queries
are not cached, because you only ever need the query for the token you are
currently generating.

```python title="reasoning_from_scratch/qwen3.py"
class KVCache:
    def __init__(self, n_layers):
        self.cache = [None] * n_layers

    def get(self, layer_idx):
        return self.cache[layer_idx]

    def update(self, layer_idx, value):
        self.cache[layer_idx] = value

    def get_all(self):
        return self.cache

    def reset(self):
        for i in range(len(self.cache)):
            self.cache[i] = None
```

That is the whole data structure: a list with one slot per layer. The model does
`cache.get(layer_idx)` and `cache.update(layer_idx, ...)` inside its attention
blocks. It is deliberately simple — this is the chapter where you should stop
being mystified by the phrase "KV cache".

### How big is it, actually

Compute it from the config. Per token, per layer:

```text
keys   : n_kv_groups × head_dim = 8 × 128 = 1024 values
values : n_kv_groups × head_dim = 8 × 128 = 1024 values
total  : 2048 values per token per layer
× 28 layers                     = 57,344 values per token
× 2 bytes (bfloat16)            = 114,688 bytes ≈ 112 KiB per token
```

So for a single sequence:

| Context length | KV cache |
| ---: | ---: |
| 512 tokens | 56 MiB |
| 1,024 tokens | 112 MiB |
| 2,048 tokens | 224 MiB |
| 8,192 tokens | 896 MiB |
| 40,960 (full context) | 4,480 MiB |

!!! note "This is why GQA exists"

    With 16 KV heads instead of 8, every figure above doubles. Grouped-query
    attention is not a modelling nicety; it is a memory decision. When you get to
    chapter 6 and start generating 1,024-token reasoning traces, this table is
    the reason your GPU is full.

## The cached streaming generator

The function you will use for the rest of the course. Read it closely: the
counter-intuitive part is the `if step + 1 < max_new_tokens` guard.

```python title="reasoning_from_scratch/ch02.py"
@torch.inference_mode()
def generate_text_basic_stream_cache(
    model, token_ids, max_new_tokens, eos_token_id=None
):
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()

    out = model(token_ids, cache=cache)[:, -1]      # prefill: whole prompt at once
    for step in range(max_new_tokens):
        next_token = torch.argmax(out, dim=-1, keepdim=True)

        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break

        yield next_token                             # streaming: hand back one token

        if step + 1 < max_new_tokens:                # skip the final wasted pass
            out = model(next_token, cache=cache)[:, -1]   # decode: ONE token in
```

Four things worth pausing on:

**Prefill vs decode.** The first call passes the entire prompt, filling the
cache. Every later call passes a single token and reuses the cache. That is why
time-to-first-token scales with prompt length but per-token latency does not.

**`yield`, not `return`.** This is a generator, so you see tokens as they arrive.
Every downstream chapter needs that: chapter 4's sampler needs to intercept each
token, chapter 5's refiner needs to stream a critique.

**`torch.all`, not `.item()`.** The non-streaming variant used `next_token.item()
== eos_token_id`, which silently assumes batch size 1. The streaming version
uses `torch.all` so it stays correct under batching. If you write your own
version, copy the streaming one.

**The guard saves one pass.** After the last token is emitted there is nothing
left to compute. Skipping it is a 1/`max_new_tokens` saving — trivial here,
but the same instinct (do not compute what you will throw away) is what makes
batched inference in the bonus scripts fast.

## Worked example: measure the speed-up

```python title="wk2_benchmark.py"
import time, torch
from reasoning_from_scratch.ch02 import (
    get_device, generate_text_basic_stream, generate_text_basic_stream_cache,
)

device = get_device()
# ... load tokenizer and model as in the smoke test ...

prompt = "List the first ten prime numbers and explain the sieve of Eratosthenes."
ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)

def bench(fn, label, n=200):
    t0 = time.perf_counter()
    ntok = 0
    for _ in fn(model, ids, max_new_tokens=n, eos_token_id=tokenizer.eos_token_id):
        ntok += 1
    dt = time.perf_counter() - t0
    print(f"{label:22} {ntok:4d} tokens  {dt:6.2f}s  {ntok/dt:6.1f} tok/s")

torch.manual_seed(0); bench(generate_text_basic_stream,       "no cache", n=60)
torch.manual_seed(0); bench(generate_text_basic_stream_cache, "KV cache")
```

Reference point for judging your number: the GRPO training logs in the repo run
at roughly **47 tokens/second** on the author's GPU for this model. If you are
on a laptop CPU you will see low single digits, and that is normal — it is also
why [chapter 6](ch06-grpo.md) is where laptops stop being viable.

!!! warning "Benchmark honestly"

    Use the same `torch.manual_seed`, the same prompt, and the same
    `max_new_tokens`. Generation stops early at EOS, so a benchmark with a
    different prompt is measuring a different amount of work. Note also that the
    first call includes CUDA kernel compilation on GPU — run once to warm up
    before timing.

## `torch.compile` and TF32

`get_device()` does more than pick a device:

```python title="reasoning_from_scratch/ch02.py (abridged)"
if torch.cuda.is_available():
    device = torch.device("cuda")
    if enable_tensor_cores:
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 11):
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            torch.backends.cudnn.conv.fp32_precision = "tf32"
        else:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
```

TF32 trades a little mantissa for a lot of throughput on NVIDIA tensor cores.
The version check exists because PyTorch changed the API between 2.10 and 2.11
— a small detail that tells you the code is maintained against real breakage.

`torch.compile` is then applied in `load_model_and_tokenizer(..., use_compile=True)`.
Expect a meaningful speed-up after the first call, which pays a compilation
cost. On Windows there is a known issue; the repo ships
`ch02/04_torch-compile-windows` for it.

## Bonus material in the repo

| Folder | What it adds |
| --- | --- |
| `ch02/02_setup-tips/` | `python-instructions.md` (pip and uv), `gpu-instructions.md` (cloud GPUs) |
| `ch02/03_optimized-LLM/` | `qwen3_optimized.py` — a faster variant of the model |
| `ch02/04_torch-compile-windows/` | The Windows `torch.compile` workaround |
| `ch02/05_use_model/` | Minimal examples of using the model directly |

## Exercises

1. **Predict, then measure.** Before running the benchmark, compute the expected
   ratio from the table above. How close is your measured number?
2. **Break the cache on purpose.** Set `n_layers=27` in the `KVCache` and run.
   What happens, and what does the error tell you about where the cache is used?
3. **Batch of two.** Write a variant that generates two different prompts at
   once. Watch what `torch.all(next_token == eos_token_id)` does when one
   sequence finishes before the other.
4. **Prompt length vs latency.** Time time-to-first-token for a 10-token prompt
   and a 500-token prompt. Explain the shape of the difference.

## Checkpoint

- Your benchmark prints a tokens/sec figure for both variants, and caching wins
  by several times.
- You can compute the KV cache size for 2,048 tokens without looking (224 MiB).
- You can explain the difference between prefill and decode in one sentence each.

**Notebook:** [`01_generation_and_kv_cache.ipynb`](../notebooks/01_generation_and_kv_cache.ipynb) ·
**Source:** `reasoning_from_scratch/ch02.py`, `reasoning_from_scratch/qwen3.py` ·
**Video:** 1-2 of the course.
