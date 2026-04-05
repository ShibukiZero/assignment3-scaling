# CS336 Spring 2025 Assignment 3: Scaling

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment3_scaling.pdf](./cs336_spring2025_assignment3_scaling.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

0. Install uv

1. Add whatever dependencies you need with `uv add <package>`.

2. Run anything in the given environment with

```sh
uv run <command>
```

3. If you need the Python binary (for instance to reference the Python interpreter for VSCode), run
```sh
uv run which python
```

## Tokenizer And Dataset Infra

This repository now includes a lightweight preprocessing pipeline for training a tokenizer and
encoding a corpus into flat token-id binaries.

A `2-4 GiB` tokenizer-training subset is a reasonable default for a `32K` byte-level BPE tokenizer.
For large web corpora, it is often better to train on a reproducible random subset first instead of
feeding the entire corpus into tokenizer training.

### Supported corpus formats

- `.txt`
- `.txt.gz`
- `.jsonl`
- `.jsonl.gz`
- `.json`
- `.json.gz`
- `.parquet`

For structured formats, the default text field is `text`.

### 1. Sample a tokenizer-training subset

```sh
uv run python -m cs336_scaling.sample_corpus \
  --input /root/autodl-tmp/fineweb-edu/sample \
  --output /root/autodl-tmp/tokenizer/fineweb_edu_3g.jsonl \
  --text-key text \
  --target-bytes 3GiB \
  --max-bytes 4GiB \
  --seed 1337
```

Outputs:

- `fineweb_edu_3g.jsonl`
- `fineweb_edu_3g.jsonl.meta.json`

### 2. Train a tokenizer

This uses the Hugging Face `tokenizers` Rust backend and is a much better fit for large corpora
than the pure Python Assignment 1 implementation.

```sh
uv run python -m cs336_scaling.train_tokenizer \
  --input /root/autodl-tmp/tokenizer/fineweb_edu_3g.jsonl \
  --output-dir /root/autodl-tmp/tokenizer/fineweb_edu_32k \
  --vocab-size 32000 \
  --text-key text \
  --special-tokens "<|endoftext|>"
```

Outputs:

- `tokenizer.json`
- `training_report.json`

### 3. Encode the corpus once

After the tokenizer is fixed, you usually encode the corpus once and reuse the token IDs for all
training runs.

```sh
uv run python -m cs336_scaling.encode_dataset \
  --input /root/autodl-tmp/fineweb-edu \
  --tokenizer /root/autodl-tmp/tokenizer/fineweb_edu_32k/tokenizer.json \
  --output-prefix /root/autodl-tmp/tokids/fineweb_edu_full/train \
  --text-key text \
  --append-eod
```

Outputs:

- `train.bin`
- `train.idx`
- `train.meta.json`

### When you need to rerun preprocessing

You normally do **not** need to retrain the tokenizer or re-encode the dataset for every model run.
Rerun preprocessing only if one of these changes:

- the source corpus or split definition
- the tokenizer vocabulary size or special tokens
- the tokenizer training recipe
- the document-boundary convention, such as whether `<|endoftext|>` is appended

If none of those change, the tokenizer artifacts and tokenized dataset are reusable across all
later training or API-serving experiments.

## Assignment API Scaffold

The repository now includes an assignment-style API scaffold with the same three public endpoints
described in the handout:

- `GET /loss`
- `GET /total_flops_used`
- `GET /previous_runs`

The framework already handles:

- parameter validation against the assignment ranges
- caching identical queries per `api_key`
- FLOPs accounting per `api_key`
- persistent run history in SQLite

The API now supports a real CPU-first training backend for cache misses. When the runtime
environment is configured with a tokenized training corpus and vocabulary size, `/loss` will:

- build a training plan from the assignment FLOPs budget
- train on the tokenized training corpus
- return the final **training loss**
- cache the result in SQLite so repeated identical queries do not retrain

### Run the API

```sh
uv run python -m cs336_scaling.api_server --host 0.0.0.0 --port 8000
```

Optional environment variables:

- `CS336_API_DB_PATH`: SQLite path for cached runs and FLOPs accounting
- `CS336_API_ACCEPT_ALL_KEYS=1`: accept any non-empty API key (default)
- `CS336_API_KEYS`: comma-separated allowlist when you want fixed API keys
- `CS336_TRAIN_DATA_META_PATH`: path to `train.meta.json` for the tokenized training corpus
- `CS336_VOCAB_SIZE`: tokenizer vocabulary size used by the model
- `CS336_CONTEXT_LENGTH`: sequence length for training and FLOPs planning (default `512`)
- `CS336_DEVICE`: training device, such as `cpu` or `cuda` (default `cpu`)
- `CS336_MAX_STEPS_CAP`: optional hard cap on training steps for smoke tests or CPU debugging

### Current backend status

The current mainline now supports:

- assignment-style API validation and error semantics
- per-`api_key` SQLite caching and FLOPs accounting
- tokenized dataset loading from `.bin/.idx/.meta.json`
- training-budget planning from the handout approximation
- a minimal real backend that returns final **training loss**

### Example: run the real backend on CPU

```sh
CS336_TRAIN_DATA_META_PATH=/root/autodl-tmp/tokids/fineweb_edu_full/train.meta.json \
CS336_VOCAB_SIZE=32000 \
CS336_API_DB_PATH=/root/autodl-tmp/api/api.db \
CS336_DEVICE=cpu \
uv run python -m cs336_scaling.api_server --host 0.0.0.0 --port 8000
```

For a quick smoke test on CPU, you can also add a small step cap:

```sh
CS336_TRAIN_DATA_META_PATH=/root/autodl-tmp/tokids/fineweb_edu_full/train.meta.json \
CS336_VOCAB_SIZE=32000 \
CS336_API_DB_PATH=/root/autodl-tmp/api/api.db \
CS336_DEVICE=cpu \
CS336_MAX_STEPS_CAP=2 \
uv run python -m cs336_scaling.api_server --host 0.0.0.0 --port 8000
```
