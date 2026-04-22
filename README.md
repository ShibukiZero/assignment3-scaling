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
The current examples in this section assume the FineWeb-Edu corpus stored on the remote server.

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

This repository includes a local assignment-style API replica for querying training loss on the
tokenized corpus you prepared above.

Available endpoints:

- `GET /loss`
- `GET /total_flops_used`
- `GET /previous_runs`

What it is useful for:

- validating assignment-style parameter ranges
- caching repeated queries per `api_key`
- tracking FLOPs usage up to the `2e18` cap
- running a real local training backend on cache misses

Main differences from the official handout API:

- it trains on your local tokenized corpus, not the hidden official backend
- it accepts any non-empty API key by default
- error responses still use FastAPI's `{"detail": {"message": ...}}` wrapper

### Minimal usage

Start the server:

```sh
./scripts/start_api.sh
```

Run it in the background instead:

```sh
./scripts/start_api.sh --daemon
```

Stop it:

```sh
./scripts/stop_api.sh
```

If you want to force the device:

```sh
./scripts/start_api.sh cpu
./scripts/start_api.sh cuda
```

### Minimal query flow

Issue a training query:

```sh
curl "http://127.0.0.1:8000/loss?d_model=64&num_layers=2&num_heads=2&batch_size=128&learning_rate=0.0003&train_flops=10000000000000&api_key=test-key"
```

Then inspect accounting and run history:

```sh
curl "http://127.0.0.1:8000/total_flops_used?api_key=test-key"
curl "http://127.0.0.1:8000/previous_runs?api_key=test-key"
```

Repeating the same `/loss` query with the same `api_key` should hit the cache instead of charging
the FLOPs budget again.

### Most useful environment variables

- `CS336_TRAIN_DATA_META_PATH`
- `CS336_VOCAB_SIZE`
- `CS336_DEVICE`
- `CS336_DEVICES`
- `CS336_MAX_STEPS_CAP`
- `CS336_LOG_DIR`

Runtime defaults:

- prefer `cuda` if available, otherwise `cpu`
- prefer `bf16` on CUDA when supported, otherwise fall back to `fp16`
- enable activation checkpointing automatically on CUDA runs
- preload the tokenized dataset into RAM
- use all visible CUDA devices when `CS336_DEVICE=cuda`

If you start the server with `--daemon`, logs and the PID file are written under
`/root/autodl-tmp/api-logs/`.

For a longer curl-based regression sweep, use:

```sh
bash scripts/run_api_smoke_suite.sh
```

The saved responses go under `.agents/logs/`, which makes it easier to compare repeated remote runs.
