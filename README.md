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
- the assignment-style `2e18` scaling-law FLOPs hard cap
- persistent run history in SQLite

The API now supports a real training backend for cache misses. When the runtime environment is
configured with a tokenized training corpus and vocabulary size, `/loss` will:

- build a training plan from the assignment FLOPs budget
- train on the tokenized training corpus
- return the final **training loss**
- cache the result in SQLite so repeated identical queries do not retrain

### What this replica matches

- endpoint names and main request parameters from the handout
- training-loss semantics for `/loss`
- repeated-query caching semantics
- per-`api_key` FLOPs accounting
- the `2e18` scaling-law budget cap
- `404` for invalid hyperparameters and `422` for missing history

### What still differs from the official handout API

- this local replica trains on the local tokenized corpus, not the hidden official backend
- by default it accepts any non-empty API key unless allowlist mode is enabled
- error bodies still use FastAPI's `{"detail": {"message": ...}}` wrapper instead of a bare
  `{"message": ...}` object
- the local training data and tokenizer are not the official SlimPajama-based artifacts

### Run the API

```sh
uv run python -m cs336_scaling.api_server --host 0.0.0.0 --port 8000
```

You can also use the helper script:

```sh
./scripts/start_api.sh
```

The helper script auto-detects `cuda` first and falls back to `cpu` if no GPU is visible. You can
still force a specific device:

```sh
./scripts/start_api.sh cpu
```

Optional environment variables:

- `CS336_API_DB_PATH`: SQLite path for cached runs and FLOPs accounting
- `CS336_API_ACCEPT_ALL_KEYS=1`: accept any non-empty API key (default)
- `CS336_API_KEYS`: comma-separated allowlist when you want fixed API keys
- `CS336_TRAIN_DATA_META_PATH`: path to `train.meta.json` for the tokenized training corpus
- `CS336_VOCAB_SIZE`: tokenizer vocabulary size used by the model
- `CS336_CONTEXT_LENGTH`: sequence length for training and FLOPs planning (default `512`)
- `CS336_DEVICE`: training device, such as `cpu` or `cuda` (default `cpu`)
- `CS336_MIXED_PRECISION`: mixed precision mode, one of `off`, `bf16`, or `fp16`
- `CS336_ACTIVATION_CHECKPOINTING`: `1` to enable checkpointing, `0` to disable it
- `CS336_PRELOAD_DATASET`: `1` to preload the tokenized corpus into RAM on startup, `0` to lazy-load it per request
- `CS336_MAX_STEPS_CAP`: optional hard cap on training steps for smoke tests or CPU debugging

### Runtime defaults

When you start the service with `./scripts/start_api.sh`, the current defaults are:

- prefer `cuda` if a visible GPU exists, otherwise use `cpu`
- use `bf16` automatically on CUDA runs
- enable activation checkpointing automatically on CUDA runs
- preload the tokenized dataset into RAM on startup
- accept any non-empty API key unless allowlist mode is enabled

### Current backend status

The current mainline now supports:

- assignment-style API validation and error semantics
- per-`api_key` SQLite caching and FLOPs accounting
- tokenized dataset loading from `.bin/.idx/.meta.json`
- training-budget planning from the handout approximation
- a minimal real backend that returns final **training loss**
- graceful request-level OOM handling so a single oversized query does not crash the whole service

The current GPU path has been smoke-tested on the largest handout-legal configuration:

- `d_model=1024`
- `num_layers=24`
- `num_heads=16`
- `batch_size=128`
- `batch_size=256`

### Example: run the real backend on CPU

```sh
CS336_TRAIN_DATA_META_PATH=/root/autodl-tmp/tokids/fineweb_edu_full/train.meta.json \
CS336_VOCAB_SIZE=32000 \
CS336_API_DB_PATH=/root/autodl-tmp/api/api.db \
CS336_DEVICE=cpu \
uv run python -m cs336_scaling.api_server --host 0.0.0.0 --port 8000
```

Equivalent helper-script command:

```sh
./scripts/start_api.sh cpu
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

Equivalent helper-script command:

```sh
CS336_MAX_STEPS_CAP=2 ./scripts/start_api.sh cpu
```

### Example: run the real backend on GPU

```sh
CS336_TRAIN_DATA_META_PATH=/root/autodl-tmp/tokids/fineweb_edu_full/train.meta.json \
CS336_VOCAB_SIZE=32000 \
CS336_API_DB_PATH=/root/autodl-tmp/api/api.db \
CS336_DEVICE=cuda \
uv run python -m cs336_scaling.api_server --host 0.0.0.0 --port 8000
```

Equivalent helper-script command:

```sh
./scripts/start_api.sh cuda
```

For an initial GPU smoke test, it is still a good idea to keep a very small cap:

```sh
CS336_TRAIN_DATA_META_PATH=/root/autodl-tmp/tokids/fineweb_edu_full/train.meta.json \
CS336_VOCAB_SIZE=32000 \
CS336_API_DB_PATH=/root/autodl-tmp/api/api.db \
CS336_DEVICE=cuda \
CS336_MAX_STEPS_CAP=2 \
uv run python -m cs336_scaling.api_server --host 0.0.0.0 --port 8000
```

Equivalent helper-script command:

```sh
CS336_MAX_STEPS_CAP=2 ./scripts/start_api.sh cuda
```

By default, the backend uses `bf16` whenever `CS336_DEVICE` starts with `cuda`. You can override
that explicitly if needed:

```sh
CS336_MIXED_PRECISION=fp16 ./scripts/start_api.sh cuda
CS336_MIXED_PRECISION=off ./scripts/start_api.sh cuda
```

Checkpointing also defaults on for CUDA runs and off for CPU runs. You can override it explicitly:

```sh
CS336_ACTIVATION_CHECKPOINTING=1 ./scripts/start_api.sh cuda
CS336_ACTIVATION_CHECKPOINTING=0 ./scripts/start_api.sh cuda
```

Dataset preloading defaults on, so the service loads the tokenized corpus into memory during
startup and reuses it across requests:

```sh
./scripts/start_api.sh
CS336_PRELOAD_DATASET=0 ./scripts/start_api.sh
```

### Reset the local API database

If you want to clear local run history and FLOPs accounting, remove the SQLite file and restart the
service:

```sh
rm -f /root/autodl-tmp/api/api.db
bash scripts/start_api.sh
```

### Example: smoke-test the API with curl

Start the server in one terminal, then issue these requests from another terminal:

```sh
curl "http://127.0.0.1:8000/loss?d_model=64&num_layers=2&num_heads=2&batch_size=128&learning_rate=0.0003&train_flops=10000000000000&api_key=test-key"
```

Repeat the same query to confirm the cache behavior:

```sh
curl "http://127.0.0.1:8000/loss?d_model=64&num_layers=2&num_heads=2&batch_size=128&learning_rate=0.0003&train_flops=10000000000000&api_key=test-key"
```

Then verify FLOPs accounting and run history:

```sh
curl "http://127.0.0.1:8000/total_flops_used?api_key=test-key"
curl "http://127.0.0.1:8000/previous_runs?api_key=test-key"
```

Expected behavior:

- the first `/loss` request should return a float `loss` and `total_flops_used=1e13`
- the second identical `/loss` request should return the same `loss` without increasing `total_flops_used`
- `/total_flops_used` should still report `1e13`
- `/previous_runs` should contain exactly one matching run

### Batch smoke-suite script

If you want to run a longer curl-based sweep and save every response to disk, use:

```sh
bash scripts/run_api_smoke_suite.sh
```

By default this writes a new timestamped directory under `.agents/logs/`, for example:

```text
.agents/logs/api_smoke_20260406-013000/
```

Each case writes:

- one `*.url.txt` file with the exact request URL
- one `*.status.txt` file with the HTTP status code
- one `*.body.json` file with the raw response body
- one `summary.tsv` file covering the whole suite

The suite does **not** include the `2e18` hard-cap check by default, because a real backend would
need to execute two `1e18` runs to reach the limit. If you are running against a smoke-test server
with a very small `CS336_MAX_STEPS_CAP`, you can opt in:

```sh
INCLUDE_CAP_TESTS=1 bash scripts/run_api_smoke_suite.sh
```

### Recommended overnight workflow

1. Start the service:

```sh
bash scripts/start_api.sh
```

2. Run the saved curl sweep in another terminal:

```sh
bash scripts/run_api_smoke_suite.sh
```

3. In the morning, inspect:

```sh
cat .agents/logs/api_smoke_*/summary.tsv
cat .agents/logs/api_smoke_*/metadata.txt
```
