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

This repository includes a local assignment-style API replica with:

- `GET /loss`
- `GET /total_flops_used`
- `GET /previous_runs`

It supports:

- assignment-range parameter validation
- per-`api_key` caching and FLOPs accounting
- the `2e18` hard cap
- a real training backend for cache misses
- reservation persistence and startup recovery
- worker-pool execution across visible CUDA devices
- request-level OOM handling
- graceful shutdown

Main differences from the official handout API:

- it trains on the local tokenized corpus, not the hidden official backend
- it accepts any non-empty API key by default
- error responses still use FastAPI's `{"detail": {"message": ...}}` wrapper

### Start and stop

Foreground:

```sh
./scripts/start_api.sh
```

Background:

```sh
./scripts/start_api.sh --daemon
```

Stop:

```sh
./scripts/stop_api.sh
```

You can still force a device:

```sh
./scripts/start_api.sh cpu
./scripts/start_api.sh cuda
```

### Key environment variables

- `CS336_API_DB_PATH`
- `CS336_API_ACCEPT_ALL_KEYS`
- `CS336_API_KEYS`
- `CS336_TRAIN_DATA_META_PATH`
- `CS336_VOCAB_SIZE`
- `CS336_CONTEXT_LENGTH`
- `CS336_DEVICE`
- `CS336_DEVICES`
- `CS336_MIXED_PRECISION`
- `CS336_ACTIVATION_CHECKPOINTING`
- `CS336_PRELOAD_DATASET`
- `CS336_MAX_STEPS_CAP`
- `CS336_LOG_DIR`

### Runtime defaults

- prefer `cuda` if available, otherwise `cpu`
- use `bf16` automatically on CUDA runs
- enable activation checkpointing automatically on CUDA runs
- preload the tokenized dataset into RAM
- use all visible CUDA devices when `CS336_DEVICE=cuda`
- accept any non-empty API key unless allowlist mode is enabled

Daemon mode writes:

- `/root/autodl-tmp/api-logs/api-<port>.log`
- `/root/autodl-tmp/api-logs/api-<port>.pid`

### Debugging and inspection

Reservation state for one key:

```sh
curl "http://127.0.0.1:8000/__admin__/reservations?api_key=test-key"
```

Recent reservations globally:

```sh
curl "http://127.0.0.1:8000/__admin__/reservations?limit=20"
```

Tail the fixed daemon log:

```sh
tail -f /root/autodl-tmp/api-logs/api-8000.log
```

The current GPU path has been smoke-tested on the largest handout-legal configuration:

- `d_model=1024`
- `num_layers=24`
- `num_heads=16`
- `batch_size=128`
- `batch_size=256`

### Common commands

Quick smoke run:

```sh
CS336_MAX_STEPS_CAP=2 ./scripts/start_api.sh
```

Manual `/loss` request:

```sh
curl "http://127.0.0.1:8000/loss?d_model=64&num_layers=2&num_heads=2&batch_size=128&learning_rate=0.0003&train_flops=10000000000000&api_key=test-key"
```

Reset the local API database:

```sh
rm -f /root/autodl-tmp/api/api.db
```

Run the full test suite:

```sh
uv run pytest -q
```

Run the API smoke suite:

```sh
bash scripts/run_api_smoke_suite.sh
```

Optional CUDA overrides:

```sh
CS336_MIXED_PRECISION=fp16 ./scripts/start_api.sh cuda
CS336_MIXED_PRECISION=off ./scripts/start_api.sh cuda
CS336_ACTIVATION_CHECKPOINTING=0 ./scripts/start_api.sh cuda
CS336_PRELOAD_DATASET=0 ./scripts/start_api.sh cuda
```

The stop script first asks the local admin shutdown endpoint to drain running jobs gracefully. If
the endpoint is unavailable, it falls back to `SIGTERM`.

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
