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

Before encoding, create a deterministic document-level train/eval split. A `0.5%` eval split is
usually enough for a large web corpus.

```sh
uv run python -m cs336_scaling.split_corpus \
  --input /root/autodl-tmp/fineweb-edu \
  --train-output /root/autodl-tmp/splits/fineweb_edu_train.jsonl \
  --eval-output /root/autodl-tmp/splits/fineweb_edu_eval.jsonl \
  --text-key text \
  --eval-ratio 0.005 \
  --seed 1337
```

Then encode each split separately:

```sh
uv run python -m cs336_scaling.encode_dataset \
  --input /root/autodl-tmp/splits/fineweb_edu_train.jsonl \
  --tokenizer /root/autodl-tmp/tokenizer/fineweb_edu_32k/tokenizer.json \
  --output-prefix /root/autodl-tmp/tokids/fineweb_edu/train \
  --text-key text \
  --append-eod
```

```sh
uv run python -m cs336_scaling.encode_dataset \
  --input /root/autodl-tmp/splits/fineweb_edu_eval.jsonl \
  --tokenizer /root/autodl-tmp/tokenizer/fineweb_edu_32k/tokenizer.json \
  --output-prefix /root/autodl-tmp/tokids/fineweb_edu/eval \
  --text-key text \
  --append-eod
```

Outputs:

- `train.bin` / `train.idx` / `train.meta.json`
- `eval.bin` / `eval.idx` / `eval.meta.json`

### When you need to rerun preprocessing

You normally do **not** need to retrain the tokenizer or re-encode the dataset for every model run.
Rerun preprocessing only if one of these changes:

- the source corpus or split definition
- the tokenizer vocabulary size or special tokens
- the tokenizer training recipe
- the document-boundary convention, such as whether `<|endoftext|>` is appended

If none of those change, the tokenizer artifacts and tokenized dataset are reusable across all
later training or API-serving experiments.
