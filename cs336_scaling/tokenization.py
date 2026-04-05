from __future__ import annotations

import argparse
import hashlib
import gzip
import json
import sys
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - exercised only when parquet support is missing.
    pq = None


DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"]
DEFAULT_TEXT_KEY = "text"
DEFAULT_PARQUET_BATCH_SIZE = 1024
DEFAULT_PROGRESS_INTERVAL_DOCS = 100_000


@dataclass(frozen=True)
class CorpusStats:
    num_files: int
    num_documents: int
    num_tokens: int = 0


@dataclass(frozen=True)
class CorpusByteStats:
    num_files: int
    num_documents: int
    total_utf8_bytes: int


@dataclass(frozen=True)
class SamplingStats:
    num_files: int
    num_input_documents: int
    num_output_documents: int
    total_input_utf8_bytes: int
    total_output_utf8_bytes: int
    sample_ratio: float


def log_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def path_looks_supported(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(
        (
            ".txt",
            ".txt.gz",
            ".jsonl",
            ".jsonl.gz",
            ".json",
            ".json.gz",
            ".parquet",
        )
    )


def discover_corpus_files(input_path: str | Path) -> list[Path]:
    root = Path(input_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")

    if root.is_file():
        if not path_looks_supported(root):
            raise ValueError(f"Unsupported file type: {root}")
        return [root]

    files = [path for path in sorted(root.rglob("*")) if path.is_file() and path_looks_supported(path)]
    if not files:
        raise ValueError(f"No supported corpus files found under: {root}")
    return files


def open_maybe_gzip(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def extract_text(record: Any, *, text_key: str) -> str | None:
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        value = record.get(text_key)
        return value if isinstance(value, str) else None
    return None


def iter_texts_from_file(
    path: Path,
    *,
    text_key: str = DEFAULT_TEXT_KEY,
    parquet_batch_size: int = DEFAULT_PARQUET_BATCH_SIZE,
) -> Iterator[str]:
    lower_name = path.name.lower()

    if lower_name.endswith(".txt") or lower_name.endswith(".txt.gz"):
        with open_maybe_gzip(path) as handle:
            for line in handle:
                text = line.rstrip("\n")
                if text:
                    yield text
        return

    if lower_name.endswith(".jsonl") or lower_name.endswith(".jsonl.gz"):
        with open_maybe_gzip(path) as handle:
            for raw_line in handle:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                text = extract_text(json.loads(raw_line), text_key=text_key)
                if text:
                    yield text
        return

    if lower_name.endswith(".json") or lower_name.endswith(".json.gz"):
        with open_maybe_gzip(path) as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            for record in payload:
                text = extract_text(record, text_key=text_key)
                if text:
                    yield text
        else:
            text = extract_text(payload, text_key=text_key)
            if text:
                yield text
        return

    if lower_name.endswith(".parquet"):
        if pq is None:
            raise ImportError(
                "Parquet support requires pyarrow. Install project dependencies before encoding parquet corpora."
            )
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(columns=[text_key], batch_size=parquet_batch_size):
            column = batch.column(0)
            for value in column.to_pylist():
                if isinstance(value, str) and value:
                    yield value
        return

    raise ValueError(f"Unsupported file type: {path}")


def iter_corpus_texts(
    input_path: str | Path,
    *,
    text_key: str = DEFAULT_TEXT_KEY,
    parquet_batch_size: int = DEFAULT_PARQUET_BATCH_SIZE,
) -> Iterator[str]:
    for path in discover_corpus_files(input_path):
        yield from iter_texts_from_file(
            path,
            text_key=text_key,
            parquet_batch_size=parquet_batch_size,
        )


def count_documents(
    input_path: str | Path,
    *,
    text_key: str = DEFAULT_TEXT_KEY,
    parquet_batch_size: int = DEFAULT_PARQUET_BATCH_SIZE,
    progress_interval_docs: int = DEFAULT_PROGRESS_INTERVAL_DOCS,
) -> int:
    total = 0
    files = discover_corpus_files(input_path)
    num_files = len(files)
    started_at = time.time()

    for index, path in enumerate(files, start=1):
        docs_in_file = 0
        for _ in iter_texts_from_file(
            path,
            text_key=text_key,
            parquet_batch_size=parquet_batch_size,
        ):
            docs_in_file += 1
            total += 1
            if progress_interval_docs > 0 and total % progress_interval_docs == 0:
                elapsed = time.time() - started_at
                log_progress(
                    f"[scan] counted {total:,} documents in {elapsed:.1f}s "
                    f"(current file {index}/{num_files}: {path.name})"
                )

        elapsed = time.time() - started_at
        log_progress(
            f"[scan] finished file {index}/{num_files}: {path.name} "
            f"({docs_in_file:,} docs, total {total:,}, {elapsed:.1f}s elapsed)"
        )

    return total


def scan_corpus_bytes(
    input_path: str | Path,
    *,
    text_key: str = DEFAULT_TEXT_KEY,
    parquet_batch_size: int = DEFAULT_PARQUET_BATCH_SIZE,
    progress_interval_docs: int = DEFAULT_PROGRESS_INTERVAL_DOCS,
) -> CorpusByteStats:
    files = discover_corpus_files(input_path)
    started_at = time.time()
    total_docs = 0
    total_utf8_bytes = 0

    for index, path in enumerate(files, start=1):
        docs_in_file = 0
        bytes_in_file = 0
        for text in iter_texts_from_file(
            path,
            text_key=text_key,
            parquet_batch_size=parquet_batch_size,
        ):
            encoded_bytes = len(text.encode("utf-8"))
            docs_in_file += 1
            bytes_in_file += encoded_bytes
            total_docs += 1
            total_utf8_bytes += encoded_bytes

            if progress_interval_docs > 0 and total_docs % progress_interval_docs == 0:
                elapsed = time.time() - started_at
                gib = total_utf8_bytes / (1024**3)
                log_progress(
                    f"[scan_bytes] counted {total_docs:,} documents / {gib:.2f} GiB "
                    f"in {elapsed:.1f}s (current file {index}/{len(files)}: {path.name})"
                )

        elapsed = time.time() - started_at
        file_gib = bytes_in_file / (1024**3)
        total_gib = total_utf8_bytes / (1024**3)
        log_progress(
            f"[scan_bytes] finished file {index}/{len(files)}: {path.name} "
            f"({docs_in_file:,} docs, {file_gib:.2f} GiB, total {total_gib:.2f} GiB, {elapsed:.1f}s elapsed)"
        )

    return CorpusByteStats(
        num_files=len(files),
        num_documents=total_docs,
        total_utf8_bytes=total_utf8_bytes,
    )


def parse_bytes_arg(raw_value: str) -> int:
    value = raw_value.strip().lower().replace("_", "")
    units = {
        "b": 1,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "tb": 1000**4,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
        "tib": 1024**4,
    }

    for suffix, multiplier in sorted(units.items(), key=lambda item: len(item[0]), reverse=True):
        if value.endswith(suffix):
            number = float(value[: -len(suffix)])
            return int(number * multiplier)

    return int(float(value))


def format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024**3):.2f} GiB"


def score_document(seed: int, doc_index: int) -> float:
    payload = f"{seed}:{doc_index}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    integer = int.from_bytes(digest, byteorder="big", signed=False)
    return integer / 2**64


def sample_corpus_to_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    target_bytes: int,
    max_bytes: int | None = None,
    seed: int = 0,
    text_key: str = DEFAULT_TEXT_KEY,
    parquet_batch_size: int = DEFAULT_PARQUET_BATCH_SIZE,
    progress_interval_docs: int = DEFAULT_PROGRESS_INTERVAL_DOCS,
) -> SamplingStats:
    if target_bytes <= 0:
        raise ValueError(f"target_bytes must be positive, got {target_bytes}.")
    if max_bytes is not None and max_bytes < target_bytes:
        raise ValueError("max_bytes must be greater than or equal to target_bytes.")

    log_progress("[sample_corpus] phase 1/2: scanning corpus byte size")
    corpus_stats = scan_corpus_bytes(
        input_path,
        text_key=text_key,
        parquet_batch_size=parquet_batch_size,
        progress_interval_docs=progress_interval_docs,
    )
    if corpus_stats.total_utf8_bytes == 0:
        raise ValueError("Corpus appears to be empty after text extraction.")

    sample_ratio = min(1.0, target_bytes / corpus_stats.total_utf8_bytes)
    hard_cap = max_bytes if max_bytes is not None else target_bytes
    log_progress(
        f"[sample_corpus] phase 1/2 complete: {corpus_stats.num_documents:,} docs, "
        f"{format_gib(corpus_stats.total_utf8_bytes)} total, sample_ratio={sample_ratio:.6f}"
    )
    log_progress(
        f"[sample_corpus] phase 2/2: writing sampled jsonl to {Path(output_path).expanduser()} "
        f"with hard cap {format_gib(hard_cap)}"
    )

    output_file = Path(output_path).expanduser()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    written_docs = 0
    written_utf8_bytes = 0
    seen_docs = 0
    started_at = time.time()

    with output_file.open("w", encoding="utf-8") as handle:
        for text in iter_corpus_texts(
            input_path,
            text_key=text_key,
            parquet_batch_size=parquet_batch_size,
        ):
            seen_docs += 1
            if score_document(seed, seen_docs) > sample_ratio:
                continue

            text_bytes = len(text.encode("utf-8"))
            projected_size = written_utf8_bytes + text_bytes
            if projected_size > hard_cap:
                continue

            record = json.dumps({text_key: text}, ensure_ascii=False)
            handle.write(record)
            handle.write("\n")

            written_docs += 1
            written_utf8_bytes = projected_size

            if progress_interval_docs > 0 and seen_docs % progress_interval_docs == 0:
                elapsed = time.time() - started_at
                log_progress(
                    f"[sample_corpus] processed {seen_docs:,} docs, wrote {written_docs:,} docs / "
                    f"{format_gib(written_utf8_bytes)} in {elapsed:.1f}s"
                )

    elapsed = time.time() - started_at
    log_progress(
        f"[sample_corpus] phase 2/2 complete: wrote {written_docs:,} docs / "
        f"{format_gib(written_utf8_bytes)} in {elapsed:.1f}s"
    )

    return SamplingStats(
        num_files=corpus_stats.num_files,
        num_input_documents=corpus_stats.num_documents,
        num_output_documents=written_docs,
        total_input_utf8_bytes=corpus_stats.total_utf8_bytes,
        total_output_utf8_bytes=written_utf8_bytes,
        sample_ratio=sample_ratio,
    )


def train_byte_level_bpe(
    input_path: str | Path,
    *,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    min_frequency: int = 2,
    text_key: str = DEFAULT_TEXT_KEY,
    parquet_batch_size: int = DEFAULT_PARQUET_BATCH_SIZE,
) -> tuple[Tokenizer, CorpusStats]:
    special_tokens = list(special_tokens or DEFAULT_SPECIAL_TOKENS)
    files = discover_corpus_files(input_path)
    log_progress(f"[train_tokenizer] discovered {len(files)} input files under {Path(input_path).expanduser()}")
    log_progress("[train_tokenizer] phase 1/2: scanning corpus to count documents")
    num_documents = count_documents(
        input_path,
        text_key=text_key,
        parquet_batch_size=parquet_batch_size,
    )
    log_progress(f"[train_tokenizer] phase 1/2 complete: {num_documents:,} documents")

    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    log_progress("[train_tokenizer] phase 2/2: fitting byte-level BPE")
    tokenizer.train_from_iterator(
        iter_corpus_texts(
            input_path,
            text_key=text_key,
            parquet_batch_size=parquet_batch_size,
        ),
        trainer=trainer,
        length=num_documents,
    )
    log_progress("[train_tokenizer] phase 2/2 complete")
    return tokenizer, CorpusStats(num_files=len(files), num_documents=num_documents)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_tokenizer_artifacts(
    tokenizer: Tokenizer,
    output_dir: str | Path,
    *,
    input_path: str | Path,
    stats: CorpusStats,
    vocab_size_target: int,
    min_frequency: int,
    text_key: str,
    special_tokens: list[str],
) -> None:
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer_path = output_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    metadata = {
        "input_path": str(Path(input_path).expanduser()),
        "num_input_files": stats.num_files,
        "num_documents": stats.num_documents,
        "vocab_size_target": vocab_size_target,
        "final_vocab_size": tokenizer.get_vocab_size(),
        "min_frequency": min_frequency,
        "text_key": text_key,
        "special_tokens": special_tokens,
        "tokenizer_path": str(tokenizer_path),
    }
    write_json(output_path / "training_report.json", metadata)


def get_uint_array_type(vocab_size: int) -> str:
    if vocab_size <= 65_535:
        return "H"
    if vocab_size <= 4_294_967_295:
        return "I"
    raise ValueError(f"Vocabulary size {vocab_size} is too large for uint32 serialization.")


def encode_corpus_to_binary(
    input_path: str | Path,
    tokenizer_path: str | Path,
    output_prefix: str | Path,
    *,
    text_key: str = DEFAULT_TEXT_KEY,
    parquet_batch_size: int = DEFAULT_PARQUET_BATCH_SIZE,
    append_eod: bool = True,
    eod_token: str = "<|endoftext|>",
    batch_size: int = 256,
    progress_interval_docs: int = DEFAULT_PROGRESS_INTERVAL_DOCS,
) -> CorpusStats:
    tokenizer = Tokenizer.from_file(str(Path(tokenizer_path).expanduser()))
    vocab_size = tokenizer.get_vocab_size()
    token_array_type = get_uint_array_type(vocab_size)
    output_prefix = Path(output_prefix).expanduser()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    vocab = tokenizer.get_vocab()
    eod_id = vocab.get(eod_token)
    if append_eod and eod_id is None:
        raise ValueError(f"append_eod=True but token {eod_token!r} is not in the tokenizer vocabulary.")

    ids_path = output_prefix.with_suffix(".bin")
    idx_path = output_prefix.with_suffix(".idx")
    meta_path = output_prefix.with_suffix(".meta.json")

    num_files = len(discover_corpus_files(input_path))
    num_documents = 0
    num_tokens = 0
    batch: list[str] = []
    started_at = time.time()

    log_progress(
        f"[encode_dataset] encoding {num_files} input files from {Path(input_path).expanduser()} "
        f"to prefix {output_prefix}"
    )

    with ids_path.open("wb") as ids_handle, idx_path.open("wb") as idx_handle:
        offsets = array("Q", [0])

        def flush_batch() -> None:
            nonlocal num_documents, num_tokens, batch
            if not batch:
                return
            for encoding in tokenizer.encode_batch(batch):
                ids = list(encoding.ids)
                if append_eod and eod_id is not None:
                    ids.append(eod_id)
                array(token_array_type, ids).tofile(ids_handle)
                num_documents += 1
                num_tokens += len(ids)
                offsets.append(num_tokens)
                if progress_interval_docs > 0 and num_documents % progress_interval_docs == 0:
                    elapsed = time.time() - started_at
                    log_progress(
                        f"[encode_dataset] encoded {num_documents:,} docs / {num_tokens:,} tokens "
                        f"in {elapsed:.1f}s"
                    )
            batch = []

        for text in iter_corpus_texts(
            input_path,
            text_key=text_key,
            parquet_batch_size=parquet_batch_size,
        ):
            batch.append(text)
            if len(batch) >= batch_size:
                flush_batch()

        flush_batch()
        offsets.tofile(idx_handle)

    metadata = {
        "input_path": str(Path(input_path).expanduser()),
        "tokenizer_path": str(Path(tokenizer_path).expanduser()),
        "num_input_files": num_files,
        "num_documents": num_documents,
        "num_tokens": num_tokens,
        "ids_path": str(ids_path),
        "idx_path": str(idx_path),
        "dtype": "uint16" if token_array_type == "H" else "uint32",
        "append_eod": append_eod,
        "eod_token": eod_token if append_eod else None,
        "text_key": text_key,
        "parquet_batch_size": parquet_batch_size,
        "encode_batch_size": batch_size,
    }
    write_json(meta_path, metadata)
    elapsed = time.time() - started_at
    log_progress(
        f"[encode_dataset] complete: {num_documents:,} docs / {num_tokens:,} tokens "
        f"written in {elapsed:.1f}s"
    )
    return CorpusStats(num_files=num_files, num_documents=num_documents, num_tokens=num_tokens)


def parse_special_tokens(raw_tokens: str | None) -> list[str]:
    if raw_tokens is None or raw_tokens.strip() == "":
        return list(DEFAULT_SPECIAL_TOKENS)
    return [token.strip() for token in raw_tokens.split(",") if token.strip()]


def add_shared_corpus_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Path to a corpus file or directory.")
    parser.add_argument(
        "--text-key",
        default=DEFAULT_TEXT_KEY,
        help="JSON/JSONL/Parquet field name that contains document text.",
    )
    parser.add_argument(
        "--parquet-batch-size",
        type=int,
        default=DEFAULT_PARQUET_BATCH_SIZE,
        help="Row batch size when streaming parquet shards.",
    )
