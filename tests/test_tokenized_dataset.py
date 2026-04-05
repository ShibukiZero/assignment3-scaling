from __future__ import annotations

import importlib
import json
from array import array
from pathlib import Path

def write_toy_tokenized_corpus(tmp_path: Path) -> tuple[Path, Path, Path]:
    output_prefix = tmp_path / "toy_train"
    ids_path = output_prefix.with_suffix(".bin")
    idx_path = output_prefix.with_suffix(".idx")
    meta_path = output_prefix.with_suffix(".meta.json")

    # Three toy documents:
    # doc0 = [11, 12, 13]
    # doc1 = [21, 22]
    # doc2 = [31, 32, 33, 34]
    array("H", [11, 12, 13, 21, 22, 31, 32, 33, 34]).tofile(ids_path.open("wb"))
    array("Q", [0, 3, 5, 9]).tofile(idx_path.open("wb"))
    meta_path.write_text(
        json.dumps(
            {
                "dtype": "uint16",
                "num_documents": 3,
                "num_tokens": 9,
                "ids_path": str(ids_path),
                "idx_path": str(idx_path),
            }
        ),
        encoding="utf-8",
    )
    return ids_path, idx_path, meta_path


def load_token_dataset_module():
    try:
        return importlib.import_module("cs336_scaling.token_dataset")
    except ModuleNotFoundError as exc:
        raise AssertionError(
            "Expected module cs336_scaling.token_dataset to exist with a TokenizedDataset reader."
        ) from exc


def test_tokenized_dataset_reports_document_and_token_counts(tmp_path: Path) -> None:
    _, _, meta_path = write_toy_tokenized_corpus(tmp_path)
    token_dataset_module = load_token_dataset_module()

    dataset = token_dataset_module.TokenizedDataset.from_meta(meta_path)

    assert dataset.num_documents == 3
    assert dataset.num_tokens == 9


def test_tokenized_dataset_reads_one_document_without_crossing_boundaries(tmp_path: Path) -> None:
    _, _, meta_path = write_toy_tokenized_corpus(tmp_path)
    token_dataset_module = load_token_dataset_module()

    dataset = token_dataset_module.TokenizedDataset.from_meta(meta_path)

    assert dataset.get_document(0) == [11, 12, 13]
    assert dataset.get_document(1) == [21, 22]
    assert dataset.get_document(2) == [31, 32, 33, 34]


def test_tokenized_dataset_reads_contiguous_token_blocks_from_flat_storage(tmp_path: Path) -> None:
    _, _, meta_path = write_toy_tokenized_corpus(tmp_path)
    token_dataset_module = load_token_dataset_module()

    dataset = token_dataset_module.TokenizedDataset.from_meta(meta_path)

    assert dataset.get_token_block(start=0, length=4) == [11, 12, 13, 21]
    assert dataset.get_token_block(start=4, length=5) == [22, 31, 32, 33, 34]
