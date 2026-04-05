from __future__ import annotations

import json
from array import array
from pathlib import Path

import torch

from cs336_scaling.api_contract import TrainingConfig


def write_runner_corpus(tmp_path: Path) -> Path:
    output_prefix = tmp_path / "runner_train"
    ids_path = output_prefix.with_suffix(".bin")
    idx_path = output_prefix.with_suffix(".idx")
    meta_path = output_prefix.with_suffix(".meta.json")

    tokens = [1, 2, 3, 4, 5, 6, 7, 8] * 16
    array("H", tokens).tofile(ids_path.open("wb"))
    array("Q", [0, len(tokens)]).tofile(idx_path.open("wb"))
    meta_path.write_text(
        json.dumps(
            {
                "dtype": "uint16",
                "num_documents": 1,
                "num_tokens": len(tokens),
                "ids_path": str(ids_path),
                "idx_path": str(idx_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return meta_path


def build_tiny_config(**overrides: object) -> TrainingConfig:
    payload: dict[str, object] = {
        "api_key": "runner-key",
        "d_model": 16,
        "num_layers": 2,
        "num_heads": 2,
        "batch_size": 2,
        "learning_rate": 1e-3,
        "train_flops": 589824,
    }
    payload.update(overrides)
    return TrainingConfig(**payload)


def test_training_runner_completes_tiny_run_and_returns_float_loss(tmp_path: Path) -> None:
    from cs336_scaling.training_runner import TrainingRunner

    meta_path = write_runner_corpus(tmp_path)
    runner = TrainingRunner(
        train_data_meta_path=meta_path,
        vocab_size=32,
        context_length=4,
        device="cpu",
    )
    result = runner.run(build_tiny_config())

    assert isinstance(result.loss, float)
    assert result.steps_completed == 2
    assert result.training_plan.max_steps == 2


def test_training_runner_respects_max_steps_cap(tmp_path: Path) -> None:
    from cs336_scaling.training_runner import TrainingRunner

    meta_path = write_runner_corpus(tmp_path)
    runner = TrainingRunner(
        train_data_meta_path=meta_path,
        vocab_size=32,
        context_length=4,
        device="cpu",
        max_steps_cap=1,
    )
    result = runner.run(build_tiny_config())

    assert result.steps_completed == 1
    assert result.training_plan.max_steps == 1


def test_training_runner_raises_when_corpus_is_too_short_for_context(tmp_path: Path) -> None:
    from cs336_scaling.training_runner import TrainingRunner

    output_prefix = tmp_path / "short_train"
    ids_path = output_prefix.with_suffix(".bin")
    idx_path = output_prefix.with_suffix(".idx")
    meta_path = output_prefix.with_suffix(".meta.json")
    array("H", [1, 2, 3, 4]).tofile(ids_path.open("wb"))
    array("Q", [0, 4]).tofile(idx_path.open("wb"))
    meta_path.write_text(
        json.dumps(
            {
                "dtype": "uint16",
                "num_documents": 1,
                "num_tokens": 4,
                "ids_path": str(ids_path),
                "idx_path": str(idx_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    runner = TrainingRunner(
        train_data_meta_path=meta_path,
        vocab_size=32,
        context_length=4,
        device="cpu",
    )

    try:
        runner.run(build_tiny_config())
    except ValueError as exc:
        assert str(exc) == "Tokenized corpus must contain at least context_length + 1 tokens."
    else:
        raise AssertionError("Expected TrainingRunner to reject too-short tokenized corpora.")


def test_training_runner_defaults_to_disabled_mixed_precision_on_cpu(tmp_path: Path) -> None:
    from cs336_scaling.training_runner import TrainingRunner

    meta_path = write_runner_corpus(tmp_path)
    runner = TrainingRunner(
        train_data_meta_path=meta_path,
        vocab_size=32,
        context_length=4,
        device="cpu",
    )

    assert runner.mixed_precision == "off"
    assert runner.autocast_dtype is None


def test_training_runner_resolves_cuda_bf16_mixed_precision(tmp_path: Path) -> None:
    from cs336_scaling.training_runner import TrainingRunner

    meta_path = write_runner_corpus(tmp_path)
    runner = TrainingRunner(
        train_data_meta_path=meta_path,
        vocab_size=32,
        context_length=4,
        device="cuda",
        mixed_precision="bf16",
    )

    assert runner.mixed_precision == "bf16"
    assert runner.autocast_dtype == torch.bfloat16


def test_training_runner_resolves_cuda_fp16_mixed_precision(tmp_path: Path) -> None:
    from cs336_scaling.training_runner import TrainingRunner

    meta_path = write_runner_corpus(tmp_path)
    runner = TrainingRunner(
        train_data_meta_path=meta_path,
        vocab_size=32,
        context_length=4,
        device="cuda",
        mixed_precision="fp16",
    )

    assert runner.mixed_precision == "fp16"
    assert runner.autocast_dtype == torch.float16


def test_training_runner_rejects_unknown_mixed_precision_mode(tmp_path: Path) -> None:
    from cs336_scaling.training_runner import TrainingRunner

    meta_path = write_runner_corpus(tmp_path)

    try:
        TrainingRunner(
            train_data_meta_path=meta_path,
            vocab_size=32,
            context_length=4,
            device="cuda",
            mixed_precision="weird",
        )
    except ValueError as exc:
        assert str(exc) == "Unsupported mixed precision mode: weird"
    else:
        raise AssertionError("Expected TrainingRunner to reject unknown mixed precision modes.")
