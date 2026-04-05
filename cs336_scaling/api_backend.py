from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from cs336_scaling.api_contract import TrainingConfig, estimate_non_embedding_parameters, estimate_train_tokens
from cs336_scaling.training_runner import TrainingRunner


class BackendUnavailableError(RuntimeError):
    pass


class TrainingOOMError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainingResult:
    loss: float


class TrainingBackend(Protocol):
    def run(self, config: TrainingConfig) -> TrainingResult: ...


class PlaceholderTrainingBackend:
    """Stub backend for the assignment-style API.

    The contract layer, persistence, caching, and FLOPs accounting can be built
    before the real trainer is ready. This backend makes that state explicit so
    the API returns a clear 503 on cache misses instead of silently inventing a loss.
    """

    def run(self, config: TrainingConfig) -> TrainingResult:
        estimated_tokens = estimate_train_tokens(
            train_flops=config.train_flops,
            d_model=config.d_model,
            num_layers=config.num_layers,
        )
        estimated_params = estimate_non_embedding_parameters(
            d_model=config.d_model,
            num_layers=config.num_layers,
        )
        raise BackendUnavailableError(
            "Training backend is not configured yet. "
            f"This query would train roughly {estimated_tokens:.0f} tokens "
            f"for a model with about {estimated_params} non-embedding parameters."
        )


class TorchTrainingBackend:
    def __init__(
        self,
        *,
        train_data_meta_path: str | Path,
        vocab_size: int,
        context_length: int = 512,
        device: str = "cpu",
        mixed_precision: str = "off",
        activation_checkpointing: bool = False,
        preload_dataset: bool = True,
        max_steps_cap: int | None = None,
    ) -> None:
        self.runner = TrainingRunner(
            train_data_meta_path=train_data_meta_path,
            vocab_size=vocab_size,
            context_length=context_length,
            device=device,
            mixed_precision=mixed_precision,
            activation_checkpointing=activation_checkpointing,
            preload_dataset=preload_dataset,
            max_steps_cap=max_steps_cap,
        )

    def run(self, config: TrainingConfig) -> TrainingResult:
        try:
            result = self.runner.run(config)
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" in message:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise TrainingOOMError(
                    "Training run ran out of memory for this configuration."
                ) from exc
            raise
        return TrainingResult(loss=result.loss)


def get_training_backend() -> TrainingBackend:
    train_data_meta_path = os.environ.get("CS336_TRAIN_DATA_META_PATH")
    vocab_size = os.environ.get("CS336_VOCAB_SIZE")
    if not train_data_meta_path or not vocab_size:
        return PlaceholderTrainingBackend()

    context_length = int(os.environ.get("CS336_CONTEXT_LENGTH", "512"))
    device = os.environ.get("CS336_DEVICE", "cpu")
    default_mixed_precision = "bf16" if device.startswith("cuda") else "off"
    mixed_precision = os.environ.get("CS336_MIXED_PRECISION", default_mixed_precision)
    activation_checkpointing_raw = os.environ.get(
        "CS336_ACTIVATION_CHECKPOINTING",
        "1" if device.startswith("cuda") else "0",
    )
    activation_checkpointing = activation_checkpointing_raw == "1"
    preload_dataset_raw = os.environ.get("CS336_PRELOAD_DATASET", "1")
    preload_dataset = preload_dataset_raw == "1"
    max_steps_cap_raw = os.environ.get("CS336_MAX_STEPS_CAP")
    max_steps_cap = int(max_steps_cap_raw) if max_steps_cap_raw else None
    return TorchTrainingBackend(
        train_data_meta_path=train_data_meta_path,
        vocab_size=int(vocab_size),
        context_length=context_length,
        device=device,
        mixed_precision=mixed_precision,
        activation_checkpointing=activation_checkpointing,
        preload_dataset=preload_dataset,
        max_steps_cap=max_steps_cap,
    )
