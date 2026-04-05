from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from cs336_scaling.api_contract import TrainingConfig, estimate_non_embedding_parameters, estimate_train_tokens


class BackendUnavailableError(RuntimeError):
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


def get_training_backend() -> TrainingBackend:
    return PlaceholderTrainingBackend()
