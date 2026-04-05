from __future__ import annotations

import math
from dataclasses import dataclass

from cs336_scaling.api_contract import (
    TrainingConfig,
    estimate_non_embedding_parameters,
    estimate_train_tokens,
)


@dataclass(frozen=True)
class TrainingPlan:
    non_embedding_parameters: int
    train_tokens: float
    tokens_per_step: int
    max_steps: int
    effective_train_tokens: int
    context_length: int


def build_training_plan(config: TrainingConfig, context_length: int) -> TrainingPlan:
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")

    non_embedding_parameters = estimate_non_embedding_parameters(
        d_model=config.d_model,
        num_layers=config.num_layers,
    )
    train_tokens = estimate_train_tokens(
        train_flops=config.train_flops,
        d_model=config.d_model,
        num_layers=config.num_layers,
    )
    tokens_per_step = config.batch_size * context_length
    max_steps = max(1, math.floor(train_tokens / tokens_per_step))
    effective_train_tokens = max_steps * tokens_per_step

    return TrainingPlan(
        non_embedding_parameters=non_embedding_parameters,
        train_tokens=train_tokens,
        tokens_per_step=tokens_per_step,
        max_steps=max_steps,
        effective_train_tokens=effective_train_tokens,
        context_length=context_length,
    )
