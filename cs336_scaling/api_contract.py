from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


MIN_D_MODEL = 64
MAX_D_MODEL = 1024
MIN_NUM_LAYERS = 2
MAX_NUM_LAYERS = 24
MIN_NUM_HEADS = 2
MAX_NUM_HEADS = 16
MIN_LEARNING_RATE = 1e-4
MAX_LEARNING_RATE = 1e-3
ALLOWED_BATCH_SIZES = (128, 256)
ALLOWED_TRAIN_FLOPS = (
    int(1e13),
    int(3e13),
    int(6e13),
    int(1e14),
    int(3e14),
    int(6e14),
    int(1e15),
    int(3e15),
    int(6e15),
    int(1e16),
    int(3e16),
    int(6e16),
    int(1e17),
    int(3e17),
    int(6e17),
    int(1e18),
)


@dataclass(frozen=True)
class TrainingConfig:
    api_key: str
    d_model: int
    num_layers: int
    num_heads: int
    batch_size: int
    learning_rate: float
    train_flops: int

    def to_public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("api_key", None)
        return payload


def validate_api_key(api_key: str | None) -> str:
    if api_key is None or api_key == "":
        raise ValueError("api_key must be provided.")
    return api_key


def validate_d_model(value: int) -> int:
    if not MIN_D_MODEL <= value <= MAX_D_MODEL:
        raise ValueError(f"d_model must be in range [{MIN_D_MODEL}, {MAX_D_MODEL}], got {value}")
    return value


def validate_num_layers(value: int) -> int:
    if not MIN_NUM_LAYERS <= value <= MAX_NUM_LAYERS:
        raise ValueError(
            f"num_layers must be in range [{MIN_NUM_LAYERS}, {MAX_NUM_LAYERS}], got {value}"
        )
    return value


def validate_num_heads(value: int) -> int:
    if not MIN_NUM_HEADS <= value <= MAX_NUM_HEADS:
        raise ValueError(f"num_heads must be in range [{MIN_NUM_HEADS}, {MAX_NUM_HEADS}], got {value}")
    return value


def validate_batch_size(value: int) -> int:
    if value not in ALLOWED_BATCH_SIZES:
        raise ValueError(f"batch_size must be one of {set(ALLOWED_BATCH_SIZES)}, got {value}")
    return value


def validate_learning_rate(value: float) -> float:
    if not MIN_LEARNING_RATE <= value <= MAX_LEARNING_RATE:
        raise ValueError(
            f"learning_rate must be in range [{MIN_LEARNING_RATE}, {MAX_LEARNING_RATE}], got {value}"
        )
    return value


def validate_train_flops(value: int) -> int:
    if value not in ALLOWED_TRAIN_FLOPS:
        raise ValueError(f"train_flops must be one of {list(ALLOWED_TRAIN_FLOPS)}, got {value}")
    return value


def validate_head_compatibility(d_model: int, num_heads: int) -> None:
    if d_model % num_heads != 0:
        raise ValueError(
            f"d_model must be divisible by num_heads, got d_model={d_model} and num_heads={num_heads}"
        )


def build_training_config(
    *,
    api_key: str,
    d_model: int,
    num_layers: int,
    num_heads: int,
    batch_size: int,
    learning_rate: float,
    train_flops: int,
) -> TrainingConfig:
    api_key = validate_api_key(api_key)
    d_model = validate_d_model(d_model)
    num_layers = validate_num_layers(num_layers)
    num_heads = validate_num_heads(num_heads)
    batch_size = validate_batch_size(batch_size)
    learning_rate = validate_learning_rate(learning_rate)
    train_flops = validate_train_flops(train_flops)
    validate_head_compatibility(d_model, num_heads)
    return TrainingConfig(
        api_key=api_key,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        batch_size=batch_size,
        learning_rate=learning_rate,
        train_flops=train_flops,
    )


def estimate_non_embedding_parameters(d_model: int, num_layers: int) -> int:
    return 12 * num_layers * d_model * d_model


def estimate_train_tokens(train_flops: int, d_model: int, num_layers: int) -> float:
    non_embedding_parameters = estimate_non_embedding_parameters(d_model, num_layers)
    return train_flops / (6 * non_embedding_parameters)
