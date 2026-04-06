from __future__ import annotations

import math

from cs336_scaling.api_contract import estimate_non_embedding_parameters, estimate_train_tokens


CONTEXT_LENGTH = 512


def estimate_steps(*, train_flops: int, d_model: int, num_layers: int, batch_size: int) -> int:
    train_tokens = estimate_train_tokens(
        train_flops=train_flops,
        d_model=d_model,
        num_layers=num_layers,
    )
    tokens_per_step = batch_size * CONTEXT_LENGTH
    return max(1, math.floor(train_tokens / tokens_per_step))


def test_estimate_non_embedding_parameters_matches_assignment_formula() -> None:
    assert estimate_non_embedding_parameters(d_model=64, num_layers=2) == 12 * 2 * 64 * 64
    assert estimate_non_embedding_parameters(d_model=512, num_layers=8) == 12 * 8 * 512 * 512
    assert estimate_non_embedding_parameters(d_model=1024, num_layers=24) == 12 * 24 * 1024 * 1024


def test_estimate_train_tokens_matches_c_equals_6nd() -> None:
    non_embedding_parameters = estimate_non_embedding_parameters(d_model=512, num_layers=8)
    expected_tokens = int(3e16) / (6 * non_embedding_parameters)

    actual_tokens = estimate_train_tokens(train_flops=int(3e16), d_model=512, num_layers=8)

    assert actual_tokens == expected_tokens


def test_estimate_train_tokens_shrinks_as_model_size_grows_for_fixed_flops() -> None:
    small_model_tokens = estimate_train_tokens(train_flops=int(1e16), d_model=256, num_layers=4)
    large_model_tokens = estimate_train_tokens(train_flops=int(1e16), d_model=1024, num_layers=24)

    assert small_model_tokens > large_model_tokens


def test_estimate_steps_uses_context_length_512_and_batch_size() -> None:
    steps_bs128 = estimate_steps(
        train_flops=int(1e16),
        d_model=512,
        num_layers=8,
        batch_size=128,
    )
    steps_bs256 = estimate_steps(
        train_flops=int(1e16),
        d_model=512,
        num_layers=8,
        batch_size=256,
    )

    assert steps_bs128 == 1010
    assert steps_bs256 == 505


def test_estimate_steps_has_flooring_behavior_and_lower_bound_of_one() -> None:
    steps = estimate_steps(
        train_flops=int(1e13),
        d_model=1024,
        num_layers=24,
        batch_size=256,
    )

    assert steps == 1
