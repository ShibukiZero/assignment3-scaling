from __future__ import annotations

import importlib
import math

from cs336_scaling.api_contract import TrainingConfig


def load_backend_budget_module():
    try:
        return importlib.import_module("cs336_scaling.training_budget")
    except ModuleNotFoundError as exc:
        raise AssertionError(
            "Expected module cs336_scaling.training_budget to exist with backend budget helpers."
        ) from exc


def build_config(**overrides: object) -> TrainingConfig:
    payload: dict[str, object] = {
        "api_key": "demo-key",
        "d_model": 512,
        "num_layers": 8,
        "num_heads": 8,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "train_flops": int(1e16),
    }
    payload.update(overrides)
    return TrainingConfig(**payload)


def test_backend_budget_plan_exposes_expected_fields() -> None:
    training_budget = load_backend_budget_module()
    config = build_config()

    plan = training_budget.build_training_plan(config=config, context_length=512)

    assert plan.non_embedding_parameters == 12 * 8 * 512 * 512
    assert math.isclose(plan.train_tokens, int(1e16) / (6 * (12 * 8 * 512 * 512)))
    assert plan.tokens_per_step == 128 * 512
    assert plan.max_steps == 1010
    assert plan.context_length == 512


def test_backend_budget_plan_has_lower_bound_of_one_step() -> None:
    training_budget = load_backend_budget_module()
    config = build_config(
        d_model=1024,
        num_layers=24,
        num_heads=16,
        batch_size=256,
        train_flops=int(1e13),
    )

    plan = training_budget.build_training_plan(config=config, context_length=512)

    assert plan.max_steps == 1


def test_backend_budget_plan_respects_context_length_override() -> None:
    training_budget = load_backend_budget_module()
    config = build_config(batch_size=256, train_flops=int(3e16))

    plan = training_budget.build_training_plan(config=config, context_length=256)

    assert plan.context_length == 256
    assert plan.tokens_per_step == 256 * 256
    assert plan.max_steps == 3031


def test_backend_budget_plan_reports_effective_token_budget_after_step_rounding() -> None:
    training_budget = load_backend_budget_module()
    config = build_config()

    plan = training_budget.build_training_plan(config=config, context_length=512)

    assert plan.effective_train_tokens == plan.max_steps * plan.tokens_per_step
    assert plan.effective_train_tokens <= plan.train_tokens


def test_backend_budget_plan_rejects_non_positive_context_length() -> None:
    training_budget = load_backend_budget_module()
    config = build_config()

    try:
        training_budget.build_training_plan(config=config, context_length=0)
    except ValueError as exc:
        assert str(exc) == "context_length must be positive, got 0"
    else:
        raise AssertionError("Expected build_training_plan to reject context_length=0")
