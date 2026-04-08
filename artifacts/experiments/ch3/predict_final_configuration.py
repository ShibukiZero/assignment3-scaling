#!/usr/bin/env python3
"""Produce the final Chapter 3 prediction at a target compute budget.

The script combines the currently adopted Chapter 3 rules:

- optimal model size law: quadratic-derived `N_opt(C)`
- architecture rule: deterministic family with `head_dim = 64` and a width/depth
  prior anchored at `d_model / num_layers = 128`
- batch size rule: `batch_size = 128`
- learning-rate rule: capped `lr(N)` fit from `3_2_2_large_n_lr_refine`
- training-loss rule: observed-best-loss log-linear fit

The output is a single JSON payload that can be referenced directly in the
final write-up.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from cs336_scaling.api_contract import estimate_non_embedding_parameters


TARGET_BATCH_SIZE = 128
HEAD_DIM = 64
ANCHOR_WIDTH_DEPTH_RATIO = 128.0
MIN_NUM_LAYERS = 2

N_COEFFICIENT = 3.6246761814705494e-05
N_EXPONENT = 0.6816077585644719

LR_MAX = 1e-3
LR_REF_N = 42_467_328.0
LR_REF_VALUE = 9e-4
LR_EXPONENT = -0.4716889721

LOSS_INTERCEPT = 14.923424104464745
LOSS_LOG10_SLOPE = -0.6446054758683744


@dataclass(frozen=True)
class ShapeCandidate:
    d_model: int
    num_layers: int
    num_heads: int
    estimated_parameters: int
    relative_parameter_error: float
    ratio_error: float
    family_score: float


def predicted_optimal_n(target_flops: float) -> float:
    return N_COEFFICIENT * (target_flops ** N_EXPONENT)


def recommended_lr(n_params: float) -> float:
    uncapped = LR_REF_VALUE * (n_params / LR_REF_N) ** LR_EXPONENT
    return min(LR_MAX, uncapped)


def predicted_loss(target_flops: float) -> float:
    return LOSS_INTERCEPT + LOSS_LOG10_SLOPE * math.log10(target_flops)


def candidate_d_models(ideal_d_model: float) -> list[int]:
    center = max(HEAD_DIM, ideal_d_model)
    multipliers = range(-6, 7)
    values = {
        max(HEAD_DIM, int(round((center + delta * HEAD_DIM) / HEAD_DIM) * HEAD_DIM))
        for delta in multipliers
    }
    values.add(int(round(center / HEAD_DIM) * HEAD_DIM))
    return sorted(values)


def continuous_family_estimate(target_n: float) -> tuple[float, float, float]:
    d_model = (32.0 * target_n / 3.0) ** (1.0 / 3.0)
    num_layers = d_model / ANCHOR_WIDTH_DEPTH_RATIO
    num_heads = d_model / HEAD_DIM
    return d_model, num_layers, num_heads


def score_candidate(
    target_n: float,
    d_model: int,
    num_layers: int,
    ideal_d_model: float,
    ideal_num_layers: float,
) -> ShapeCandidate:
    estimated_parameters = estimate_non_embedding_parameters(d_model, num_layers)
    parameter_error = abs(math.log(estimated_parameters / target_n))
    ratio_error = abs(math.log((d_model / num_layers) / ANCHOR_WIDTH_DEPTH_RATIO))
    d_model_error = abs(math.log(d_model / ideal_d_model))
    num_layers_error = abs(math.log(num_layers / ideal_num_layers))
    family_score = parameter_error + 0.35 * ratio_error + 0.20 * d_model_error + 0.20 * num_layers_error
    return ShapeCandidate(
        d_model=d_model,
        num_layers=num_layers,
        num_heads=d_model // HEAD_DIM,
        estimated_parameters=estimated_parameters,
        relative_parameter_error=(estimated_parameters - target_n) / target_n,
        ratio_error=ratio_error,
        family_score=family_score,
    )


def choose_shape(target_n: float) -> ShapeCandidate:
    ideal_d_model, ideal_num_layers, _ = continuous_family_estimate(target_n)
    candidates: list[ShapeCandidate] = []
    for d_model in candidate_d_models(ideal_d_model):
        layer_center = target_n / (12.0 * d_model * d_model)
        nearby_layers = {
            max(MIN_NUM_LAYERS, int(round(layer_center))),
            max(MIN_NUM_LAYERS, int(math.floor(layer_center))),
            max(MIN_NUM_LAYERS, int(math.ceil(layer_center))),
            max(MIN_NUM_LAYERS, int(round(d_model / ANCHOR_WIDTH_DEPTH_RATIO))),
            max(MIN_NUM_LAYERS, int(round(ideal_num_layers))),
        }
        for num_layers in nearby_layers:
            candidates.append(
                score_candidate(
                    target_n=target_n,
                    d_model=d_model,
                    num_layers=num_layers,
                    ideal_d_model=ideal_d_model,
                    ideal_num_layers=ideal_num_layers,
                )
            )
    return min(
        candidates,
        key=lambda item: (
            item.family_score,
            abs(item.relative_parameter_error),
            item.ratio_error,
            -item.estimated_parameters,
        ),
    )


def build_payload(target_flops: float) -> dict[str, object]:
    target_n = predicted_optimal_n(target_flops)
    ideal_d_model, ideal_num_layers, ideal_num_heads = continuous_family_estimate(target_n)
    shape = choose_shape(target_n)
    actual_lr = recommended_lr(shape.estimated_parameters)
    return {
        "target_flops": target_flops,
        "adopted_rules": {
            "n_opt_formula": f"N_opt(C) = {N_COEFFICIENT:.6e} * C^{N_EXPONENT:.6f}",
            "lr_formula": "min(1e-3, 9e-4 * (N / 42467328)^(-0.4716889721))",
            "loss_formula": f"L(C) = {LOSS_INTERCEPT:.6f} + {LOSS_LOG10_SLOPE:.6f} * log10(C)",
            "batch_size": TARGET_BATCH_SIZE,
        },
        "continuous_prediction": {
            "predicted_optimal_n": target_n,
            "ideal_d_model": ideal_d_model,
            "ideal_num_layers": ideal_num_layers,
            "ideal_num_heads": ideal_num_heads,
        },
        "selected_legal_shape": asdict(shape),
        "selected_hyperparameters": {
            "batch_size": TARGET_BATCH_SIZE,
            "learning_rate": actual_lr,
        },
        "predicted_final_training_loss": predicted_loss(target_flops),
        "constraint_notes": {
            "api_service_limits_used": False,
            "selection_rule": (
                "Use an unconstrained discrete architecture near the continuous family "
                "estimate, while preserving head_dim=64 and the width/depth prior."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-flops",
        type=float,
        default=1e19,
        help="Target compute budget.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/experiments/ch3/3_4_3_final_prediction/final_prediction.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Also print the JSON payload to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args.target_flops)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    if args.print:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
