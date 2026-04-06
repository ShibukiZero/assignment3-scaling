#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass

from cs336_scaling.api_contract import (
    MAX_D_MODEL,
    MAX_NUM_HEADS,
    MAX_NUM_LAYERS,
    MIN_D_MODEL,
    MIN_NUM_HEADS,
    MIN_NUM_LAYERS,
    estimate_non_embedding_parameters,
)


HEAD_DIM = 64
ANCHOR_D_MODEL = 640
ANCHOR_NUM_LAYERS = 5
ANCHOR_WIDTH_DEPTH_RATIO = ANCHOR_D_MODEL / ANCHOR_NUM_LAYERS


@dataclass(frozen=True)
class ShapeCandidate:
    d_model: int
    num_layers: int
    num_heads: int
    estimated_parameters: int
    relative_parameter_error: float
    family_score: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Given a target non-embedding parameter count N, return the unique main Chapter 3 "
            "shape under the current deterministic shape family."
        )
    )
    parser.add_argument(
        "target_n",
        type=float,
        help="Target non-embedding parameter count.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the output as JSON.",
    )
    return parser


def legal_d_models() -> list[int]:
    lower = max(MIN_D_MODEL, HEAD_DIM * MIN_NUM_HEADS)
    upper = min(MAX_D_MODEL, HEAD_DIM * MAX_NUM_HEADS)
    start = ((lower + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    return list(range(start, upper + 1, HEAD_DIM))


def estimate_continuous_family(target_n: float) -> tuple[float, float, float]:
    d_model = (32.0 * target_n / 3.0) ** (1.0 / 3.0)
    num_layers = d_model / ANCHOR_WIDTH_DEPTH_RATIO
    num_heads = d_model / HEAD_DIM
    return d_model, num_layers, num_heads


def score_candidate(
    *,
    target_n: float,
    d_model: int,
    num_layers: int,
    ideal_d_model: float,
    ideal_num_layers: float,
) -> tuple[float, float]:
    estimated_parameters = estimate_non_embedding_parameters(d_model, num_layers)
    parameter_error = abs(math.log(estimated_parameters / target_n))
    d_model_error = abs(math.log(d_model / ideal_d_model))
    num_layers_error = abs(math.log(num_layers / ideal_num_layers))
    score = parameter_error + 0.35 * d_model_error + 0.35 * num_layers_error
    relative_parameter_error = (estimated_parameters - target_n) / target_n
    return score, relative_parameter_error


def suggest_shape(target_n: float) -> ShapeCandidate:
    ideal_d_model, ideal_num_layers, _ = estimate_continuous_family(target_n)
    candidates: dict[tuple[int, int, int], ShapeCandidate] = {}

    for d_model in legal_d_models():
        num_heads = d_model // HEAD_DIM
        layer_center = target_n / (12.0 * d_model * d_model)
        nearby_layers = {
            round(layer_center),
            math.floor(layer_center),
            math.ceil(layer_center),
            round(d_model / ANCHOR_WIDTH_DEPTH_RATIO),
        }

        for raw_num_layers in nearby_layers:
            num_layers = int(raw_num_layers)
            if not MIN_NUM_LAYERS <= num_layers <= MAX_NUM_LAYERS:
                continue

            score, relative_parameter_error = score_candidate(
                target_n=target_n,
                d_model=d_model,
                num_layers=num_layers,
                ideal_d_model=ideal_d_model,
                ideal_num_layers=ideal_num_layers,
            )
            estimated_parameters = estimate_non_embedding_parameters(d_model, num_layers)
            candidate = ShapeCandidate(
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                estimated_parameters=estimated_parameters,
                relative_parameter_error=relative_parameter_error,
                family_score=score,
            )
            candidates[(d_model, num_layers, num_heads)] = candidate

    ranked = sorted(
        candidates.values(),
        key=lambda candidate: (
            candidate.family_score,
            abs(candidate.relative_parameter_error),
            abs(candidate.d_model - ideal_d_model),
        ),
    )
    return ranked[0]


def main() -> None:
    args = build_parser().parse_args()
    target_n = float(args.target_n)

    ideal_d_model, ideal_num_layers, ideal_num_heads = estimate_continuous_family(target_n)
    best_shape = suggest_shape(target_n=target_n)

    payload = {
        "target_n": target_n,
        "continuous_family_estimate": {
            "d_model": ideal_d_model,
            "num_layers": ideal_num_layers,
            "num_heads": ideal_num_heads,
        },
        "best_shape": asdict(best_shape),
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"Target N: {target_n:.6e}")
    print(
        "Continuous family estimate: "
        f"d_model={ideal_d_model:.2f}, "
        f"num_layers={ideal_num_layers:.2f}, "
        f"num_heads={ideal_num_heads:.2f}"
    )
    print("Best shape:")
    print(
        f"d_model={best_shape.d_model}, "
        f"num_layers={best_shape.num_layers}, "
        f"num_heads={best_shape.num_heads}, "
        f"N={best_shape.estimated_parameters}, "
        f"relative_parameter_error={best_shape.relative_parameter_error:+.4%}, "
        f"family_score={best_shape.family_score:.6f}"
    )


if __name__ == "__main__":
    main()
