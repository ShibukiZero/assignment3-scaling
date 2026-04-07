#!/usr/bin/env python3
"""Plot the final chosen learning-rate rule as a function of model size.

This plot combines:
- the exact-family lookup chosen after the joint bs/lr sweep, and
- the refined large-N results from `3_2_2_large_n_lr_refine`.

The figure emphasizes the final adopted rule for downstream IsoFLOPs sweeps.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ShapeChoice:
    n_params: int
    d_model: int
    num_layers: int
    num_heads: int
    learning_rate: float

    @property
    def label(self) -> str:
        return f"{self.d_model}/{self.num_layers}/{self.num_heads}"


EXACT_FAMILY = [
    (256, 2, 4),
    (384, 3, 6),
    (512, 4, 8),
    (640, 5, 10),
    (768, 6, 12),
    (896, 7, 14),
    (1024, 8, 16),
]


def estimate_n_params(d_model: int, num_layers: int) -> int:
    return 12 * num_layers * d_model * d_model


def load_best_lr(results_path: Path) -> dict[tuple[int, int, int], float]:
    payload = json.loads(results_path.read_text())
    best: dict[tuple[int, int, int], tuple[float, float]] = {}
    for response in payload["responses"]:
        params = response["params_without_api_key"]
        key = (int(params["d_model"]), int(params["num_layers"]), int(params["num_heads"]))
        loss = float(response["body_json"]["loss"])
        learning_rate = float(params["learning_rate"])
        current = best.get(key)
        if current is None or loss < current[1]:
            best[key] = (learning_rate, loss)
    return {key: value[0] for key, value in best.items()}


def build_final_rule(joint_results_path: Path, refine_results_path: Path) -> list[ShapeChoice]:
    joint_best = load_best_lr(joint_results_path)
    refine_best = load_best_lr(refine_results_path)

    final_choices: list[ShapeChoice] = []
    for d_model, num_layers, num_heads in EXACT_FAMILY:
        key = (d_model, num_layers, num_heads)
        learning_rate = refine_best.get(key, joint_best[key])
        final_choices.append(
            ShapeChoice(
                n_params=estimate_n_params(d_model, num_layers),
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                learning_rate=learning_rate,
            )
        )
    return final_choices


def capped_lr_rule(n_params: np.ndarray) -> np.ndarray:
    return np.minimum(1e-3, 9e-4 * (n_params / 42467328.0) ** (-0.4716889721))


def make_plot(choices: list[ShapeChoice], output_path: Path) -> None:
    xs = np.array([choice.n_params for choice in choices], dtype=float)
    ys = np.array([choice.learning_rate for choice in choices], dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    ax.plot(xs, ys, marker="o", linewidth=2.0, color="#1f77b4", label="Chosen exact-family lookup")

    overlay_xs = np.geomspace(xs.min(), xs.max(), 200)
    overlay_ys = capped_lr_rule(overlay_xs)
    ax.plot(
        overlay_xs,
        overlay_ys,
        linestyle="--",
        linewidth=1.8,
        color="#ff7f0e",
        label="Capped large-N interpolation rule",
    )

    for choice in choices:
        ax.annotate(
            choice.label,
            (choice.n_params, choice.learning_rate),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Parameter count N")
    ax.set_ylabel("Chosen learning rate")
    ax.set_title("Final learning-rate rule used in IsoFLOPs sweeps")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--joint-results",
        type=Path,
        default=Path("artifacts/experiments/ch3/3_2_1_bs_lr_joint_sweep/results.json"),
        help="Path to the archived 3_2_1 results JSON.",
    )
    parser.add_argument(
        "--refine-results",
        type=Path,
        default=Path("artifacts/experiments/ch3/3_2_2_large_n_lr_refine/results.json"),
        help="Path to the archived 3_2_2 results JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/experiments/ch3/3_2_2_large_n_lr_refine/lr_rule.png"),
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    choices = build_final_rule(args.joint_results, args.refine_results)
    make_plot(choices, args.output)


if __name__ == "__main__":
    main()
