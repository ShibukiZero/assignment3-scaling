#!/usr/bin/env python3
"""Plot the joint batch-size / learning-rate calibration sweep.

This figure is intended to answer two questions:
1. Does batch size 128 or 256 perform better across the exact-family shapes?
2. How does the preferred learning-rate region change with model size?

The plot uses one panel per batch size and one line per tested learning rate.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Trial:
    n_params: int
    batch_size: int
    learning_rate: float
    loss: float
    label: str


def estimate_n_params(d_model: int, num_layers: int) -> int:
    return 12 * num_layers * d_model * d_model


def load_trials(results_path: Path) -> list[Trial]:
    payload = json.loads(results_path.read_text())
    trials: list[Trial] = []
    for response in payload["responses"]:
        params = response["params_without_api_key"]
        d_model = int(params["d_model"])
        num_layers = int(params["num_layers"])
        num_heads = int(params["num_heads"])
        trials.append(
            Trial(
                n_params=estimate_n_params(d_model, num_layers),
                batch_size=int(params["batch_size"]),
                learning_rate=float(params["learning_rate"]),
                loss=float(response["body_json"]["loss"]),
                label=f"{d_model}/{num_layers}/{num_heads}",
            )
        )
    return trials


def group_trials(trials: list[Trial]) -> dict[int, dict[float, list[Trial]]]:
    grouped: dict[int, dict[float, list[Trial]]] = {}
    for trial in trials:
        grouped.setdefault(trial.batch_size, {}).setdefault(trial.learning_rate, []).append(trial)
    for lr_groups in grouped.values():
        for lr_trials in lr_groups.values():
            lr_trials.sort(key=lambda trial: trial.n_params)
    return dict(sorted(grouped.items()))


def make_plot(trials: list[Trial], output_path: Path) -> None:
    grouped = group_trials(trials)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True, sharey=True)

    cmap = plt.get_cmap("viridis")
    all_lrs = sorted({trial.learning_rate for trial in trials})

    for panel_idx, (batch_size, lr_groups) in enumerate(grouped.items()):
        ax = axes[panel_idx]
        for idx, learning_rate in enumerate(all_lrs):
            lr_trials = lr_groups.get(learning_rate, [])
            if not lr_trials:
                continue
            xs = [trial.n_params for trial in lr_trials]
            ys = [trial.loss for trial in lr_trials]
            color = cmap(idx / max(1, len(all_lrs) - 1))
            ax.plot(xs, ys, marker="o", linewidth=1.8, color=color, label=f"lr={learning_rate:g}")

        # Annotate only the highest-LR line so that the shape order is visible
        # without cluttering every series.
        label_trials = lr_groups.get(max(all_lrs), [])
        for trial in label_trials:
            ax.annotate(
                trial.label,
                (trial.n_params, trial.loss),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Parameter count N")
        ax.set_title(f"batch_size = {batch_size}")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

    axes[0].set_ylabel("Final training loss")
    fig.suptitle("Experiment 3_2_1: joint batch-size / learning-rate calibration")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("artifacts/experiments/ch3/3_2_1_bs_lr_joint_sweep/results.json"),
        help="Path to the archived joint sweep results JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/experiments/ch3/3_2_1_bs_lr_joint_sweep/bs_lr_calibration.png"),
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trials = load_trials(args.results)
    make_plot(trials, args.output)


if __name__ == "__main__":
    main()
