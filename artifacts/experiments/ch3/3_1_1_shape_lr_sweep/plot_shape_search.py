#!/usr/bin/env python3
"""Plot the fixed-N shape search used to choose the Chapter 3 model family.

This script reads the archived `results.json` from experiment `3_1_1_shape_lr_sweep`
and produces a figure with two panels:

1. Loss versus width/depth ratio for every tested learning rate.
2. Best loss per shape after the local LR sweep, with the winner and top-3 shapes highlighted.

The goal is to document that the initial search was a fixed-parameter-scale shape
comparison rather than a full IsoFLOPs sweep.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ShapeTrial:
    shape_id: str
    d_model: int
    num_layers: int
    num_heads: int
    learning_rate: float
    loss: float

    @property
    def width_depth_ratio(self) -> float:
        return self.d_model / self.num_layers

    @property
    def head_dim(self) -> float:
        return self.d_model / self.num_heads

    @property
    def label(self) -> str:
        return f"{self.d_model}/{self.num_layers}/{self.num_heads}"


def load_trials(results_path: Path) -> list[ShapeTrial]:
    payload = json.loads(results_path.read_text())
    trials: list[ShapeTrial] = []
    for response in payload["responses"]:
        params = response["params_without_api_key"]
        trials.append(
            ShapeTrial(
                shape_id=response["axis_choices"]["shape"],
                d_model=int(params["d_model"]),
                num_layers=int(params["num_layers"]),
                num_heads=int(params["num_heads"]),
                learning_rate=float(params["learning_rate"]),
                loss=float(response["body_json"]["loss"]),
            )
        )
    return trials


def group_by_lr(trials: list[ShapeTrial]) -> dict[float, list[ShapeTrial]]:
    grouped: dict[float, list[ShapeTrial]] = {}
    for trial in trials:
        grouped.setdefault(trial.learning_rate, []).append(trial)
    for lr_trials in grouped.values():
        lr_trials.sort(key=lambda trial: trial.width_depth_ratio)
    return dict(sorted(grouped.items()))


def best_per_shape(trials: list[ShapeTrial]) -> list[ShapeTrial]:
    best: dict[str, ShapeTrial] = {}
    for trial in trials:
        current = best.get(trial.shape_id)
        if current is None or trial.loss < current.loss:
            best[trial.shape_id] = trial
    winners = list(best.values())
    winners.sort(key=lambda trial: trial.width_depth_ratio)
    return winners


def make_plot(trials: list[ShapeTrial], output_path: Path) -> None:
    lr_groups = group_by_lr(trials)
    winners = best_per_shape(trials)
    winning_trial = min(winners, key=lambda trial: trial.loss)
    top_three = sorted(winners, key=lambda trial: trial.loss)[:3]
    top_three_ids = {trial.shape_id for trial in top_three}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Panel 1: every LR curve on the same width/depth-ratio axis.
    ax = axes[0]
    cmap = plt.get_cmap("viridis")
    lr_values = list(lr_groups.keys())
    for idx, lr in enumerate(lr_values):
        xs = [trial.width_depth_ratio for trial in lr_groups[lr]]
        ys = [trial.loss for trial in lr_groups[lr]]
        color = cmap(idx / max(1, len(lr_values) - 1))
        ax.plot(xs, ys, marker="o", linewidth=1.8, color=color, label=f"lr={lr:g}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Width/depth ratio (d_model / num_layers)")
    ax.set_ylabel("Final training loss")
    ax.set_title("All LR sweeps at fixed N and fixed train_flops")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    # Panel 2: best loss per shape after LR sweep.
    ax = axes[1]
    xs = [trial.width_depth_ratio for trial in winners]
    ys = [trial.loss for trial in winners]
    ax.plot(xs, ys, marker="o", linewidth=2.0, color="#1f77b4")
    for trial in winners:
        if trial.shape_id == winning_trial.shape_id:
            color = "#d62728"
            size = 140
            zorder = 4
        elif trial.shape_id in top_three_ids:
            color = "#ff7f0e"
            size = 100
            zorder = 3
        else:
            color = "#1f77b4"
            size = 60
            zorder = 2
        ax.scatter(
            [trial.width_depth_ratio],
            [trial.loss],
            s=size,
            color=color,
            zorder=zorder,
        )
        ax.annotate(
            trial.label,
            (trial.width_depth_ratio, trial.loss),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    ax.axvline(
        winning_trial.width_depth_ratio,
        linestyle=":",
        linewidth=1.8,
        color="#d62728",
        label=f"Winner ratio={winning_trial.width_depth_ratio:.0f}",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Width/depth ratio (d_model / num_layers)")
    ax.set_ylabel("Best loss after LR sweep")
    ax.set_title("Best observed loss by candidate shape")
    ax.grid(True, alpha=0.25)
    ax.scatter([], [], s=140, color="#d62728", label="Selected anchor winner")
    ax.scatter([], [], s=100, color="#ff7f0e", label="Other top-3 shapes")
    ax.legend(frameon=False)

    fig.suptitle(
        "Experiment 3_1_1: fixed-N shape search near N≈2.5e7 "
        "(batch_size=128, train_flops=1e16, head_dim=64)"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("artifacts/experiments/ch3/3_1_1_shape_lr_sweep/results.json"),
        help="Path to the archived shape-search results JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/experiments/ch3/3_1_1_shape_lr_sweep/shape_search_profile.png"),
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trials = load_trials(args.results)
    make_plot(trials, args.output)


if __name__ == "__main__":
    main()
