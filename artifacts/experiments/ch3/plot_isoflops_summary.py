#!/usr/bin/env python3
"""Plot Chapter 3 IsoFLOPs experiment summaries.

This script generates two figures:

1. A Chinchilla-style IsoFLOPs profile plot showing loss versus parameter count
   for each tested compute budget.
2. A scaling-law plot for the best observed `N_opt(C)` values, with low-budget
   boundary-censored points visually distinguished from interior points.

The script merges the `1e16` budget from the hyperparameter-calibration stage
because that is where the best available exact-family losses at `1e16` were
measured after the final batch-size / learning-rate choices were fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXACT_FAMILY = [
    (256, 2, 4),
    (384, 3, 6),
    (512, 4, 8),
    (640, 5, 10),
    (768, 6, 12),
    (896, 7, 14),
    (1024, 8, 16),
]

FAMILY_LABELS = {shape: f"{shape[0]}/{shape[1]}/{shape[2]}" for shape in EXACT_FAMILY}


@dataclass(frozen=True)
class ProfilePoint:
    compute_budget: float
    n_params: int
    d_model: int
    num_layers: int
    num_heads: int
    learning_rate: float
    loss: float

    @property
    def shape_key(self) -> tuple[int, int, int]:
        return (self.d_model, self.num_layers, self.num_heads)


@dataclass(frozen=True)
class BudgetSummary:
    compute_budget: float
    best_n: int
    best_loss: float
    boundary_censored: bool
    num_points: int


def estimate_n_params(d_model: int, num_layers: int) -> int:
    return 12 * num_layers * d_model * d_model


def load_result_points(results_path: Path, compute_budget: float) -> list[ProfilePoint]:
    payload = json.loads(results_path.read_text())
    points: list[ProfilePoint] = []
    for response in payload["responses"]:
        params = response["params_without_api_key"]
        d_model = int(params["d_model"])
        num_layers = int(params["num_layers"])
        num_heads = int(params["num_heads"])
        points.append(
            ProfilePoint(
                compute_budget=compute_budget,
                n_params=estimate_n_params(d_model, num_layers),
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                learning_rate=float(params["learning_rate"]),
                loss=float(response["body_json"]["loss"]),
            )
        )
    return points


def load_best_1e16_profile(joint_results_path: Path, refine_results_path: Path) -> list[ProfilePoint]:
    """Merge 3_2_1 and 3_2_2 and keep the best available exact-family point per shape."""
    candidates = load_result_points(joint_results_path, 1e16) + load_result_points(refine_results_path, 1e16)
    best: dict[tuple[int, int, int], ProfilePoint] = {}
    for point in candidates:
        if point.shape_key not in {shape for shape in EXACT_FAMILY}:
            continue
        current = best.get(point.shape_key)
        if current is None or point.loss < current.loss:
            best[point.shape_key] = point
    points = list(best.values())
    points.sort(key=lambda point: point.n_params)
    return points


def load_all_profiles(base_dir: Path) -> dict[float, list[ProfilePoint]]:
    profiles: dict[float, list[ProfilePoint]] = {
        3e15: load_result_points(base_dir / "3_3_1_isoflops_3e15_full" / "results.json", 3e15),
        6e15: load_result_points(base_dir / "3_3_2_isoflops_6e15_full" / "results.json", 6e15),
        1e16: load_best_1e16_profile(
            base_dir / "3_2_1_bs_lr_joint_sweep" / "results.json",
            base_dir / "3_2_2_large_n_lr_refine" / "results.json",
        ),
        3e16: load_result_points(base_dir / "3_3_3_isoflops_3e16_full" / "results.json", 3e16),
        6e16: load_result_points(base_dir / "3_3_4_isoflops_6e16_full" / "results.json", 6e16),
        1e17: load_result_points(base_dir / "3_3_5_isoflops_1e17_bracket" / "results.json", 1e17),
    }
    for points in profiles.values():
        points.sort(key=lambda point: point.n_params)
    return dict(sorted(profiles.items()))


def summarize_profiles(profiles: dict[float, list[ProfilePoint]]) -> list[BudgetSummary]:
    summaries: list[BudgetSummary] = []
    min_family_n = min(estimate_n_params(shape[0], shape[1]) for shape in EXACT_FAMILY)
    for compute_budget, points in profiles.items():
        best_point = min(points, key=lambda point: point.loss)
        summaries.append(
            BudgetSummary(
                compute_budget=compute_budget,
                best_n=best_point.n_params,
                best_loss=best_point.loss,
                boundary_censored=(best_point.n_params == min_family_n),
                num_points=len(points),
            )
        )
    return summaries


def fit_scaling_law(summaries: list[BudgetSummary]) -> tuple[float, float, float]:
    """Fit `N_opt(C) = k * C^a` on non-boundary points only."""
    fit_points = [summary for summary in summaries if not summary.boundary_censored]
    x = np.log10([summary.compute_budget for summary in fit_points])
    y = np.log10([summary.best_n for summary in fit_points])
    slope, intercept = np.polyfit(x, y, deg=1)
    predictions = intercept + slope * x
    residual = y - predictions
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return intercept, slope, r_squared


def make_profiles_plot(profiles: dict[float, list[ProfilePoint]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    cmap = plt.get_cmap("plasma")
    budgets = list(profiles.keys())
    max_points = max(len(points) for points in profiles.values())

    for idx, compute_budget in enumerate(budgets):
        points = profiles[compute_budget]
        xs = [point.n_params for point in points]
        ys = [point.loss for point in points]
        color = cmap(idx / max(1, len(budgets) - 1))
        linestyle = "--" if compute_budget == 1e17 else "-"
        label = f"C={compute_budget:.0e}"
        ax.plot(xs, ys, marker="o", linewidth=2.0, linestyle=linestyle, color=color, label=label)
        best_point = min(points, key=lambda point: point.loss)
        ax.scatter([best_point.n_params], [best_point.loss], s=90, color=color, edgecolors="black", zorder=3)

    ax.set_xscale("log")
    ax.set_xlabel("Parameter count N")
    ax.set_ylabel("Final training loss")
    ax.set_title("Chapter 3 IsoFLOPs profiles")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_scaling_plot(summaries: list[BudgetSummary], output_path: Path) -> None:
    intercept, slope, r_squared = fit_scaling_law(summaries)
    fit_points = [summary for summary in summaries if not summary.boundary_censored]

    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)

    boundary = [summary for summary in summaries if summary.boundary_censored]
    interior = [summary for summary in summaries if not summary.boundary_censored]

    if boundary:
        ax.scatter(
            [summary.compute_budget for summary in boundary],
            [summary.best_n for summary in boundary],
            s=90,
            facecolors="white",
            edgecolors="#1f77b4",
            linewidths=2.0,
            label="Boundary-censored best observed points",
            zorder=3,
        )
    if interior:
        ax.scatter(
            [summary.compute_budget for summary in interior],
            [summary.best_n for summary in interior],
            s=90,
            color="#d62728",
            label="Interior / bracketed best observed points",
            zorder=3,
        )

    all_xs = np.geomspace(min(summary.compute_budget for summary in summaries), max(summary.compute_budget for summary in summaries), 300)
    fit_ys = (10 ** intercept) * (all_xs ** slope)
    ax.plot(
        all_xs,
        fit_ys,
        linestyle="--",
        linewidth=2.0,
        color="#2ca02c",
        label=f"Fit on non-boundary points: N_opt(C) = {10 ** intercept:.3e} * C^{slope:.3f}",
    )

    for summary in summaries:
        ax.annotate(
            f"{summary.best_n:.2e}",
            (summary.compute_budget, summary.best_n),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute budget C")
    ax.set_ylabel("Best observed parameter count N")
    ax.set_title(f"Observed N_opt(C) progression (fit R^2 = {r_squared:.4f})")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_summary_csv(summaries: list[BudgetSummary], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["compute_budget", "best_n", "best_loss", "boundary_censored", "num_points"])
        for summary in summaries:
            writer.writerow(
                [
                    f"{summary.compute_budget:.0f}",
                    summary.best_n,
                    summary.best_loss,
                    str(summary.boundary_censored).lower(),
                    summary.num_points,
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("artifacts/experiments/ch3"),
        help="Base directory containing Chapter 3 experiment subdirectories.",
    )
    parser.add_argument(
        "--profiles-output",
        type=Path,
        default=Path("artifacts/experiments/ch3/isoflops_profiles.png"),
        help="Output path for the IsoFLOPs profile figure.",
    )
    parser.add_argument(
        "--scaling-output",
        type=Path,
        default=Path("artifacts/experiments/ch3/nopt_scaling_law.png"),
        help="Output path for the N_opt(C) figure.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("artifacts/experiments/ch3/nopt_summary.csv"),
        help="Output path for the summarized best-observed optima table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = load_all_profiles(args.base_dir)
    summaries = summarize_profiles(profiles)
    make_profiles_plot(profiles, args.profiles_output)
    make_scaling_plot(summaries, args.scaling_output)
    write_summary_csv(summaries, args.summary_csv)


if __name__ == "__main__":
    main()
