#!/usr/bin/env python3
"""Exploratory quadratic fits for Chapter 3 IsoFLOPs profiles.

This script is intentionally exploratory. It fits a quadratic to every tested
compute budget, including the low-budget regimes that may still be boundary-
censored, so that we can inspect how reasonable or unreasonable the fitted
valleys look before deciding whether any of them belong in the final write-up.

Outputs:
- `isoflops_quadratic_profiles.png`
- `isoflops_quadratic_summary.json`

The default output directory is under `.agents/logs/`, not `artifacts/`, so the
results can be reviewed first before we decide whether to promote them into the
final report.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
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
class QuadraticFitSummary:
    compute_budget: float
    num_points: int
    observed_best_n: int
    observed_best_loss: float
    quadratic_vertex_n: float
    quadratic_vertex_loss: float
    opens_upward: bool
    vertex_in_observed_range: bool
    fit_domain_min_n: int
    fit_domain_max_n: int


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
    points.sort(key=lambda point: point.n_params)
    return points


def load_best_1e16_profile(joint_results_path: Path, refine_results_path: Path) -> list[ProfilePoint]:
    candidates = load_result_points(joint_results_path, 1e16) + load_result_points(refine_results_path, 1e16)
    best: dict[tuple[int, int, int], ProfilePoint] = {}
    exact_shapes = set(EXACT_FAMILY)
    for point in candidates:
        if point.shape_key not in exact_shapes:
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
    return dict(sorted(profiles.items()))


def fit_profile(points: list[ProfilePoint]) -> tuple[np.ndarray, QuadraticFitSummary]:
    xs = np.array([point.n_params for point in points], dtype=float)
    ys = np.array([point.loss for point in points], dtype=float)
    log_xs = np.log10(xs)
    coeffs = np.polyfit(log_xs, ys, deg=2)
    a, b, c = coeffs
    vertex_log_x = -b / (2 * a) if a != 0 else float("nan")
    vertex_n = float(10**vertex_log_x) if np.isfinite(vertex_log_x) else float("nan")
    vertex_loss = float(np.polyval(coeffs, vertex_log_x)) if np.isfinite(vertex_log_x) else float("nan")

    observed_best = min(points, key=lambda point: point.loss)
    summary = QuadraticFitSummary(
        compute_budget=points[0].compute_budget,
        num_points=len(points),
        observed_best_n=observed_best.n_params,
        observed_best_loss=observed_best.loss,
        quadratic_vertex_n=vertex_n,
        quadratic_vertex_loss=vertex_loss,
        opens_upward=bool(a > 0),
        vertex_in_observed_range=bool(xs.min() <= vertex_n <= xs.max()) if np.isfinite(vertex_n) else False,
        fit_domain_min_n=int(xs.min()),
        fit_domain_max_n=int(xs.max()),
    )
    return coeffs, summary


def make_plot(
    profiles: dict[float, list[ProfilePoint]],
    fit_results: dict[float, tuple[np.ndarray, QuadraticFitSummary]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6), constrained_layout=True)
    cmap = plt.get_cmap("plasma")
    budgets = list(profiles.keys())

    for idx, compute_budget in enumerate(budgets):
        color = cmap(idx / max(1, len(budgets) - 1))
        points = profiles[compute_budget]
        coeffs, summary = fit_results[compute_budget]

        xs = np.array([point.n_params for point in points], dtype=float)
        ys = np.array([point.loss for point in points], dtype=float)
        ax.scatter(xs, ys, color=color, s=55, label=f"C={compute_budget:.0e}")

        fit_min_x = float(xs.min())
        fit_max_x = float(xs.max())
        if np.isfinite(summary.quadratic_vertex_n):
            fit_min_x = min(fit_min_x, summary.quadratic_vertex_n)
            fit_max_x = max(fit_max_x, summary.quadratic_vertex_n)
        fit_xs = np.geomspace(fit_min_x, fit_max_x, 300)
        fit_ys = np.polyval(coeffs, np.log10(fit_xs))
        ax.plot(fit_xs, fit_ys, color=color, linestyle="--", linewidth=1.8)

        ax.scatter(
            [summary.observed_best_n],
            [summary.observed_best_loss],
            color=color,
            edgecolors="black",
            s=100,
            zorder=4,
        )
        if np.isfinite(summary.quadratic_vertex_n):
            ax.scatter(
                [summary.quadratic_vertex_n],
                [summary.quadratic_vertex_loss],
                marker="x",
                color=color,
                s=90,
                linewidths=2.0,
                zorder=5,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Parameter count N")
    ax.set_ylabel("Final training loss")
    ax.set_title("Exploratory quadratic fits for Chapter 3 IsoFLOPs profiles")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_summary_json(summaries: list[QuadraticFitSummary], output_path: Path) -> None:
    payload = {
        "note": (
            "Exploratory quadratic fits across all tested Chapter 3 compute budgets. "
            "These fits are not necessarily valid as final decision rules; they are "
            "intended for visual inspection only."
        ),
        "budgets": [asdict(summary) for summary in summaries],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("artifacts/experiments/ch3"),
        help="Base directory containing Chapter 3 experiment subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".agents/logs/ch3_isoflops_quadratic_explore"),
        help="Directory for exploratory plot and JSON outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = load_all_profiles(args.base_dir)
    fit_results = {budget: fit_profile(points) for budget, points in profiles.items()}
    summaries = [fit_results[budget][1] for budget in profiles]
    make_plot(profiles, fit_results, args.output_dir / "isoflops_quadratic_profiles.png")
    write_summary_json(summaries, args.output_dir / "isoflops_quadratic_summary.json")


if __name__ == "__main__":
    main()
