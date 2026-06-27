#!/usr/bin/env python3
"""Compare observed-min and quadratic-profile N_opt(C) fits for Chapter 3.

This is an exploratory analysis script. It:

1. Builds Chapter 3 IsoFLOPs profiles across all tested budgets.
2. Extracts one `N_opt(C)` sequence using the observed minimum in each profile.
3. Extracts a second `N_opt(C)` sequence using the quadratic vertex in each profile.
4. Fits a log-log power law to each sequence, using different point-selection rules:
   - observed minima: fit only non-boundary points
   - quadratic-profile optima: fit all plotted budgets, but optionally drop the
     smallest budgets from the actual power-law regression
5. Writes separate scaling-law figures plus a JSON summary containing the fitted formulas.

The default output directory lives under `runs/` so we can inspect the
results before deciding what belongs in the final write-up.

The quadratic-derived analysis fits each per-budget quadratic profile using all
plotted `N` points. Archived Chapter 3 analyses may still exclude the smallest
compute budgets from the final power-law regression when they behave like
outliers.
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
class BudgetPoint:
    compute_budget: float
    observed_best_n: int
    observed_best_loss: float
    quadratic_fit_points_n: list[int]
    quadratic_vertex_n: float
    quadratic_vertex_loss: float
    opens_upward: bool
    vertex_in_observed_range: bool
    fit_domain_min_n: int
    fit_domain_max_n: int


@dataclass(frozen=True)
class PowerLawFit:
    intercept_log10: float
    slope: float
    coefficient: float
    r_squared: float
    target_flops: float
    predicted_n_at_target_flops: float

    @property
    def formula(self) -> str:
        return f"N_opt(C) = {self.coefficient:.6e} * C^{self.slope:.6f}"


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


def filter_profiles(
    profiles: dict[float, list[ProfilePoint]],
    excluded_budgets: set[float],
) -> dict[float, list[ProfilePoint]]:
    return {
        compute_budget: points
        for compute_budget, points in profiles.items()
        if compute_budget not in excluded_budgets
    }


def summarize_budget(points: list[ProfilePoint]) -> BudgetPoint:
    xs = np.array([point.n_params for point in points], dtype=float)
    ys = np.array([point.loss for point in points], dtype=float)
    fit_xs = xs
    fit_ys = ys
    log_xs = np.log10(fit_xs)
    coeffs = np.polyfit(log_xs, fit_ys, deg=2)
    a, b, c = coeffs
    vertex_log_x = -b / (2 * a) if a != 0 else float("nan")
    vertex_n = float(10**vertex_log_x) if np.isfinite(vertex_log_x) else float("nan")
    vertex_loss = float(np.polyval(coeffs, vertex_log_x)) if np.isfinite(vertex_log_x) else float("nan")

    observed_best = min(points, key=lambda point: point.loss)
    return BudgetPoint(
        compute_budget=points[0].compute_budget,
        observed_best_n=observed_best.n_params,
        observed_best_loss=observed_best.loss,
        quadratic_fit_points_n=[int(value) for value in fit_xs],
        quadratic_vertex_n=vertex_n,
        quadratic_vertex_loss=vertex_loss,
        opens_upward=bool(a > 0),
        vertex_in_observed_range=bool(fit_xs.min() <= vertex_n <= fit_xs.max()) if np.isfinite(vertex_n) else False,
        fit_domain_min_n=int(fit_xs.min()),
        fit_domain_max_n=int(fit_xs.max()),
    )


def fit_power_law(xs: list[float], ys: list[float], target_flops: float) -> PowerLawFit:
    log_x = np.log10(xs)
    log_y = np.log10(ys)
    slope, intercept = np.polyfit(log_x, log_y, deg=1)
    predicted = intercept + slope * log_x
    residual = log_y - predicted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return PowerLawFit(
        intercept_log10=float(intercept),
        slope=float(slope),
        coefficient=float(10**intercept),
        r_squared=r_squared,
        target_flops=target_flops,
        predicted_n_at_target_flops=float((10**intercept) * (target_flops ** slope)),
    )


def select_quadratic_fit_points(
    summaries: list[BudgetPoint],
    num_smallest_budgets_to_drop: int,
) -> list[BudgetPoint]:
    valid = [summary for summary in summaries if np.isfinite(summary.quadratic_vertex_n)]
    valid.sort(key=lambda summary: summary.compute_budget)
    return valid[num_smallest_budgets_to_drop:]


def make_observed_plot(summaries: list[BudgetPoint], observed_fit: PowerLawFit, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)

    boundary = [summary for summary in summaries if summary.observed_best_n == min(summary.observed_best_n for summary in summaries)]
    non_boundary = [summary for summary in summaries if summary not in boundary]

    if boundary:
        ax.scatter(
            [summary.compute_budget for summary in boundary],
            [summary.observed_best_n for summary in boundary],
            s=90,
            facecolors="white",
            edgecolors="#1f77b4",
            linewidths=2.0,
            label="Observed minima at boundary",
            zorder=3,
        )
    if non_boundary:
        ax.scatter(
            [summary.compute_budget for summary in non_boundary],
            [summary.observed_best_n for summary in non_boundary],
            s=90,
            color="#d62728",
            label="Observed minima used in fit",
            zorder=3,
        )

    fit_xs = np.geomspace(
        min(summary.compute_budget for summary in summaries),
        max(max(summary.compute_budget for summary in summaries), observed_fit.target_flops),
        300,
    )
    observed_fit_ys = observed_fit.coefficient * (fit_xs ** observed_fit.slope)
    ax.plot(
        fit_xs,
        observed_fit_ys,
        color="#2ca02c",
        linestyle="--",
        linewidth=2.0,
        label=f"Observed-min fit: {observed_fit.formula}",
    )

    for summary in summaries:
        ax.annotate(
            f"{summary.observed_best_n:.2e}",
            (summary.compute_budget, summary.observed_best_n),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    ax.scatter(
        [observed_fit.target_flops],
        [observed_fit.predicted_n_at_target_flops],
        marker="*",
        s=180,
        color="#2ca02c",
        zorder=5,
        label=f"Predicted N at {observed_fit.target_flops:.0e}",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute budget C")
    ax.set_ylabel("Best observed parameter count N")
    ax.set_title(f"Observed N_opt(C) progression (fit R^2 = {observed_fit.r_squared:.4f})")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def make_quadratic_plot(summaries: list[BudgetPoint], quadratic_fit: PowerLawFit, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)

    valid = [summary for summary in summaries if np.isfinite(summary.quadratic_vertex_n)]
    ax.scatter(
        [summary.compute_budget for summary in valid],
        [summary.quadratic_vertex_n for summary in valid],
        s=90,
        color="#d62728",
        marker="x",
        linewidths=2.0,
        label="Quadratic-profile predicted optima",
        zorder=4,
    )

    fit_xs = np.geomspace(
        min(summary.compute_budget for summary in valid),
        max(max(summary.compute_budget for summary in valid), quadratic_fit.target_flops),
        300,
    )
    quadratic_fit_ys = quadratic_fit.coefficient * (fit_xs ** quadratic_fit.slope)
    ax.plot(
        fit_xs,
        quadratic_fit_ys,
        color="#2ca02c",
        linestyle="--",
        linewidth=2.0,
        label=f"Quadratic-optimum fit: {quadratic_fit.formula}",
    )

    for summary in valid:
        ax.annotate(
            f"{summary.quadratic_vertex_n:.2e}",
            (summary.compute_budget, summary.quadratic_vertex_n),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    ax.scatter(
        [quadratic_fit.target_flops],
        [quadratic_fit.predicted_n_at_target_flops],
        marker="*",
        s=180,
        color="#2ca02c",
        zorder=5,
        label=f"Predicted N at {quadratic_fit.target_flops:.0e}",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute budget C")
    ax.set_ylabel("Quadratic-profile predicted N")
    ax.set_title(f"Quadratic-derived N_opt(C) progression (fit R^2 = {quadratic_fit.r_squared:.4f})")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_comparison_plot(
    summaries: list[BudgetPoint],
    observed_fit: PowerLawFit,
    quadratic_fit: PowerLawFit,
    output_path: Path,
) -> None:
    observed_x = np.array([summary.compute_budget for summary in summaries], dtype=float)
    observed_y = np.array([summary.observed_best_n for summary in summaries], dtype=float)
    quadratic_y = np.array([summary.quadratic_vertex_n for summary in summaries], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)

    ax.scatter(observed_x, observed_y, s=70, color="#1f77b4", label="Observed minima", zorder=3)
    ax.scatter(
        observed_x,
        quadratic_y,
        s=70,
        color="#d62728",
        marker="x",
        linewidths=2.0,
        label="Quadratic-derived optima",
        zorder=4,
    )

    fit_xs = np.geomspace(observed_x.min(), max(observed_x.max(), observed_fit.target_flops, quadratic_fit.target_flops), 300)
    ax.plot(
        fit_xs,
        observed_fit.coefficient * (fit_xs ** observed_fit.slope),
        color="#1f77b4",
        linestyle="--",
        linewidth=2.0,
        label="Observed-min fit",
    )
    ax.plot(
        fit_xs,
        quadratic_fit.coefficient * (fit_xs ** quadratic_fit.slope),
        color="#d62728",
        linestyle="-.",
        linewidth=2.0,
        label="Quadratic-optimum fit",
    )
    ax.scatter(
        [observed_fit.target_flops],
        [observed_fit.predicted_n_at_target_flops],
        marker="*",
        s=160,
        color="#1f77b4",
        zorder=5,
    )
    ax.scatter(
        [quadratic_fit.target_flops],
        [quadratic_fit.predicted_n_at_target_flops],
        marker="*",
        s=160,
        color="#d62728",
        zorder=5,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute budget C")
    ax.set_ylabel("Estimated optimal parameter count N")
    ax.set_title("Observed-min vs quadratic-derived N_opt(C)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_summary_json(
    summaries: list[BudgetPoint],
    observed_fit: PowerLawFit,
    quadratic_fit: PowerLawFit,
    quadratic_fit_points: list[BudgetPoint],
    output_path: Path,
) -> None:
    payload = {
        "note": (
            "Exploratory comparison between observed-min and quadratic-profile "
            "Chapter 3 scaling-law fits."
        ),
        "observed_min_fit": {
            "formula": observed_fit.formula,
            "coefficient": observed_fit.coefficient,
            "slope": observed_fit.slope,
            "intercept_log10": observed_fit.intercept_log10,
            "r_squared": observed_fit.r_squared,
            "target_flops": observed_fit.target_flops,
            "predicted_n_at_target_flops": observed_fit.predicted_n_at_target_flops,
            "fit_budgets": [
                summary.compute_budget
                for summary in summaries
                if summary.observed_best_n != min(item.observed_best_n for item in summaries)
            ],
            "fit_selection_rule": "Use only non-boundary observed-min points.",
        },
        "quadratic_profile_fit": {
            "formula": quadratic_fit.formula,
            "coefficient": quadratic_fit.coefficient,
            "slope": quadratic_fit.slope,
            "intercept_log10": quadratic_fit.intercept_log10,
            "r_squared": quadratic_fit.r_squared,
            "target_flops": quadratic_fit.target_flops,
            "predicted_n_at_target_flops": quadratic_fit.predicted_n_at_target_flops,
            "fit_budgets": [summary.compute_budget for summary in quadratic_fit_points],
            "fit_selection_rule": "Use quadratic-profile optima, but drop the two smallest compute budgets from the power-law regression.",
        },
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
        default=Path("runs/ch3_nopt_fit_comparison"),
        help="Directory for exploratory comparison outputs.",
    )
    parser.add_argument(
        "--target-flops",
        type=float,
        default=1e19,
        help="Target FLOPs budget to cover and annotate in the scaling plots.",
    )
    parser.add_argument(
        "--exclude-budgets",
        type=str,
        default="",
        help="Comma-separated compute budgets to exclude entirely, e.g. '3e15,6e15'.",
    )
    parser.add_argument(
        "--drop-smallest-quadratic-budgets",
        type=int,
        default=2,
        help="Number of smallest compute budgets to omit from the quadratic-derived power-law fit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = load_all_profiles(args.base_dir)
    excluded_budgets = {
        float(item.strip())
        for item in args.exclude_budgets.split(",")
        if item.strip()
    }
    profiles = filter_profiles(profiles, excluded_budgets)
    summaries = [
        summarize_budget(points)
        for _, points in profiles.items()
    ]

    boundary_n = min(summary.observed_best_n for summary in summaries)
    observed_fit = fit_power_law(
        xs=[summary.compute_budget for summary in summaries if summary.observed_best_n != boundary_n],
        ys=[summary.observed_best_n for summary in summaries if summary.observed_best_n != boundary_n],
        target_flops=args.target_flops,
    )
    quadratic_fit_points = select_quadratic_fit_points(
        summaries,
        num_smallest_budgets_to_drop=args.drop_smallest_quadratic_budgets,
    )
    quadratic_fit = fit_power_law(
        xs=[summary.compute_budget for summary in quadratic_fit_points],
        ys=[summary.quadratic_vertex_n for summary in quadratic_fit_points],
        target_flops=args.target_flops,
    )

    make_observed_plot(
        summaries=summaries,
        observed_fit=observed_fit,
        output_path=args.output_dir / "observed_nopt_scaling.png",
    )
    make_quadratic_plot(
        summaries=summaries,
        quadratic_fit=quadratic_fit,
        output_path=args.output_dir / "quadratic_nopt_scaling.png",
    )
    make_comparison_plot(
        summaries=summaries,
        observed_fit=observed_fit,
        quadratic_fit=quadratic_fit,
        output_path=args.output_dir / "nopt_fit_comparison.png",
    )
    write_summary_json(
        summaries=summaries,
        observed_fit=observed_fit,
        quadratic_fit=quadratic_fit,
        quadratic_fit_points=quadratic_fit_points,
        output_path=args.output_dir / "nopt_fit_comparison.json",
    )


if __name__ == "__main__":
    main()
