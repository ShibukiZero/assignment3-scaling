#!/usr/bin/env python3
"""Explore Chapter 3 optimal-loss fits as a function of compute budget.

This script extracts one observed best-loss value per budget and compares two
simple models for `L_opt(C)`:

1. `L(C) = a + b * log10(C)`
2. `L(C) = L_inf + A * C^{-alpha}`

For the loss law, we intentionally use the observed best loss at each queried
budget rather than the quadratic-vertex loss, because the latter is an
interpolated value rather than a directly observed training result. All budget
points remain visible in the figure, but the three smallest compute budgets are
excluded from the global regression by default, since those low-budget regimes
are still boundary-censored.
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
class BudgetLossPoint:
    compute_budget: float
    observed_best_loss: float
    observed_best_n: int
    quadratic_vertex_loss: float
    quadratic_vertex_n: float
    opens_upward: bool
    vertex_in_observed_range: bool


@dataclass(frozen=True)
class LogLinearFit:
    intercept: float
    slope_log10_c: float
    r_squared: float
    target_flops: float
    predicted_loss_at_target_flops: float

    @property
    def formula(self) -> str:
        return f"L(C) = {self.intercept:.6f} + {self.slope_log10_c:.6f} * log10(C)"


@dataclass(frozen=True)
class OffsetPowerLawFit:
    loss_floor: float
    amplitude: float
    alpha: float
    r_squared: float
    target_flops: float
    predicted_loss_at_target_flops: float

    @property
    def formula(self) -> str:
        return f"L(C) = {self.loss_floor:.6f} + {self.amplitude:.6e} * C^(-{self.alpha:.6f})"


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


def summarize_budget(points: list[ProfilePoint]) -> BudgetLossPoint:
    xs = np.array([point.n_params for point in points], dtype=float)
    ys = np.array([point.loss for point in points], dtype=float)
    coeffs = np.polyfit(np.log10(xs), ys, deg=2)
    a, b, _ = coeffs
    vertex_log_x = -b / (2 * a) if a != 0 else float("nan")
    vertex_n = float(10**vertex_log_x) if np.isfinite(vertex_log_x) else float("nan")
    vertex_loss = float(np.polyval(coeffs, vertex_log_x)) if np.isfinite(vertex_log_x) else float("nan")
    observed_best = min(points, key=lambda point: point.loss)
    return BudgetLossPoint(
        compute_budget=points[0].compute_budget,
        observed_best_loss=observed_best.loss,
        observed_best_n=observed_best.n_params,
        quadratic_vertex_loss=vertex_loss,
        quadratic_vertex_n=vertex_n,
        opens_upward=bool(a > 0),
        vertex_in_observed_range=bool(xs.min() <= vertex_n <= xs.max()) if np.isfinite(vertex_n) else False,
    )


def fit_log_linear(xs: np.ndarray, ys: np.ndarray, target_flops: float) -> LogLinearFit:
    log_x = np.log10(xs)
    slope, intercept = np.polyfit(log_x, ys, deg=1)
    pred = intercept + slope * log_x
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    target_pred = float(intercept + slope * np.log10(target_flops))
    return LogLinearFit(
        intercept=float(intercept),
        slope_log10_c=float(slope),
        r_squared=r_squared,
        target_flops=target_flops,
        predicted_loss_at_target_flops=target_pred,
    )


def fit_offset_power_law(xs: np.ndarray, ys: np.ndarray, target_flops: float) -> OffsetPowerLawFit:
    y_min = float(np.min(ys))
    floors = np.linspace(max(0.0, y_min - 1.5), y_min - 1e-4, 1200)
    log_x = np.log(xs)

    best_floor = None
    best_amp = None
    best_alpha = None
    best_sse = float("inf")

    for floor in floors:
        shifted = ys - floor
        if np.any(shifted <= 0):
            continue
        slope, intercept = np.polyfit(log_x, np.log(shifted), deg=1)
        alpha = -slope
        amp = float(np.exp(intercept))
        pred = floor + amp * (xs ** (-alpha))
        sse = float(np.sum((ys - pred) ** 2))
        if sse < best_sse:
            best_sse = sse
            best_floor = float(floor)
            best_amp = amp
            best_alpha = float(alpha)

    assert best_floor is not None and best_amp is not None and best_alpha is not None
    pred = best_floor + best_amp * (xs ** (-best_alpha))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r_squared = 1.0 - best_sse / ss_tot if ss_tot > 0 else 1.0
    target_pred = float(best_floor + best_amp * (target_flops ** (-best_alpha)))
    return OffsetPowerLawFit(
        loss_floor=best_floor,
        amplitude=best_amp,
        alpha=best_alpha,
        r_squared=r_squared,
        target_flops=target_flops,
        predicted_loss_at_target_flops=target_pred,
    )


def make_plot(
    summaries: list[BudgetLossPoint],
    fit_summaries: list[BudgetLossPoint],
    log_linear_fit: LogLinearFit,
    offset_power_law_fit: OffsetPowerLawFit,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.8), constrained_layout=True)

    xs_all = np.array([summary.compute_budget for summary in summaries], dtype=float)
    ys_all = np.array([summary.observed_best_loss for summary in summaries], dtype=float)
    xs_fit = np.array([summary.compute_budget for summary in fit_summaries], dtype=float)
    ys_fit = np.array([summary.observed_best_loss for summary in fit_summaries], dtype=float)

    excluded = {summary.compute_budget for summary in summaries} - {summary.compute_budget for summary in fit_summaries}
    excluded_points = [summary for summary in summaries if summary.compute_budget in excluded]

    ax.scatter(
        xs_all,
        ys_all,
        s=90,
        color="#1f77b4",
        label="Observed best-loss points",
        zorder=2,
    )

    if excluded_points:
        ax.scatter(
            [summary.compute_budget for summary in excluded_points],
            [summary.observed_best_loss for summary in excluded_points],
            s=80,
            facecolors="white",
            edgecolors="#1f77b4",
            linewidths=2.0,
            label="Observed best losses excluded from fit",
            zorder=3,
        )

    ax.scatter(
        xs_fit,
        ys_fit,
        s=90,
        color="#1f77b4",
        label="Observed best losses used in fit",
        zorder=4,
    )
    curve_xs = np.geomspace(xs_all.min(), xs_all.max(), 400)
    ax.plot(
        curve_xs,
        log_linear_fit.intercept + log_linear_fit.slope_log10_c * np.log10(curve_xs),
        color="#2ca02c",
        linestyle="--",
        linewidth=2.0,
        label=f"Log-linear fit: {log_linear_fit.formula}",
    )
    ax.plot(
        curve_xs,
        offset_power_law_fit.loss_floor + offset_power_law_fit.amplitude * (curve_xs ** (-offset_power_law_fit.alpha)),
        color="#9467bd",
        linestyle="-.",
        linewidth=2.0,
        label=f"Offset power-law fit: {offset_power_law_fit.formula}",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Compute budget C")
    ax.set_ylabel("Observed best loss")
    ax.set_title("Observed best-loss rule used for final loss fitting")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_summary_json(
    summaries: list[BudgetLossPoint],
    fit_summaries: list[BudgetLossPoint],
    log_linear_fit: LogLinearFit,
    offset_power_law_fit: OffsetPowerLawFit,
    output_path: Path,
) -> None:
    payload = {
        "note": (
            "Exploratory Chapter 3 optimal-loss fits. By default, the best-loss "
            "sequence is derived from observed best losses, and the three "
            "smallest compute budgets are dropped from the global fit."
        ),
        "fit_selection_rule": {
            "loss_sequence": "Use observed best losses as the primary optimal-loss proxy.",
            "global_fit": "Drop the three smallest compute budgets from the regression.",
            "plotted_points": "Keep all budgets visible in the figure.",
        },
        "log_linear_fit": asdict(log_linear_fit) | {"formula": log_linear_fit.formula},
        "offset_power_law_fit": asdict(offset_power_law_fit) | {"formula": offset_power_law_fit.formula},
        "fit_budgets": [summary.compute_budget for summary in fit_summaries],
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
        default=Path("runs/ch3_loss_fit_explore"),
        help="Directory for exploratory plot and JSON outputs.",
    )
    parser.add_argument(
        "--target-flops",
        type=float,
        default=1e19,
        help="Target FLOPs budget to annotate in the fitted curves.",
    )
    parser.add_argument(
        "--drop-smallest-budgets",
        type=int,
        default=3,
        help="Number of smallest compute budgets to omit from the global loss regressions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = load_all_profiles(args.base_dir)
    summaries = [summarize_budget(points) for _, points in profiles.items()]
    fit_summaries = sorted(summaries, key=lambda summary: summary.compute_budget)[args.drop_smallest_budgets :]

    xs_fit = np.array([summary.compute_budget for summary in fit_summaries], dtype=float)
    ys_fit = np.array([summary.observed_best_loss for summary in fit_summaries], dtype=float)

    log_linear_fit = fit_log_linear(xs_fit, ys_fit, target_flops=args.target_flops)
    offset_power_law_fit = fit_offset_power_law(xs_fit, ys_fit, target_flops=args.target_flops)

    make_plot(
        summaries=summaries,
        fit_summaries=fit_summaries,
        log_linear_fit=log_linear_fit,
        offset_power_law_fit=offset_power_law_fit,
        output_path=args.output_dir / "loss_fit_comparison.png",
    )
    write_summary_json(
        summaries=summaries,
        fit_summaries=fit_summaries,
        log_linear_fit=log_linear_fit,
        offset_power_law_fit=offset_power_law_fit,
        output_path=args.output_dir / "loss_fit_comparison.json",
    )


if __name__ == "__main__":
    main()
