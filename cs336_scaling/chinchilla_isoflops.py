from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class IsoflopsRun:
    parameters: float
    compute_budget: float
    final_loss: float


@dataclass(frozen=True)
class PowerLawFit:
    slope: float
    intercept: float
    r_squared: float

    @property
    def prefactor(self) -> float:
        return 10 ** self.intercept

    def predict(self, budgets: np.ndarray | float) -> np.ndarray:
        budget_array = np.asarray(budgets, dtype=float)
        return 10 ** (self.intercept + self.slope * np.log10(budget_array))


@dataclass(frozen=True)
class QuadraticProfileFit:
    compute_budget: float
    quadratic_a: float
    quadratic_b: float
    quadratic_c: float
    optimal_log10_parameters: float
    optimal_parameters: float
    estimated_loss: float
    used_vertex: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit Chapter 2 IsoFLOPs scaling laws from data/isoflops_curves.json and "
            "generate plots plus structured summaries."
        )
    )
    parser.add_argument(
        "--input",
        default="data/isoflops_curves.json",
        help="Path to the synthetic IsoFLOPs JSON data from the handout.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/chinchilla_isoflops",
        help=(
            "Directory where plots and JSON/CSV summaries will be written. "
            "Use an artifacts/ path later once results are confirmed."
        ),
    )
    parser.add_argument(
        "--predict-budgets",
        default="1e23,1e24",
        help="Comma-separated compute budgets to report predictions for.",
    )
    parser.add_argument(
        "--plot-max-budget",
        type=float,
        default=1e24,
        help="Largest compute budget to show on the extrapolation plots.",
    )
    return parser


def load_isoflops_runs(path: str | Path) -> list[IsoflopsRun]:
    raw_runs = json.loads(Path(path).read_text())
    runs: list[IsoflopsRun] = []
    for raw_run in raw_runs:
        compute_budget = raw_run.get("compute_budget", raw_run.get("compute-budget"))
        if compute_budget is None:
            raise ValueError("Each run must contain either 'compute_budget' or 'compute-budget'.")
        runs.append(
            IsoflopsRun(
                parameters=float(raw_run["parameters"]),
                compute_budget=float(compute_budget),
                final_loss=float(raw_run["final_loss"]),
            )
        )
    return runs


def parse_budget_list(value: str) -> list[float]:
    budgets = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not budgets:
        raise ValueError("Expected at least one prediction budget.")
    return budgets


def group_runs_by_budget(runs: list[IsoflopsRun]) -> dict[float, list[IsoflopsRun]]:
    grouped_runs: dict[float, list[IsoflopsRun]] = {}
    for run in runs:
        grouped_runs.setdefault(run.compute_budget, []).append(run)
    return grouped_runs


def select_optimal_runs(runs: list[IsoflopsRun]) -> list[IsoflopsRun]:
    grouped_runs = group_runs_by_budget(runs)
    optimal_runs: list[IsoflopsRun] = []
    for budget in sorted(grouped_runs):
        # The handout recommends taking the observed minimum-loss point directly.
        # We break ties by choosing the smaller model so the result is deterministic.
        best_run = min(grouped_runs[budget], key=lambda run: (run.final_loss, run.parameters))
        optimal_runs.append(best_run)
    return optimal_runs


def fit_quadratic_profile_optima(runs: list[IsoflopsRun]) -> list[QuadraticProfileFit]:
    grouped_runs = group_runs_by_budget(runs)
    profile_fits: list[QuadraticProfileFit] = []

    for budget in sorted(grouped_runs):
        budget_runs = sorted(grouped_runs[budget], key=lambda run: run.parameters)
        if len(budget_runs) < 3:
            raise ValueError("Need at least three runs per compute budget for quadratic fitting.")

        log_parameters = np.log10(np.asarray([run.parameters for run in budget_runs], dtype=float))
        losses = np.asarray([run.final_loss for run in budget_runs], dtype=float)

        quadratic_a, quadratic_b, quadratic_c = np.polyfit(log_parameters, losses, deg=2)
        observed_best_run = min(budget_runs, key=lambda run: (run.final_loss, run.parameters))
        observed_min_log_parameter = float(np.min(log_parameters))
        observed_max_log_parameter = float(np.max(log_parameters))

        used_vertex = False
        if quadratic_a > 0:
            candidate_log_parameter = float(-quadratic_b / (2.0 * quadratic_a))
            if observed_min_log_parameter <= candidate_log_parameter <= observed_max_log_parameter:
                optimal_log_parameter = candidate_log_parameter
                used_vertex = True
            else:
                optimal_log_parameter = float(np.log10(observed_best_run.parameters))
        else:
            optimal_log_parameter = float(np.log10(observed_best_run.parameters))

        optimal_parameters = float(10 ** optimal_log_parameter)
        estimated_loss = float(
            quadratic_a * optimal_log_parameter**2
            + quadratic_b * optimal_log_parameter
            + quadratic_c
        )
        if not used_vertex:
            estimated_loss = float(observed_best_run.final_loss)

        profile_fits.append(
            QuadraticProfileFit(
                compute_budget=budget,
                quadratic_a=float(quadratic_a),
                quadratic_b=float(quadratic_b),
                quadratic_c=float(quadratic_c),
                optimal_log10_parameters=optimal_log_parameter,
                optimal_parameters=optimal_parameters,
                estimated_loss=estimated_loss,
                used_vertex=used_vertex,
            )
        )

    return profile_fits


def estimate_dataset_tokens(*, compute_budget: float, parameters: float) -> float:
    return compute_budget / (6.0 * parameters)


def fit_power_law(x_values: list[float], y_values: list[float]) -> PowerLawFit:
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length.")
    if len(x_values) < 2:
        raise ValueError("Need at least two points to fit a power law.")

    log_x = np.log10(np.asarray(x_values, dtype=float))
    log_y = np.log10(np.asarray(y_values, dtype=float))

    slope, intercept = np.polyfit(log_x, log_y, deg=1)
    predicted_log_y = intercept + slope * log_x

    residual_sum_squares = float(np.sum((log_y - predicted_log_y) ** 2))
    total_sum_squares = float(np.sum((log_y - np.mean(log_y)) ** 2))
    if total_sum_squares == 0.0:
        r_squared = 1.0
    else:
        r_squared = 1.0 - residual_sum_squares / total_sum_squares

    return PowerLawFit(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=r_squared,
    )


def write_optimal_points(
    output_path: Path,
    optimal_runs: list[IsoflopsRun],
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for run in optimal_runs:
        rows.append(
            {
                "compute_budget": run.compute_budget,
                "optimal_parameters": run.parameters,
                "optimal_dataset_tokens": estimate_dataset_tokens(
                    compute_budget=run.compute_budget,
                    parameters=run.parameters,
                ),
                "final_loss": run.final_loss,
            }
        )

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "compute_budget",
                "optimal_parameters",
                "optimal_dataset_tokens",
                "final_loss",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


def write_quadratic_optimal_points(
    output_path: Path,
    profile_fits: list[QuadraticProfileFit],
) -> list[dict[str, float | bool]]:
    rows: list[dict[str, float | bool]] = []
    for fit in profile_fits:
        rows.append(
            {
                "compute_budget": fit.compute_budget,
                "optimal_parameters": fit.optimal_parameters,
                "optimal_dataset_tokens": estimate_dataset_tokens(
                    compute_budget=fit.compute_budget,
                    parameters=fit.optimal_parameters,
                ),
                "estimated_loss": fit.estimated_loss,
                "used_vertex": fit.used_vertex,
                "quadratic_a": fit.quadratic_a,
                "quadratic_b": fit.quadratic_b,
                "quadratic_c": fit.quadratic_c,
            }
        )

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "compute_budget",
                "optimal_parameters",
                "optimal_dataset_tokens",
                "estimated_loss",
                "used_vertex",
                "quadratic_a",
                "quadratic_b",
                "quadratic_c",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


def make_plot(
    *,
    output_path: Path,
    observed_budgets: list[float],
    observed_values: list[float],
    fit: PowerLawFit,
    predicted_budgets: list[float],
    plot_max_budget: float,
    ylabel: str,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    max_budget = max(max(observed_budgets), max(predicted_budgets), plot_max_budget)
    fit_budgets = np.geomspace(min(observed_budgets), max_budget, num=256)
    fit_values = fit.predict(fit_budgets)
    predicted_values = fit.predict(np.asarray(predicted_budgets, dtype=float))

    figure, axis = plt.subplots(figsize=(7.0, 4.8))
    axis.scatter(observed_budgets, observed_values, color="tab:blue", label="Observed optima", zorder=3)
    axis.plot(fit_budgets, fit_values, color="tab:orange", label="Power-law fit", linewidth=2.0)
    axis.scatter(
        predicted_budgets,
        predicted_values,
        color="tab:red",
        marker="x",
        s=70,
        linewidths=2.0,
        label="Reported predictions",
        zorder=4,
    )

    for budget, value in zip(predicted_budgets, predicted_values, strict=True):
        axis.annotate(
            f"C={budget:.0e}\nY={value:.3e}",
            xy=(budget, value),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Compute budget (FLOPs)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def make_comparison_plot(
    *,
    output_path: Path,
    observed_budgets: list[float],
    observed_values: list[float],
    observed_fit: PowerLawFit,
    quadratic_budgets: list[float],
    quadratic_values: list[float],
    quadratic_fit: PowerLawFit,
    predicted_budgets: list[float],
    plot_max_budget: float,
    ylabel: str,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    max_budget = max(
        max(observed_budgets),
        max(quadratic_budgets),
        max(predicted_budgets),
        plot_max_budget,
    )
    fit_budgets = np.geomspace(min(observed_budgets), max_budget, num=256)

    observed_fit_values = observed_fit.predict(fit_budgets)
    quadratic_fit_values = quadratic_fit.predict(fit_budgets)
    observed_predicted_values = observed_fit.predict(np.asarray(predicted_budgets, dtype=float))
    quadratic_predicted_values = quadratic_fit.predict(np.asarray(predicted_budgets, dtype=float))

    figure, axis = plt.subplots(figsize=(7.4, 5.0))
    axis.scatter(
        observed_budgets,
        observed_values,
        color="tab:blue",
        label="Observed minima",
        zorder=3,
    )
    axis.plot(
        fit_budgets,
        observed_fit_values,
        color="tab:blue",
        linewidth=2.0,
        label="Observed-min power-law fit",
    )
    axis.scatter(
        quadratic_budgets,
        quadratic_values,
        color="tab:orange",
        marker="s",
        label="Quadratic-profile optima",
        zorder=3,
    )
    axis.plot(
        fit_budgets,
        quadratic_fit_values,
        color="tab:orange",
        linewidth=2.0,
        label="Quadratic-profile power-law fit",
    )
    axis.scatter(
        predicted_budgets,
        observed_predicted_values,
        color="tab:blue",
        marker="x",
        s=70,
        linewidths=2.0,
        label="Observed-min predictions",
        zorder=4,
    )
    axis.scatter(
        predicted_budgets,
        quadratic_predicted_values,
        color="tab:orange",
        marker="D",
        s=42,
        linewidths=1.5,
        label="Quadratic-profile predictions",
        zorder=4,
    )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Compute budget (FLOPs)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def write_summary_json(
    *,
    output_path: Path,
    input_path: str | Path,
    optimal_rows: list[dict[str, float]],
    model_fit: PowerLawFit,
    dataset_fit: PowerLawFit,
    prediction_budgets: list[float],
) -> None:
    model_predictions = [
        {
            "compute_budget": budget,
            "predicted_optimal_parameters": float(model_fit.predict(budget)),
        }
        for budget in prediction_budgets
    ]
    dataset_predictions = [
        {
            "compute_budget": budget,
            "predicted_optimal_dataset_tokens": float(dataset_fit.predict(budget)),
        }
        for budget in prediction_budgets
    ]

    summary: dict[str, Any] = {
        "input_path": str(input_path),
        "optimal_points": optimal_rows,
        "model_size_fit": {
            **asdict(model_fit),
            "prefactor": model_fit.prefactor,
            "equation": f"N_opt(C) = {model_fit.prefactor:.6e} * C^{model_fit.slope:.6f}",
        },
        "dataset_size_fit": {
            **asdict(dataset_fit),
            "prefactor": dataset_fit.prefactor,
            "equation": f"D_opt(C) = {dataset_fit.prefactor:.6e} * C^{dataset_fit.slope:.6f}",
        },
        "model_predictions": model_predictions,
        "dataset_predictions": dataset_predictions,
    }
    output_path.write_text(json.dumps(summary, indent=2))


def write_comparison_summary_json(
    *,
    output_path: Path,
    input_path: str | Path,
    observed_rows: list[dict[str, float]],
    quadratic_rows: list[dict[str, float | bool]],
    observed_model_fit: PowerLawFit,
    observed_dataset_fit: PowerLawFit,
    quadratic_model_fit: PowerLawFit,
    quadratic_dataset_fit: PowerLawFit,
    prediction_budgets: list[float],
) -> None:
    summary: dict[str, Any] = {
        "input_path": str(input_path),
        "prediction_budgets": prediction_budgets,
        "observed_minimum_method": {
            "optimal_points": observed_rows,
            "model_size_fit": {
                **asdict(observed_model_fit),
                "prefactor": observed_model_fit.prefactor,
                "equation": (
                    f"N_opt(C) = {observed_model_fit.prefactor:.6e} * C^{observed_model_fit.slope:.6f}"
                ),
            },
            "dataset_size_fit": {
                **asdict(observed_dataset_fit),
                "prefactor": observed_dataset_fit.prefactor,
                "equation": (
                    f"D_opt(C) = {observed_dataset_fit.prefactor:.6e} * C^{observed_dataset_fit.slope:.6f}"
                ),
            },
        },
        "quadratic_profile_method": {
            "optimal_points": quadratic_rows,
            "model_size_fit": {
                **asdict(quadratic_model_fit),
                "prefactor": quadratic_model_fit.prefactor,
                "equation": (
                    f"N_opt(C) = {quadratic_model_fit.prefactor:.6e} * C^{quadratic_model_fit.slope:.6f}"
                ),
            },
            "dataset_size_fit": {
                **asdict(quadratic_dataset_fit),
                "prefactor": quadratic_dataset_fit.prefactor,
                "equation": (
                    f"D_opt(C) = {quadratic_dataset_fit.prefactor:.6e} * C^{quadratic_dataset_fit.slope:.6f}"
                ),
            },
        },
    }
    output_path.write_text(json.dumps(summary, indent=2))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_budgets = parse_budget_list(args.predict_budgets)
    runs = load_isoflops_runs(input_path)
    optimal_runs = select_optimal_runs(runs)
    quadratic_profile_fits = fit_quadratic_profile_optima(runs)

    observed_budgets = [run.compute_budget for run in optimal_runs]
    observed_parameters = [run.parameters for run in optimal_runs]
    observed_dataset_tokens = [
        estimate_dataset_tokens(compute_budget=run.compute_budget, parameters=run.parameters)
        for run in optimal_runs
    ]
    quadratic_budgets = [fit.compute_budget for fit in quadratic_profile_fits]
    quadratic_parameters = [fit.optimal_parameters for fit in quadratic_profile_fits]
    quadratic_dataset_tokens = [
        estimate_dataset_tokens(compute_budget=fit.compute_budget, parameters=fit.optimal_parameters)
        for fit in quadratic_profile_fits
    ]

    model_fit = fit_power_law(observed_budgets, observed_parameters)
    dataset_fit = fit_power_law(observed_budgets, observed_dataset_tokens)
    quadratic_model_fit = fit_power_law(quadratic_budgets, quadratic_parameters)
    quadratic_dataset_fit = fit_power_law(quadratic_budgets, quadratic_dataset_tokens)

    optimal_rows = write_optimal_points(output_dir / "optimal_points.csv", optimal_runs)
    quadratic_rows = write_quadratic_optimal_points(
        output_dir / "quadratic_optimal_points.csv",
        quadratic_profile_fits,
    )
    write_summary_json(
        output_path=output_dir / "fit_summary.json",
        input_path=input_path,
        optimal_rows=optimal_rows,
        model_fit=model_fit,
        dataset_fit=dataset_fit,
        prediction_budgets=prediction_budgets,
    )
    write_comparison_summary_json(
        output_path=output_dir / "fit_comparison_summary.json",
        input_path=input_path,
        observed_rows=optimal_rows,
        quadratic_rows=quadratic_rows,
        observed_model_fit=model_fit,
        observed_dataset_fit=dataset_fit,
        quadratic_model_fit=quadratic_model_fit,
        quadratic_dataset_fit=quadratic_dataset_fit,
        prediction_budgets=prediction_budgets,
    )

    make_plot(
        output_path=output_dir / "model_size_scaling_law.png",
        observed_budgets=observed_budgets,
        observed_values=observed_parameters,
        fit=model_fit,
        predicted_budgets=prediction_budgets,
        plot_max_budget=args.plot_max_budget,
        ylabel="Optimal model size (parameters)",
        title="IsoFLOPs scaling law for model size",
    )
    make_plot(
        output_path=output_dir / "dataset_size_scaling_law.png",
        observed_budgets=observed_budgets,
        observed_values=observed_dataset_tokens,
        fit=dataset_fit,
        predicted_budgets=prediction_budgets,
        plot_max_budget=args.plot_max_budget,
        ylabel="Optimal dataset size (tokens)",
        title="IsoFLOPs scaling law for dataset size",
    )
    make_comparison_plot(
        output_path=output_dir / "model_size_scaling_law_comparison.png",
        observed_budgets=observed_budgets,
        observed_values=observed_parameters,
        observed_fit=model_fit,
        quadratic_budgets=quadratic_budgets,
        quadratic_values=quadratic_parameters,
        quadratic_fit=quadratic_model_fit,
        predicted_budgets=prediction_budgets,
        plot_max_budget=args.plot_max_budget,
        ylabel="Optimal model size (parameters)",
        title="IsoFLOPs model-size comparison: observed minima vs quadratic profiles",
    )
    make_comparison_plot(
        output_path=output_dir / "dataset_size_scaling_law_comparison.png",
        observed_budgets=observed_budgets,
        observed_values=observed_dataset_tokens,
        observed_fit=dataset_fit,
        quadratic_budgets=quadratic_budgets,
        quadratic_values=quadratic_dataset_tokens,
        quadratic_fit=quadratic_dataset_fit,
        predicted_budgets=prediction_budgets,
        plot_max_budget=args.plot_max_budget,
        ylabel="Optimal dataset size (tokens)",
        title="IsoFLOPs dataset-size comparison: observed minima vs quadratic profiles",
    )

    print("Chapter 2 IsoFLOPs fitting finished.")
    print(f"Observed budgets: {len(observed_budgets)}")
    print(f"Output directory: {output_dir}")
    print(f"Model-size fit: N_opt(C) = {model_fit.prefactor:.6e} * C^{model_fit.slope:.6f}")
    print(
        "Quadratic-profile model-size fit: "
        f"N_opt(C) = {quadratic_model_fit.prefactor:.6e} * C^{quadratic_model_fit.slope:.6f}"
    )
    for budget in prediction_budgets:
        prediction = float(model_fit.predict(budget))
        print(f"Predicted optimal model size at C={budget:.0e}: {prediction:.6e}")
        quadratic_prediction = float(quadratic_model_fit.predict(budget))
        print(
            "Quadratic-profile predicted optimal model size at "
            f"C={budget:.0e}: {quadratic_prediction:.6e}"
        )
    print(f"Dataset-size fit: D_opt(C) = {dataset_fit.prefactor:.6e} * C^{dataset_fit.slope:.6f}")
    print(
        "Quadratic-profile dataset-size fit: "
        f"D_opt(C) = {quadratic_dataset_fit.prefactor:.6e} * C^{quadratic_dataset_fit.slope:.6f}"
    )
    for budget in prediction_budgets:
        prediction = float(dataset_fit.predict(budget))
        print(f"Predicted optimal dataset size at C={budget:.0e}: {prediction:.6e}")
        quadratic_prediction = float(quadratic_dataset_fit.predict(budget))
        print(
            "Quadratic-profile predicted optimal dataset size at "
            f"C={budget:.0e}: {quadratic_prediction:.6e}"
        )


if __name__ == "__main__":
    main()
