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
        default=".agents/logs/chinchilla_isoflops",
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


def select_optimal_runs(runs: list[IsoflopsRun]) -> list[IsoflopsRun]:
    grouped_runs: dict[float, list[IsoflopsRun]] = {}
    for run in runs:
        grouped_runs.setdefault(run.compute_budget, []).append(run)

    optimal_runs: list[IsoflopsRun] = []
    for budget in sorted(grouped_runs):
        # The handout recommends taking the observed minimum-loss point directly.
        # We break ties by choosing the smaller model so the result is deterministic.
        best_run = min(grouped_runs[budget], key=lambda run: (run.final_loss, run.parameters))
        optimal_runs.append(best_run)
    return optimal_runs


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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_budgets = parse_budget_list(args.predict_budgets)
    runs = load_isoflops_runs(input_path)
    optimal_runs = select_optimal_runs(runs)

    observed_budgets = [run.compute_budget for run in optimal_runs]
    observed_parameters = [run.parameters for run in optimal_runs]
    observed_dataset_tokens = [
        estimate_dataset_tokens(compute_budget=run.compute_budget, parameters=run.parameters)
        for run in optimal_runs
    ]

    model_fit = fit_power_law(observed_budgets, observed_parameters)
    dataset_fit = fit_power_law(observed_budgets, observed_dataset_tokens)

    optimal_rows = write_optimal_points(output_dir / "optimal_points.csv", optimal_runs)
    write_summary_json(
        output_path=output_dir / "fit_summary.json",
        input_path=input_path,
        optimal_rows=optimal_rows,
        model_fit=model_fit,
        dataset_fit=dataset_fit,
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

    print("Chapter 2 IsoFLOPs fitting finished.")
    print(f"Observed budgets: {len(observed_budgets)}")
    print(f"Output directory: {output_dir}")
    print(f"Model-size fit: N_opt(C) = {model_fit.prefactor:.6e} * C^{model_fit.slope:.6f}")
    for budget in prediction_budgets:
        prediction = float(model_fit.predict(budget))
        print(f"Predicted optimal model size at C={budget:.0e}: {prediction:.6e}")
    print(f"Dataset-size fit: D_opt(C) = {dataset_fit.prefactor:.6e} * C^{dataset_fit.slope:.6f}")
    for budget in prediction_budgets:
        prediction = float(dataset_fit.predict(budget))
        print(f"Predicted optimal dataset size at C={budget:.0e}: {prediction:.6e}")


if __name__ == "__main__":
    main()
