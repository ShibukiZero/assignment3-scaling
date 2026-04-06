from __future__ import annotations

import math

from cs336_scaling.chinchilla_isoflops import (
    IsoflopsRun,
    estimate_dataset_tokens,
    fit_power_law,
    parse_budget_list,
    select_optimal_runs,
)


def test_select_optimal_runs_uses_lowest_loss_per_budget() -> None:
    runs = [
        IsoflopsRun(parameters=200.0, compute_budget=1e6, final_loss=4.0),
        IsoflopsRun(parameters=100.0, compute_budget=1e6, final_loss=3.0),
        IsoflopsRun(parameters=300.0, compute_budget=2e6, final_loss=2.5),
        IsoflopsRun(parameters=150.0, compute_budget=2e6, final_loss=2.0),
    ]

    optimal_runs = select_optimal_runs(runs)

    assert [(run.compute_budget, run.parameters) for run in optimal_runs] == [
        (1e6, 100.0),
        (2e6, 150.0),
    ]


def test_select_optimal_runs_breaks_loss_ties_with_smaller_model() -> None:
    runs = [
        IsoflopsRun(parameters=250.0, compute_budget=1e6, final_loss=3.0),
        IsoflopsRun(parameters=100.0, compute_budget=1e6, final_loss=3.0),
    ]

    optimal_runs = select_optimal_runs(runs)

    assert len(optimal_runs) == 1
    assert optimal_runs[0].parameters == 100.0


def test_estimate_dataset_tokens_matches_c_equals_6nd() -> None:
    assert estimate_dataset_tokens(compute_budget=6e18, parameters=5e8) == 2e9


def test_fit_power_law_recovers_exact_log_linear_relationship() -> None:
    budgets = [1e2, 1e4, 1e6, 1e8]
    values = [3.0 * (budget ** 0.5) for budget in budgets]

    fit = fit_power_law(budgets, values)

    assert math.isclose(fit.slope, 0.5, rel_tol=1e-9)
    assert math.isclose(fit.prefactor, 3.0, rel_tol=1e-9)
    assert math.isclose(fit.r_squared, 1.0, rel_tol=1e-9)


def test_parse_budget_list_accepts_scientific_notation() -> None:
    assert parse_budget_list("1e23, 1e24") == [1e23, 1e24]
