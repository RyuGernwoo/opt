from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

from .instance import WirelessConfig, generate_instance
from .models import AllocationResult
from .plots import write_line_plot
from .solvers import (
    solve_entropy_relaxation,
    solve_greedy_cost,
    solve_hybrid_score_greedy,
    solve_hungarian,
    solve_lp_relaxation,
)


@dataclass(frozen=True)
class ExperimentSettings:
    output_dir: Path = Path("results")
    seeds: int = 5
    user_sweep: tuple[int, ...] = (10, 15, 20, 30, 50, 80)
    rb_sweep: tuple[int, ...] = (5, 9, 12, 15, 20)
    fixed_users: int = 15
    fixed_rbs: int = 12
    entropy_taus: tuple[float, ...] = (0.1, 0.3, 0.8)


@dataclass(frozen=True)
class ExperimentPaths:
    raw_csv: Path
    summary_csv: Path
    objective_plot: Path
    runtime_plot: Path
    convergence_plot: Path


RAW_FIELDS = [
    "scenario",
    "x_value",
    "seed",
    "users",
    "rbs",
    "method",
    "objective",
    "objective_gap_pct",
    "selected_users",
    "expected_successful_samples",
    "mean_packet_error",
    "convergence_gap",
    "fl_quality",
    "runtime_seconds",
]

SUMMARY_FIELDS = [
    "scenario",
    "x_value",
    "method",
    "runs",
    "objective_mean",
    "objective_gap_pct_mean",
    "selected_users_mean",
    "expected_successful_samples_mean",
    "mean_packet_error_mean",
    "convergence_gap_mean",
    "fl_quality_mean",
    "runtime_seconds_mean",
    "runtime_seconds_std",
]


def _run_methods(instance, entropy_taus: tuple[float, ...]) -> list[AllocationResult]:
    results = [
        solve_hungarian(instance),
        solve_greedy_cost(instance),
        solve_lp_relaxation(instance),
    ]
    for tau in entropy_taus:
        results.append(solve_entropy_relaxation(instance, tau=tau, iterations=250, rounding="greedy"))
        results.append(solve_entropy_relaxation(instance, tau=tau, iterations=250, rounding="projection"))
    results.append(solve_hybrid_score_greedy(instance, alpha=0.25, tau=0.3, iterations=120))
    return results


def _raw_rows_for_instance(
    scenario: str,
    x_value: int,
    instance,
    entropy_taus: tuple[float, ...],
) -> list[dict[str, str]]:
    results = _run_methods(instance, entropy_taus)
    hungarian = next(result for result in results if result.method == "Hungarian")
    rows: list[dict[str, str]] = []
    for result in results:
        gap_pct = 100.0 * (result.objective - hungarian.objective) / max(abs(hungarian.objective), 1e-12)
        rows.append(
            {
                "scenario": scenario,
                "x_value": str(x_value),
                "seed": str(instance.seed),
                "users": str(instance.user_count),
                "rbs": str(instance.rb_count),
                "method": result.method,
                "objective": f"{result.objective:.10f}",
                "objective_gap_pct": f"{gap_pct:.10f}",
                "selected_users": str(result.selected_users),
                "expected_successful_samples": f"{result.expected_successful_samples:.10f}",
                "mean_packet_error": f"{result.mean_packet_error:.10f}",
                "convergence_gap": f"{result.convergence_gap:.10f}",
                "fl_quality": f"{result.fl_quality:.10f}",
                "runtime_seconds": f"{result.runtime_seconds:.10f}",
            }
        )
    return rows


def _summarize(raw_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(row["scenario"], row["x_value"], row["method"])].append(row)

    summary: list[dict[str, str]] = []
    numeric_fields = [
        "objective",
        "objective_gap_pct",
        "selected_users",
        "expected_successful_samples",
        "mean_packet_error",
        "convergence_gap",
        "fl_quality",
        "runtime_seconds",
    ]
    for (scenario, x_value, method), rows in sorted(grouped.items(), key=lambda item: (item[0][0], float(item[0][1]), item[0][2])):
        values = {field: [float(row[field]) for row in rows] for field in numeric_fields}
        summary.append(
            {
                "scenario": scenario,
                "x_value": x_value,
                "method": method,
                "runs": str(len(rows)),
                "objective_mean": f"{mean(values['objective']):.10f}",
                "objective_gap_pct_mean": f"{mean(values['objective_gap_pct']):.10f}",
                "selected_users_mean": f"{mean(values['selected_users']):.10f}",
                "expected_successful_samples_mean": f"{mean(values['expected_successful_samples']):.10f}",
                "mean_packet_error_mean": f"{mean(values['mean_packet_error']):.10f}",
                "convergence_gap_mean": f"{mean(values['convergence_gap']):.10f}",
                "fl_quality_mean": f"{mean(values['fl_quality']):.10f}",
                "runtime_seconds_mean": f"{mean(values['runtime_seconds']):.10f}",
                "runtime_seconds_std": f"{pstdev(values['runtime_seconds']):.10f}",
            }
        )
    return summary


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment_suite(settings: ExperimentSettings) -> ExperimentPaths:
    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, str]] = []

    for user_count in settings.user_sweep:
        for seed in range(settings.seeds):
            instance = generate_instance(
                WirelessConfig(user_count=user_count, rb_count=settings.fixed_rbs, seed=10_000 + user_count * 101 + seed)
            )
            raw_rows.extend(_raw_rows_for_instance("users", user_count, instance, settings.entropy_taus))

    for rb_count in settings.rb_sweep:
        for seed in range(settings.seeds):
            instance = generate_instance(
                WirelessConfig(user_count=settings.fixed_users, rb_count=rb_count, seed=20_000 + rb_count * 101 + seed)
            )
            raw_rows.extend(_raw_rows_for_instance("rbs", rb_count, instance, settings.entropy_taus))

    summary_rows = _summarize(raw_rows)
    raw_csv = output_dir / "raw_results.csv"
    summary_csv = output_dir / "summary.csv"
    _write_csv(raw_csv, raw_rows, RAW_FIELDS)
    _write_csv(summary_csv, summary_rows, SUMMARY_FIELDS)

    plots_dir = output_dir / "plots"
    objective_plot = plots_dir / "objective_gap_by_users.svg"
    runtime_plot = plots_dir / "runtime_by_users.svg"
    convergence_plot = plots_dir / "convergence_gap_by_rbs.svg"
    users_rows = [row for row in summary_rows if row["scenario"] == "users"]
    rbs_rows = [row for row in summary_rows if row["scenario"] == "rbs"]
    write_line_plot(
        users_rows,
        objective_plot,
        "Objective Gap vs Hungarian by Users",
        x_key="x_value",
        y_key="objective_gap_pct_mean",
        x_label="Number of users",
        y_label="Objective gap (%)",
    )
    write_line_plot(
        users_rows,
        runtime_plot,
        "Runtime by Users",
        x_key="x_value",
        y_key="runtime_seconds_mean",
        x_label="Number of users",
        y_label="Mean runtime (s)",
    )
    write_line_plot(
        rbs_rows,
        convergence_plot,
        "Convergence Gap by RB Count",
        x_key="x_value",
        y_key="convergence_gap_mean",
        x_label="Number of RBs",
        y_label="Mean convergence-gap objective",
    )

    return ExperimentPaths(
        raw_csv=raw_csv,
        summary_csv=summary_csv,
        objective_plot=objective_plot,
        runtime_plot=runtime_plot,
        convergence_plot=convergence_plot,
    )
