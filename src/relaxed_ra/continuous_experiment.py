from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

from .instance import WirelessConfig, generate_instance
from .metrics import evaluate_soft_assignment
from .plots import write_line_plot
from .solvers import solve_hungarian, solve_soft_entropy_kkt, solve_soft_lp


@dataclass(frozen=True)
class ContinuousExperimentSettings:
    output_dir: Path = Path("results/continuous")
    seeds: int = 5
    user_sweep: tuple[int, ...] = (10, 15, 20, 30, 50, 80)
    rb_sweep: tuple[int, ...] = (5, 9, 12, 15, 20)
    fixed_users: int = 15
    fixed_rbs: int = 12
    entropy_taus: tuple[float, ...] = (1.0, 5.0, 25.0)


@dataclass(frozen=True)
class ContinuousExperimentPaths:
    raw_csv: Path
    summary_csv: Path
    objective_plot: Path
    fractional_plot: Path
    runtime_plot: Path


RAW_FIELDS = [
    "scenario",
    "x_value",
    "seed",
    "users",
    "rbs",
    "method",
    "linear_objective",
    "linear_gap_pct",
    "entropy_objective",
    "soft_mass",
    "expected_successful_samples",
    "mean_packet_error",
    "fl_quality",
    "fractional_mass_ratio",
    "kkt_residual",
    "runtime_seconds",
]

SUMMARY_FIELDS = [
    "scenario",
    "x_value",
    "method",
    "runs",
    "linear_objective_mean",
    "linear_gap_pct_mean",
    "entropy_objective_mean",
    "soft_mass_mean",
    "expected_successful_samples_mean",
    "mean_packet_error_mean",
    "fl_quality_mean",
    "fractional_mass_ratio_mean",
    "kkt_residual_mean",
    "runtime_seconds_mean",
    "runtime_seconds_std",
]


def _run_methods(instance, entropy_taus: tuple[float, ...]):
    hungarian = solve_hungarian(instance)
    results = [
        evaluate_soft_assignment(
            instance,
            hungarian.assignment.astype(float),
            "Hungarian(reference)",
            hungarian.runtime_seconds,
        ),
        solve_soft_lp(instance),
    ]
    for tau in entropy_taus:
        results.append(solve_soft_entropy_kkt(instance, tau=tau, iterations=1_000))
    return results


def _raw_rows_for_instance(scenario: str, x_value: int, instance, entropy_taus: tuple[float, ...]) -> list[dict[str, str]]:
    results = _run_methods(instance, entropy_taus)
    reference = next(result for result in results if result.method == "Hungarian(reference)")
    rows: list[dict[str, str]] = []
    for result in results:
        gap_pct = 100.0 * (result.linear_objective - reference.linear_objective) / max(abs(reference.linear_objective), 1e-12)
        rows.append(
            {
                "scenario": scenario,
                "x_value": str(x_value),
                "seed": str(instance.seed),
                "users": str(instance.user_count),
                "rbs": str(instance.rb_count),
                "method": result.method,
                "linear_objective": f"{result.linear_objective:.10f}",
                "linear_gap_pct": f"{gap_pct:.10f}",
                "entropy_objective": f"{result.entropy_objective:.10f}",
                "soft_mass": f"{result.soft_mass:.10f}",
                "expected_successful_samples": f"{result.expected_successful_samples:.10f}",
                "mean_packet_error": f"{result.mean_packet_error:.10f}",
                "fl_quality": f"{result.fl_quality:.10f}",
                "fractional_mass_ratio": f"{result.fractional_mass_ratio:.10f}",
                "kkt_residual": f"{result.kkt_residual:.10e}",
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
        "linear_objective",
        "linear_gap_pct",
        "entropy_objective",
        "soft_mass",
        "expected_successful_samples",
        "mean_packet_error",
        "fl_quality",
        "fractional_mass_ratio",
        "kkt_residual",
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
                "linear_objective_mean": f"{mean(values['linear_objective']):.10f}",
                "linear_gap_pct_mean": f"{mean(values['linear_gap_pct']):.10f}",
                "entropy_objective_mean": f"{mean(values['entropy_objective']):.10f}",
                "soft_mass_mean": f"{mean(values['soft_mass']):.10f}",
                "expected_successful_samples_mean": f"{mean(values['expected_successful_samples']):.10f}",
                "mean_packet_error_mean": f"{mean(values['mean_packet_error']):.10f}",
                "fl_quality_mean": f"{mean(values['fl_quality']):.10f}",
                "fractional_mass_ratio_mean": f"{mean(values['fractional_mass_ratio']):.10f}",
                "kkt_residual_mean": f"{mean(values['kkt_residual']):.10e}",
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


def run_continuous_experiment_suite(settings: ContinuousExperimentSettings) -> ContinuousExperimentPaths:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, str]] = []

    for user_count in settings.user_sweep:
        for seed in range(settings.seeds):
            instance = generate_instance(
                WirelessConfig(user_count=user_count, rb_count=settings.fixed_rbs, seed=30_000 + user_count * 101 + seed)
            )
            raw_rows.extend(_raw_rows_for_instance("users", user_count, instance, settings.entropy_taus))

    for rb_count in settings.rb_sweep:
        for seed in range(settings.seeds):
            instance = generate_instance(
                WirelessConfig(user_count=settings.fixed_users, rb_count=rb_count, seed=40_000 + rb_count * 101 + seed)
            )
            raw_rows.extend(_raw_rows_for_instance("rbs", rb_count, instance, settings.entropy_taus))

    summary_rows = _summarize(raw_rows)
    raw_csv = settings.output_dir / "raw_results.csv"
    summary_csv = settings.output_dir / "summary.csv"
    _write_csv(raw_csv, raw_rows, RAW_FIELDS)
    _write_csv(summary_csv, summary_rows, SUMMARY_FIELDS)

    plots_dir = settings.output_dir / "plots"
    objective_plot = plots_dir / "soft_objective_gap_by_users.svg"
    fractional_plot = plots_dir / "fractional_mass_by_users.svg"
    runtime_plot = plots_dir / "soft_runtime_by_users.svg"
    users_rows = [row for row in summary_rows if row["scenario"] == "users"]
    write_line_plot(
        users_rows,
        objective_plot,
        "Continuous Soft Linear Objective Gap by Users",
        x_key="x_value",
        y_key="linear_gap_pct_mean",
        x_label="Number of users",
        y_label="Linear objective gap (%)",
    )
    write_line_plot(
        users_rows,
        fractional_plot,
        "Fractional Mass by Users",
        x_key="x_value",
        y_key="fractional_mass_ratio_mean",
        x_label="Number of users",
        y_label="Fractional mass ratio",
    )
    write_line_plot(
        users_rows,
        runtime_plot,
        "Continuous Soft Runtime by Users",
        x_key="x_value",
        y_key="runtime_seconds_mean",
        x_label="Number of users",
        y_label="Mean runtime (s)",
    )
    return ContinuousExperimentPaths(
        raw_csv=raw_csv,
        summary_csv=summary_csv,
        objective_plot=objective_plot,
        fractional_plot=fractional_plot,
        runtime_plot=runtime_plot,
    )
