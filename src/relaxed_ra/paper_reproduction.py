from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean, pstdev
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from .instance import WirelessConfig, generate_instance
from .models import ProblemInstance
from .plots import write_line_plot
from .solvers import (
    solve_hungarian,
    solve_hybrid_score_greedy,
    solve_lp_relaxation,
    solve_soft_entropy_kkt,
    solve_soft_lp,
)


@dataclass(frozen=True)
class PaperReproductionSettings:
    output_dir: Path = Path("results/paper")
    seeds: int = 3
    user_sweep: tuple[int, ...] = (10, 12, 15, 18, 20)
    rb_sweep: tuple[int, ...] = (5, 9, 12, 15, 20)
    sample_sweep: tuple[int, ...] = (10, 20, 30, 40, 60)
    fixed_users: int = 15
    fixed_rbs: int = 12
    digit_rounds: int = 12
    iteration_points: tuple[int, ...] = (0, 2, 4, 6, 8, 10, 12)


@dataclass(frozen=True)
class PaperReproductionPaths:
    raw_csv: Path
    summary_csv: Path
    plots: tuple[Path, ...]


@dataclass(frozen=True)
class MethodProfile:
    method: str
    allocation: np.ndarray
    user_weights: np.ndarray
    objective: float
    selected_mass: float
    mean_packet_error: float
    matching_work_units: float
    fractional_mass_ratio: float
    runtime_seconds: float


RAW_FIELDS = [
    "scenario",
    "x_value",
    "seed",
    "method",
    "metric",
    "value",
    "users",
    "rbs",
    "samples_per_user",
    "round",
    "objective",
    "selected_mass",
    "mean_packet_error",
    "matching_work_units",
    "fractional_mass_ratio",
    "runtime_seconds",
]

SUMMARY_FIELDS = [
    "scenario",
    "x_value",
    "method",
    "metric",
    "runs",
    "value_mean",
    "value_std",
    "objective_mean",
    "selected_mass_mean",
    "mean_packet_error_mean",
    "matching_work_units_mean",
    "fractional_mass_ratio_mean",
    "runtime_seconds_mean",
]

PAPER_METHODS = (
    "Paper-Hungarian",
    "Paper-OptUser-RandomRB",
    "Paper-RandomUserRB",
    "Paper-Wireless-PER",
    "LP-Relax+Projection",
    "Hybrid-Score-Greedy(alpha=0.25)",
    "Continuous-LP(HiGHS)",
    "Soft-Entropy-KKT(tau=1)",
)


def _with_sample_counts(instance: ProblemInstance, sample_counts: np.ndarray) -> ProblemInstance:
    counts = sample_counts.astype(float)
    return replace(instance, sample_counts=counts, cost=counts[:, None] * (instance.packet_error - 1.0))


def _assignment_profile(instance: ProblemInstance, method: str, allocation: np.ndarray, runtime_seconds: float, fractional: float = 0.0) -> MethodProfile:
    x = np.where(instance.feasible, np.maximum(allocation.astype(float), 0.0), 0.0)
    user_weights = np.clip(np.sum((1.0 - instance.packet_error) * x, axis=1), 0.0, 1.0)
    soft_mass = float(np.sum(x))
    mean_q = float(np.sum(instance.packet_error * x) / max(soft_mass, 1e-12))
    objective = float(instance.total_samples + np.sum(instance.cost * x))
    u_count, r_count = instance.user_count, instance.rb_count
    edge_count = u_count * r_count
    if method == "Soft-Entropy-KKT(tau=1)":
        matching_work = float(edge_count + 1_000 * (u_count + r_count) ** 2)
    elif method == "Continuous-LP(HiGHS)":
        matching_work = float(edge_count + edge_count**2)
    elif method == "Hybrid-Score-Greedy(alpha=0.25)":
        matching_work = float(edge_count + 120 * edge_count + edge_count * np.log2(max(edge_count, 2)))
    elif method in {"Paper-OptUser-RandomRB", "Paper-RandomUserRB"}:
        matching_work = float(edge_count)
    else:
        matching_work = float(edge_count + (min(u_count, r_count) ** 2) * max(u_count, r_count))
    return MethodProfile(
        method=method,
        allocation=x,
        user_weights=user_weights,
        objective=objective,
        selected_mass=soft_mass,
        mean_packet_error=mean_q,
        matching_work_units=matching_work,
        fractional_mass_ratio=float(fractional),
        runtime_seconds=runtime_seconds,
    )


def _random_user_rb_profile(instance: ProblemInstance, rng: np.random.Generator) -> MethodProfile:
    start = time.perf_counter()
    assignment = np.zeros((instance.user_count, instance.rb_count), dtype=float)
    users = rng.permutation(instance.user_count)
    rbs = rng.permutation(instance.rb_count)
    for user, rb in zip(users, rbs, strict=False):
        if instance.feasible[user, rb]:
            assignment[user, rb] = 1.0
    return _assignment_profile(instance, "Paper-RandomUserRB", assignment, time.perf_counter() - start)


def _opt_user_random_rb_profile(instance: ProblemInstance, rng: np.random.Generator) -> MethodProfile:
    start = time.perf_counter()
    assignment = np.zeros((instance.user_count, instance.rb_count), dtype=float)
    random_rb = rng.integers(0, instance.rb_count, size=instance.user_count)
    for rb in range(instance.rb_count):
        candidates = [user for user in range(instance.user_count) if random_rb[user] == rb and instance.feasible[user, rb]]
        if candidates:
            best_user = min(candidates, key=lambda user: instance.cost[user, rb])
            assignment[best_user, rb] = 1.0
    return _assignment_profile(instance, "Paper-OptUser-RandomRB", assignment, time.perf_counter() - start)


def _wireless_per_profile(instance: ProblemInstance) -> MethodProfile:
    start = time.perf_counter()
    solver_cost = np.where(instance.feasible, instance.packet_error, 1.0e6)
    rows, cols = linear_sum_assignment(solver_cost)
    assignment = np.zeros((instance.user_count, instance.rb_count), dtype=float)
    for row, col in zip(rows, cols, strict=False):
        if instance.feasible[row, col]:
            assignment[row, col] = 1.0
    return _assignment_profile(instance, "Paper-Wireless-PER", assignment, time.perf_counter() - start)


def _method_profiles(instance: ProblemInstance, rng: np.random.Generator) -> dict[str, MethodProfile]:
    hungarian = solve_hungarian(instance)
    lp = solve_lp_relaxation(instance)
    hybrid = solve_hybrid_score_greedy(instance, alpha=0.25, tau=0.3, iterations=120)
    soft_lp = solve_soft_lp(instance)
    soft_kkt = solve_soft_entropy_kkt(instance, tau=1.0, iterations=1_000)

    profiles = [
        _assignment_profile(instance, "Paper-Hungarian", hungarian.assignment, hungarian.runtime_seconds),
        _opt_user_random_rb_profile(instance, rng),
        _random_user_rb_profile(instance, rng),
        _wireless_per_profile(instance),
        _assignment_profile(instance, lp.method, lp.assignment, lp.runtime_seconds),
        _assignment_profile(instance, hybrid.method, hybrid.assignment, hybrid.runtime_seconds),
        _assignment_profile(instance, soft_lp.method, soft_lp.soft_assignment, soft_lp.runtime_seconds, soft_lp.fractional_mass_ratio),
        _assignment_profile(instance, soft_kkt.method, soft_kkt.soft_assignment, soft_kkt.runtime_seconds, soft_kkt.fractional_mass_ratio),
    ]
    return {profile.method: profile for profile in profiles}


def _raw_row(
    scenario: str,
    x_value: float,
    seed: int,
    profile: MethodProfile,
    metric: str,
    value: float,
    users: int,
    rbs: int,
    samples_per_user: int = 0,
    round_index: int = 0,
) -> dict[str, str]:
    return {
        "scenario": scenario,
        "x_value": f"{x_value:g}",
        "seed": str(seed),
        "method": profile.method,
        "metric": metric,
        "value": f"{value:.10f}",
        "users": str(users),
        "rbs": str(rbs),
        "samples_per_user": str(samples_per_user),
        "round": str(round_index),
        "objective": f"{profile.objective:.10f}",
        "selected_mass": f"{profile.selected_mass:.10f}",
        "mean_packet_error": f"{profile.mean_packet_error:.10f}",
        "matching_work_units": f"{profile.matching_work_units:.10f}",
        "fractional_mass_ratio": f"{profile.fractional_mass_ratio:.10f}",
        "runtime_seconds": f"{profile.runtime_seconds:.10f}",
    }


def _make_regression_clients(user_count: int, samples_per_user: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    clients = []
    for user in range(user_count):
        low = (user % 5) * 0.12
        high = min(1.0, low + 0.55)
        x = rng.uniform(low, high, size=samples_per_user)
        y = -2.0 * x + 1.0 + rng.normal(0.0, 0.4, size=samples_per_user)
        clients.append((x[:, None], y))
    return clients


def _weighted_linear_regression_loss(clients: list[tuple[np.ndarray, np.ndarray]], user_weights: np.ndarray) -> float:
    xs = []
    ys = []
    weights = []
    for idx, (x_client, y_client) in enumerate(clients):
        xs.append(x_client[:, 0])
        ys.append(y_client)
        weights.append(np.full(y_client.shape, max(float(user_weights[idx]), 0.0)))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    w = np.concatenate(weights)
    if float(np.sum(w)) < 1e-9:
        prediction = np.full_like(y, np.average(y))
        return float(np.mean((prediction - y) ** 2))

    design = np.column_stack([x, np.ones_like(x)])
    sqrt_w = np.sqrt(w + 1e-12)
    coef, *_ = np.linalg.lstsq(design * sqrt_w[:, None], y * sqrt_w, rcond=None)
    x_val = np.linspace(0.0, 1.0, 400)
    y_val = -2.0 * x_val + 1.0
    pred = coef[0] * x_val + coef[1]
    return float(np.mean((pred - y_val) ** 2))


def _digit_data(seed: int, user_count: int):
    digits = load_digits()
    x = digits.data.astype(float) / 16.0
    y = digits.target.astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=seed, stratify=y)
    order = np.argsort(y_train, kind="stable")
    x_sorted = x_train[order]
    y_sorted = y_train[order]
    shards = np.array_split(np.arange(len(y_sorted)), user_count)
    clients = [(x_sorted[indexes], y_sorted[indexes]) for indexes in shards]
    return clients, x_test, y_test


def _softmax(z: np.ndarray) -> np.ndarray:
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _accuracy(weights: np.ndarray, bias: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> float:
    pred = np.argmax(x_test @ weights + bias, axis=1)
    return float(np.mean(pred == y_test))


def _local_softmax_step(weights: np.ndarray, bias: np.ndarray, x: np.ndarray, y: np.ndarray, lr: float) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        return weights, bias
    probs = _softmax(x @ weights + bias)
    target = np.zeros_like(probs)
    target[np.arange(len(y)), y] = 1.0
    diff = (probs - target) / max(len(y), 1)
    grad_w = x.T @ diff + 1e-4 * weights
    grad_b = np.sum(diff, axis=0)
    return weights - lr * grad_w, bias - lr * grad_b


def _train_digit_methods(
    user_count: int,
    rb_count: int,
    seed: int,
    rounds: int,
    record_points: tuple[int, ...],
) -> tuple[list[dict[str, float | str | int]], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    clients, x_test, y_test = _digit_data(seed, user_count)
    client_counts = np.asarray([len(y_client) for _, y_client in clients], dtype=float)
    models = {method: (np.zeros((64, 10), dtype=float), np.zeros(10, dtype=float)) for method in PAPER_METHODS}
    rows: list[dict[str, float | str | int]] = []
    last_profiles: dict[str, MethodProfile] = {}

    def append_accuracy(round_index: int, profile_map: dict[str, MethodProfile]) -> None:
        for method, (weights, bias) in models.items():
            profile = profile_map.get(method)
            if profile is None:
                profile = MethodProfile(method, np.zeros((user_count, rb_count)), np.zeros(user_count), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            rows.append(
                {
                    "round": round_index,
                    "method": method,
                    "accuracy": _accuracy(weights, bias, x_test, y_test),
                    "objective": profile.objective,
                    "selected_mass": profile.selected_mass,
                    "mean_packet_error": profile.mean_packet_error,
                    "matching_work_units": profile.matching_work_units,
                    "fractional_mass_ratio": profile.fractional_mass_ratio,
                    "runtime_seconds": profile.runtime_seconds,
                }
            )

    if 0 in record_points:
        append_accuracy(0, last_profiles)

    for round_index in range(1, rounds + 1):
        instance = generate_instance(WirelessConfig(user_count=user_count, rb_count=rb_count, seed=seed * 10_000 + round_index))
        instance = _with_sample_counts(instance, client_counts)
        profiles = _method_profiles(instance, np.random.default_rng(seed * 100_000 + round_index))
        last_profiles = profiles
        for method, profile in profiles.items():
            weights, bias = models[method]
            total_weight = 0.0
            delta_w = np.zeros_like(weights)
            delta_b = np.zeros_like(bias)
            for user_idx, ((x_client, y_client), user_weight) in enumerate(zip(clients, profile.user_weights, strict=False)):
                effective = float(user_weight) * len(y_client)
                if effective <= 1e-9:
                    continue
                local_w, local_b = _local_softmax_step(weights, bias, x_client, y_client, lr=0.8)
                delta_w += effective * (local_w - weights)
                delta_b += effective * (local_b - bias)
                total_weight += effective
            if total_weight > 0:
                models[method] = (weights + delta_w / total_weight, bias + delta_b / total_weight)
        if round_index in record_points:
            append_accuracy(round_index, profiles)

    final_models = {method: (weights, bias, x_test, y_test) for method, (weights, bias) in models.items()}
    return rows, final_models


def _summarize(raw_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(row["scenario"], row["x_value"], row["method"], row["metric"])].append(row)

    summary: list[dict[str, str]] = []
    numeric_fields = [
        "value",
        "objective",
        "selected_mass",
        "mean_packet_error",
        "matching_work_units",
        "fractional_mass_ratio",
        "runtime_seconds",
    ]
    for (scenario, x_value, method, metric), rows in sorted(grouped.items(), key=lambda item: (item[0][0], float(item[0][1]), item[0][2], item[0][3])):
        values = {field: [float(row[field]) for row in rows] for field in numeric_fields}
        summary.append(
            {
                "scenario": scenario,
                "x_value": x_value,
                "method": method,
                "metric": metric,
                "runs": str(len(rows)),
                "value_mean": f"{mean(values['value']):.10f}",
                "value_std": f"{pstdev(values['value']):.10f}",
                "objective_mean": f"{mean(values['objective']):.10f}",
                "selected_mass_mean": f"{mean(values['selected_mass']):.10f}",
                "mean_packet_error_mean": f"{mean(values['mean_packet_error']):.10f}",
                "matching_work_units_mean": f"{mean(values['matching_work_units']):.10f}",
                "fractional_mass_ratio_mean": f"{mean(values['fractional_mass_ratio']):.10f}",
                "runtime_seconds_mean": f"{mean(values['runtime_seconds']):.10f}",
            }
        )
    return summary


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_rows(summary_rows: list[dict[str, str]], scenario: str, metric: str) -> list[dict[str, str]]:
    return [row for row in summary_rows if row["scenario"] == scenario and row["metric"] == metric]


def _write_digit_examples(path: Path, model_tuple: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    weights, bias, x_test, y_test = model_tuple
    preds = np.argmax(x_test @ weights + bias, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cell = 42
    gap = 8
    cols = 6
    rows = 6
    width = cols * cell + (cols + 1) * gap
    height = rows * (cell + 18) + (rows + 1) * gap + 44
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="12" y="28" font-family="Arial, sans-serif" font-size="18" font-weight="700">Digit examples: Paper-Hungarian</text>',
    ]
    for idx in range(min(36, len(x_test))):
        row = idx // cols
        col = idx % cols
        x0 = gap + col * (cell + gap)
        y0 = 44 + gap + row * (cell + 18 + gap)
        image = x_test[idx].reshape(8, 8)
        stroke = "#2ca02c" if preds[idx] == y_test[idx] else "#d62728"
        elements.append(f'<rect x="{x0}" y="{y0}" width="{cell}" height="{cell}" fill="#fff" stroke="{stroke}" stroke-width="2"/>')
        pixel = cell / 8.0
        for py in range(8):
            for px in range(8):
                shade = int(255 - image[py, px] * 255)
                elements.append(
                    f'<rect x="{x0 + px * pixel:.2f}" y="{y0 + py * pixel:.2f}" width="{pixel + 0.2:.2f}" height="{pixel + 0.2:.2f}" fill="rgb({shade},{shade},{shade})"/>'
                )
        elements.append(
            f'<text x="{x0 + cell / 2:.2f}" y="{y0 + cell + 13:.2f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10">p:{preds[idx]} / y:{y_test[idx]}</text>'
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def run_paper_reproduction_suite(settings: PaperReproductionSettings) -> PaperReproductionPaths:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, str]] = []

    for samples_per_user in settings.sample_sweep:
        for seed_index in range(settings.seeds):
            seed = 50_000 + samples_per_user * 101 + seed_index
            clients = _make_regression_clients(settings.fixed_users, samples_per_user, seed)
            instance = generate_instance(WirelessConfig(user_count=settings.fixed_users, rb_count=settings.fixed_rbs, seed=seed))
            instance = _with_sample_counts(instance, np.full(settings.fixed_users, samples_per_user, dtype=float))
            profiles = _method_profiles(instance, np.random.default_rng(seed))
            for profile in profiles.values():
                loss = _weighted_linear_regression_loss(clients, profile.user_weights)
                raw_rows.append(
                    _raw_row("linear_samples", samples_per_user, seed_index, profile, "linear_regression_mse", loss, settings.fixed_users, settings.fixed_rbs, samples_per_user)
                )

    for user_count in settings.user_sweep:
        for seed_index in range(settings.seeds):
            seed = 60_000 + user_count * 101 + seed_index
            instance = generate_instance(WirelessConfig(user_count=user_count, rb_count=settings.fixed_rbs, seed=seed))
            profiles = _method_profiles(instance, np.random.default_rng(seed))
            for profile in profiles.values():
                convergence_gap = profile.objective / max(instance.total_samples, 1e-12)
                raw_rows.append(_raw_row("matching_users", user_count, seed_index, profile, "matching_work_units", profile.matching_work_units, user_count, settings.fixed_rbs))
                raw_rows.append(_raw_row("convergence_users", user_count, seed_index, profile, "normalized_convergence_gap", convergence_gap, user_count, settings.fixed_rbs))

    digit_example_model: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    for seed_index in range(settings.seeds):
        seed = 70_000 + seed_index
        accuracy_rows, final_models = _train_digit_methods(settings.fixed_users, settings.fixed_rbs, seed, settings.digit_rounds, settings.iteration_points)
        if digit_example_model is None:
            digit_example_model = final_models["Paper-Hungarian"]
        for row in accuracy_rows:
            profile = MethodProfile(
                str(row["method"]),
                np.zeros((settings.fixed_users, settings.fixed_rbs)),
                np.zeros(settings.fixed_users),
                float(row["objective"]),
                float(row["selected_mass"]),
                float(row["mean_packet_error"]),
                float(row["matching_work_units"]),
                float(row["fractional_mass_ratio"]),
                float(row["runtime_seconds"]),
            )
            raw_rows.append(
                _raw_row(
                    "digit_iterations",
                    float(row["round"]),
                    seed_index,
                    profile,
                    "digit_accuracy",
                    float(row["accuracy"]),
                    settings.fixed_users,
                    settings.fixed_rbs,
                    round_index=int(row["round"]),
                )
            )

    for user_count in settings.user_sweep:
        for seed_index in range(settings.seeds):
            seed = 80_000 + user_count * 101 + seed_index
            accuracy_rows, _ = _train_digit_methods(user_count, settings.fixed_rbs, seed, settings.digit_rounds, (settings.digit_rounds,))
            for row in accuracy_rows:
                profile = MethodProfile(
                    str(row["method"]),
                    np.zeros((user_count, settings.fixed_rbs)),
                    np.zeros(user_count),
                    float(row["objective"]),
                    float(row["selected_mass"]),
                    float(row["mean_packet_error"]),
                    float(row["matching_work_units"]),
                    float(row["fractional_mass_ratio"]),
                    float(row["runtime_seconds"]),
                )
                raw_rows.append(_raw_row("digit_users", user_count, seed_index, profile, "digit_accuracy", float(row["accuracy"]), user_count, settings.fixed_rbs))

    for rb_count in settings.rb_sweep:
        for seed_index in range(settings.seeds):
            seed = 90_000 + rb_count * 101 + seed_index
            accuracy_rows, _ = _train_digit_methods(settings.fixed_users, rb_count, seed, settings.digit_rounds, (settings.digit_rounds,))
            for row in accuracy_rows:
                profile = MethodProfile(
                    str(row["method"]),
                    np.zeros((settings.fixed_users, rb_count)),
                    np.zeros(settings.fixed_users),
                    float(row["objective"]),
                    float(row["selected_mass"]),
                    float(row["mean_packet_error"]),
                    float(row["matching_work_units"]),
                    float(row["fractional_mass_ratio"]),
                    float(row["runtime_seconds"]),
                )
                raw_rows.append(_raw_row("digit_rbs", rb_count, seed_index, profile, "digit_accuracy", float(row["accuracy"]), settings.fixed_users, rb_count))

    summary_rows = _summarize(raw_rows)
    raw_csv = settings.output_dir / "paper_raw_results.csv"
    summary_csv = settings.output_dir / "paper_summary.csv"
    _write_csv(raw_csv, raw_rows, RAW_FIELDS)
    _write_csv(summary_csv, summary_rows, SUMMARY_FIELDS)

    plots_dir = settings.output_dir / "plots"
    plots = (
        plots_dir / "paper_fig4_linear_loss_by_samples.svg",
        plots_dir / "paper_fig5_matching_work_by_users.svg",
        plots_dir / "paper_fig6_convergence_gap_by_users.svg",
        plots_dir / "paper_fig7_digit_accuracy_by_iterations.svg",
        plots_dir / "paper_fig8_digit_accuracy_by_users.svg",
        plots_dir / "paper_fig9_digit_accuracy_by_rbs.svg",
        plots_dir / "paper_fig10_digit_examples.svg",
    )
    write_line_plot(_plot_rows(summary_rows, "linear_samples", "linear_regression_mse"), plots[0], "Paper Fig.4-style Linear Loss", "x_value", "value_mean", x_label="Samples per user", y_label="Validation MSE")
    write_line_plot(_plot_rows(summary_rows, "matching_users", "matching_work_units"), plots[1], "Paper Fig.5-style Matching Work", "x_value", "value_mean", x_label="Number of users", y_label="Work units")
    write_line_plot(_plot_rows(summary_rows, "convergence_users", "normalized_convergence_gap"), plots[2], "Paper Fig.6-style Convergence Gap", "x_value", "value_mean", x_label="Number of users", y_label="Normalized gap")
    write_line_plot(_plot_rows(summary_rows, "digit_iterations", "digit_accuracy"), plots[3], "Paper Fig.7-style Digit Accuracy", "x_value", "value_mean", x_label="FL iterations", y_label="Accuracy")
    write_line_plot(_plot_rows(summary_rows, "digit_users", "digit_accuracy"), plots[4], "Paper Fig.8-style Digit Accuracy by Users", "x_value", "value_mean", x_label="Number of users", y_label="Accuracy")
    write_line_plot(_plot_rows(summary_rows, "digit_rbs", "digit_accuracy"), plots[5], "Paper Fig.9-style Digit Accuracy by RBs", "x_value", "value_mean", x_label="Number of RBs", y_label="Accuracy")
    if digit_example_model is not None:
        _write_digit_examples(plots[6], digit_example_model)

    return PaperReproductionPaths(raw_csv=raw_csv, summary_csv=summary_csv, plots=plots)
