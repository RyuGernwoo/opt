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
    user_sweep: tuple[int, ...] = (3, 6, 9, 12, 15, 18)
    matching_user_sweep: tuple[int, ...] = (3, 5, 10, 15, 20, 25)
    rb_sweep: tuple[int, ...] = (3, 6, 9, 12)
    sample_sweep: tuple[int, ...] = (10, 20, 30, 40, 50)
    fixed_users: int = 15
    fixed_rbs: int = 12
    digit_rounds: int = 120
    digit_sweep_rounds: int = 12
    iteration_points: tuple[int, ...] = (0, 10, 20, 40, 60, 80, 100, 120)


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

PAPER_STYLE = {
    "Paper-Hungarian": ("Proposed FL", "#000000", "", "none"),
    "Paper-OptUser-RandomRB": ("Baseline a)", "#0000ff", "9 7", "none"),
    "Paper-RandomUserRB": ("Baseline b)", "#ff0000", "", "none"),
    "Paper-Wireless-PER": ("Baseline c)", "#b00020", "2 3", "none"),
    "LP-Relax+Projection": ("LP-Relax+Proj.", "#ff00ff", "8 5", "circle"),
    "Hybrid-Score-Greedy(alpha=0.25)": ("Hybrid-Greedy", "#00aa00", "", "square"),
    "Continuous-LP(HiGHS)": ("Continuous-LP", "#00a6d6", "7 4", "triangle"),
    "Soft-Entropy-KKT(tau=1)": ("Soft-KKT tau=1", "#ff8c00", "", "diamond"),
}


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


def _svg_marker(marker: str, x: float, y: float, color: str) -> str:
    if marker == "circle":
        return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.3" fill="{color}" stroke="{color}"/>'
    if marker == "square":
        return f'<rect x="{x - 3.1:.2f}" y="{y - 3.1:.2f}" width="6.2" height="6.2" fill="{color}" stroke="{color}"/>'
    if marker == "triangle":
        return f'<polygon points="{x:.2f},{y - 4.0:.2f} {x - 4.0:.2f},{y + 3.5:.2f} {x + 4.0:.2f},{y + 3.5:.2f}" fill="{color}" stroke="{color}"/>'
    if marker == "diamond":
        return f'<polygon points="{x:.2f},{y - 4.2:.2f} {x - 4.2:.2f},{y:.2f} {x:.2f},{y + 4.2:.2f} {x + 4.2:.2f},{y:.2f}" fill="{color}" stroke="{color}"/>'
    return ""


def _scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if abs(src_max - src_min) < 1e-12:
        return (dst_min + dst_max) / 2.0
    return dst_min + (value - src_min) * (dst_max - dst_min) / (src_max - src_min)


def _tick_values(low: float, high: float, count: int = 6) -> list[float]:
    if abs(high - low) < 1e-12:
        return [low]
    return [low + (high - low) * index / max(count - 1, 1) for index in range(count)]


def _fmt_tick(value: float) -> str:
    if abs(value) >= 10:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.2g}"
    return f"{value:.3g}"


def _write_paper_line_plot(
    rows: list[dict[str, str]],
    output_path: Path,
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    caption: str,
    x_ticks: tuple[float, ...] | None = None,
    y_limits: tuple[float, float] | None = None,
    legend_anchor: str = "upper_right",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        method = row.get("method", "")
        if method not in PAPER_STYLE:
            continue
        try:
            grouped[method].append((float(row[x_key]), float(row[y_key])))
        except (KeyError, ValueError):
            continue

    width, height = 540, 390
    left, top, plot_w, plot_h = 78, 22, 386, 260
    caption_y = 344
    points = [point for method in PAPER_METHODS for point in grouped.get(method, [])]
    if x_ticks:
        x_min, x_max = min(x_ticks), max(x_ticks)
    elif points:
        x_min, x_max = min(point[0] for point in points), max(point[0] for point in points)
    else:
        x_min, x_max = 0.0, 1.0

    if y_limits:
        y_min, y_max = y_limits
    elif points:
        y_min = min(point[1] for point in points)
        y_max = max(point[1] for point in points)
        pad = max((y_max - y_min) * 0.08, 1e-6)
        y_min -= pad
        y_max += pad
        if y_min > 0 and y_max > 0:
            y_min = max(0.0, y_min)
    else:
        y_min, y_max = 0.0, 1.0

    def sx(x_value: float) -> float:
        return _scale(x_value, x_min, x_max, left, left + plot_w)

    def sy(y_value: float) -> float:
        return _scale(y_value, y_min, y_max, top + plot_h, top)

    x_values = list(x_ticks) if x_ticks else sorted({point[0] for point in points})
    y_values = _tick_values(y_min, y_max, 6)
    font = "Arial, Helvetica, sans-serif"
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#000000" stroke-width="1"/>',
    ]
    for y_value in y_values:
        y = sy(y_value)
        elements.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#d9d9d9" stroke-width="1"/>')
        elements.append(f'<text x="{left - 9}" y="{y + 4:.2f}" text-anchor="end" font-family="{font}" font-size="14">{_fmt_tick(y_value)}</text>')
    for x_value in x_values:
        x = sx(x_value)
        elements.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#d9d9d9" stroke-width="1"/>')
        elements.append(f'<text x="{x:.2f}" y="{top + plot_h + 23}" text-anchor="middle" font-family="{font}" font-size="14">{_fmt_tick(x_value)}</text>')

    elements.append(f'<text x="{left + plot_w / 2}" y="{top + plot_h + 47}" text-anchor="middle" font-family="{font}" font-size="16">{x_label}</text>')
    elements.append(
        f'<text transform="translate(22 {top + plot_h / 2}) rotate(-90)" text-anchor="middle" font-family="{font}" font-size="16">{y_label}</text>'
    )

    for method in PAPER_METHODS:
        series = sorted(grouped.get(method, []))
        if not series:
            continue
        label, color, dash, marker = PAPER_STYLE[method]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        if len(series) >= 2:
            coords = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in series)
            elements.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3"{dash_attr}/>')
        for x_value, y_value in series:
            elements.append(_svg_marker(marker, sx(x_value), sy(y_value), color))

    used_methods = [method for method in PAPER_METHODS if grouped.get(method)]
    legend_w, legend_h = 174, 20 + 18 * len(used_methods)
    if legend_anchor == "lower_right":
        legend_x, legend_y = left + plot_w - legend_w - 12, top + plot_h - legend_h - 14
    elif legend_anchor == "upper_left":
        legend_x, legend_y = left + 14, top + 14
    else:
        legend_x, legend_y = left + plot_w - legend_w - 12, top + 14
    elements.append(f'<rect x="{legend_x}" y="{legend_y}" width="{legend_w}" height="{legend_h}" fill="#ffffff" stroke="#000000" stroke-width="1"/>')
    for idx, method in enumerate(used_methods):
        label, color, dash, marker = PAPER_STYLE[method]
        y = legend_y + 17 + idx * 18
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        elements.append(f'<line x1="{legend_x + 10}" y1="{y}" x2="{legend_x + 42}" y2="{y}" stroke="{color}" stroke-width="3"{dash_attr}/>')
        elements.append(_svg_marker(marker, legend_x + 26, y, color))
        elements.append(f'<text x="{legend_x + 48}" y="{y + 5}" font-family="{font}" font-size="12">{label}</text>')

    caption_lines = [caption] if len(caption) <= 72 else [caption[:72].rstrip(), caption[72:].strip()]
    for index, caption_line in enumerate(caption_lines):
        elements.append(f'<text x="{left}" y="{caption_y + index * 17}" font-family="Times New Roman, serif" font-size="14">{caption_line}</text>')
    elements.append("</svg>")
    output_path.write_text("\n".join(part for part in elements if part), encoding="utf-8")


def _plot_rows(summary_rows: list[dict[str, str]], scenario: str, metric: str) -> list[dict[str, str]]:
    return [row for row in summary_rows if row["scenario"] == scenario and row["metric"] == metric]


def _write_digit_examples(
    path: Path,
    proposed_tuple: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    baseline_tuple: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    weights, bias, x_test, y_test = proposed_tuple
    baseline_weights, baseline_bias, _, _ = baseline_tuple
    preds = np.argmax(x_test @ weights + bias, axis=1)
    baseline_preds = np.argmax(x_test @ baseline_weights + baseline_bias, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cell = 44
    gap = 13
    cols = 6
    rows = 6
    label_w = 98
    width = label_w + cols * cell + (cols - 1) * gap + 22
    height = rows * (cell + 28) + 40
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="12" y="22" font-family="Arial, sans-serif" font-size="13" font-weight="700">Proposed FL:</text>',
        '<text x="12" y="42" font-family="Arial, sans-serif" font-size="13">Baseline b):</text>',
    ]
    for idx in range(min(36, len(x_test))):
        row = idx // cols
        col = idx % cols
        x0 = label_w + col * (cell + gap)
        y0 = 52 + row * (cell + 28)
        image = x_test[idx].reshape(8, 8)
        proposed_color = "#ff0000" if preds[idx] != y_test[idx] else "#000000"
        baseline_color = "#ff0000" if baseline_preds[idx] != y_test[idx] else "#000000"
        elements.append(
            f'<text x="{x0 + cell / 2:.2f}" y="{y0 - 8:.2f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="700" fill="{proposed_color}">{preds[idx]}</text>'
        )
        elements.append(f'<rect x="{x0}" y="{y0}" width="{cell}" height="{cell}" fill="#000000" stroke="#000000" stroke-width="1"/>')
        pixel = cell / 8.0
        for py in range(8):
            for px in range(8):
                shade = int(image[py, px] * 255)
                elements.append(
                    f'<rect x="{x0 + px * pixel:.2f}" y="{y0 + py * pixel:.2f}" width="{pixel + 0.2:.2f}" height="{pixel + 0.2:.2f}" fill="rgb({shade},{shade},{shade})"/>'
                )
        elements.append(
            f'<text x="{x0 + cell / 2:.2f}" y="{y0 + cell + 14:.2f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="{baseline_color}">{baseline_preds[idx]}</text>'
        )
        elements.append(
            f'<text x="{x0 + cell / 2:.2f}" y="{y0 + cell + 27:.2f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10">{y_test[idx]}</text>'
        )
    elements.append(
        f'<text x="12" y="{height - 10}" font-family="Times New Roman, serif" font-size="14">Fig. 10. An example of implementing FL for handwritten digit identification.</text>'
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

    for user_count in settings.matching_user_sweep:
        for seed_index in range(settings.seeds):
            seed = 60_000 + user_count * 101 + seed_index
            instance = generate_instance(WirelessConfig(user_count=user_count, rb_count=settings.fixed_rbs, seed=seed))
            profiles = _method_profiles(instance, np.random.default_rng(seed))
            for profile in profiles.values():
                iteration_proxy = 12.0 * float(np.log1p(profile.matching_work_units))
                raw_rows.append(_raw_row("matching_users", user_count, seed_index, profile, "matching_iterations", iteration_proxy, user_count, settings.fixed_rbs))

    for user_count in settings.user_sweep:
        for seed_index in range(settings.seeds):
            seed = 65_000 + user_count * 101 + seed_index
            instance = generate_instance(WirelessConfig(user_count=user_count, rb_count=settings.fixed_rbs, seed=seed))
            profiles = _method_profiles(instance, np.random.default_rng(seed))
            for profile in profiles.values():
                convergence_gap = profile.objective / max(instance.total_samples, 1e-12)
                raw_rows.append(_raw_row("convergence_users", user_count, seed_index, profile, "normalized_convergence_gap", convergence_gap, user_count, settings.fixed_rbs))

    digit_example_model: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    digit_example_baseline: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    for seed_index in range(settings.seeds):
        seed = 70_000 + seed_index
        accuracy_rows, final_models = _train_digit_methods(settings.fixed_users, settings.fixed_rbs, seed, settings.digit_rounds, settings.iteration_points)
        if digit_example_model is None:
            digit_example_model = final_models["Paper-Hungarian"]
            digit_example_baseline = final_models["Paper-RandomUserRB"]
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
            accuracy_rows, _ = _train_digit_methods(user_count, settings.fixed_rbs, seed, settings.digit_sweep_rounds, (settings.digit_sweep_rounds,))
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
            accuracy_rows, _ = _train_digit_methods(settings.fixed_users, rb_count, seed, settings.digit_sweep_rounds, (settings.digit_sweep_rounds,))
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
    _write_paper_line_plot(
        _plot_rows(summary_rows, "linear_samples", "linear_regression_mse"),
        plots[0],
        "x_value",
        "value_mean",
        "Number of data samples per user",
        "Value of the loss function",
        "Fig. 4. Training loss as the number of data samples per user varies.",
        x_ticks=(10, 20, 30, 40, 50),
        legend_anchor="upper_right",
    )
    _write_paper_line_plot(
        _plot_rows(summary_rows, "matching_users", "matching_iterations"),
        plots[1],
        "x_value",
        "value_mean",
        "Number of users",
        "Number of iterations",
        "Fig. 5. Number of iterations as the number of users varies.",
        x_ticks=(3, 5, 10, 15, 20, 25),
        y_limits=(0, 180),
        legend_anchor="upper_left",
    )
    _write_paper_line_plot(
        _plot_rows(summary_rows, "convergence_users", "normalized_convergence_gap"),
        plots[2],
        "x_value",
        "value_mean",
        "Number of users",
        "Convergence gap due to wireless factors",
        "Fig. 6. Convergence gap caused by wireless factors as the number of users changes.",
        x_ticks=(3, 6, 9, 12, 15, 18),
        legend_anchor="upper_left",
    )
    _write_paper_line_plot(
        _plot_rows(summary_rows, "digit_iterations", "digit_accuracy"),
        plots[3],
        "x_value",
        "value_mean",
        "Number of iterations",
        "Identification accuracy",
        "Fig. 7. Identification accuracy as the number of iterations varies.",
        x_ticks=(0, 20, 40, 60, 80, 100, 120),
        y_limits=(0.08, 0.96),
        legend_anchor="lower_right",
    )
    _write_paper_line_plot(
        _plot_rows(summary_rows, "digit_users", "digit_accuracy"),
        plots[4],
        "x_value",
        "value_mean",
        "Total number of users",
        "Identification accuracy",
        "Fig. 8. Identification accuracy as the total number of users varies (R = 12).",
        x_ticks=(3, 6, 9, 12, 15, 18),
        y_limits=(0.1, 0.92),
        legend_anchor="lower_right",
    )
    _write_paper_line_plot(
        _plot_rows(summary_rows, "digit_rbs", "digit_accuracy"),
        plots[5],
        "x_value",
        "value_mean",
        "Number of RBs",
        "Identification accuracy",
        "Fig. 9. Identification accuracy changes as the number of RBs varies (U = 15).",
        x_ticks=(3, 6, 9, 12),
        y_limits=(0.1, 0.92),
        legend_anchor="lower_right",
    )
    if digit_example_model is not None and digit_example_baseline is not None:
        _write_digit_examples(plots[6], digit_example_model, digit_example_baseline)

    return PaperReproductionPaths(raw_csv=raw_csv, summary_csv=summary_csv, plots=plots)
