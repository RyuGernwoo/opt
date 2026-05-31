from __future__ import annotations

import time

import numpy as np
from scipy.optimize import linear_sum_assignment, linprog

from .metrics import evaluate_assignment
from .models import AllocationResult, ProblemInstance


def _empty_assignment(instance: ProblemInstance) -> np.ndarray:
    return np.zeros((instance.user_count, instance.rb_count), dtype=int)


def _linear_sum_round(cost_matrix: np.ndarray, feasible: np.ndarray, keep_scores: np.ndarray) -> np.ndarray:
    rows, cols = linear_sum_assignment(cost_matrix)
    assignment = np.zeros_like(feasible, dtype=int)
    for row, col in zip(rows, cols, strict=False):
        if feasible[row, col] and keep_scores[row, col] > 1e-10:
            assignment[row, col] = 1
    return assignment


def round_by_projection(soft_assignment: np.ndarray, feasible: np.ndarray) -> np.ndarray:
    soft = np.where(feasible, np.maximum(soft_assignment, 0.0), 0.0)
    if not np.any(soft > 1e-10):
        return np.zeros_like(feasible, dtype=int)
    return _linear_sum_round(-soft, feasible, soft)


def round_by_greedy(score: np.ndarray, feasible: np.ndarray) -> np.ndarray:
    score = np.where(feasible, score, 0.0)
    assignment = np.zeros_like(feasible, dtype=int)
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    order = np.argsort(score, axis=None)[::-1]
    rb_count = score.shape[1]
    for flat_index in order:
        row = int(flat_index // rb_count)
        col = int(flat_index % rb_count)
        if score[row, col] <= 1e-10:
            break
        if row in used_rows or col in used_cols:
            continue
        assignment[row, col] = 1
        used_rows.add(row)
        used_cols.add(col)
    return assignment


def _unit_normalize(values: np.ndarray, feasible: np.ndarray) -> np.ndarray:
    masked = np.where(feasible, values, 0.0)
    feasible_values = masked[feasible]
    if feasible_values.size == 0:
        return masked
    low = float(np.min(feasible_values))
    high = float(np.max(feasible_values))
    if abs(high - low) < 1e-12:
        return np.where(feasible, 1.0, 0.0)
    return np.where(feasible, (masked - low) / (high - low), 0.0)


def _entropy_soft_assignment(
    instance: ProblemInstance,
    tau: float,
    iterations: int,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    if tau <= 0:
        raise ValueError("tau must be positive")

    utility = np.where(instance.feasible, -instance.cost, 0.0)
    scale = max(float(np.max(utility)), 1e-12)
    logits = np.where(instance.feasible, utility / scale / tau, -60.0)
    if initial is not None:
        logits = logits + 0.25 * np.log(np.maximum(np.where(instance.feasible, initial, 0.0), 1e-12))
    soft = np.exp(np.clip(logits, -60.0, 60.0))
    soft = np.where(instance.feasible, soft, 0.0)

    for _ in range(iterations):
        row_sums = soft.sum(axis=1, keepdims=True)
        soft = np.where(row_sums > 1.0, soft / np.maximum(row_sums, 1e-12), soft)
        col_sums = soft.sum(axis=0, keepdims=True)
        soft = np.where(col_sums > 1.0, soft / np.maximum(col_sums, 1e-12), soft)
    return soft


def solve_hungarian(instance: ProblemInstance) -> AllocationResult:
    start = time.perf_counter()
    solver_cost = np.where(instance.feasible, instance.cost, 0.0)
    keep_scores = np.where(instance.feasible, -instance.cost, 0.0)
    assignment = _linear_sum_round(solver_cost, instance.feasible, keep_scores)
    elapsed = time.perf_counter() - start
    return evaluate_assignment(instance, assignment, "Hungarian", elapsed)


def solve_greedy_cost(instance: ProblemInstance) -> AllocationResult:
    start = time.perf_counter()
    utility = np.where(instance.feasible, -instance.cost, 0.0)
    assignment = round_by_greedy(utility, instance.feasible)
    elapsed = time.perf_counter() - start
    return evaluate_assignment(instance, assignment, "Greedy-Cost", elapsed)


def solve_lp_relaxation(instance: ProblemInstance) -> AllocationResult:
    start = time.perf_counter()
    u_count, r_count = instance.user_count, instance.rb_count
    n_vars = u_count * r_count
    c = instance.cost.reshape(-1)

    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []
    for i in range(u_count):
        row = np.zeros(n_vars)
        row[i * r_count : (i + 1) * r_count] = 1.0
        a_ub.append(row)
        b_ub.append(1.0)
    for n in range(r_count):
        col = np.zeros(n_vars)
        col[n::r_count] = 1.0
        a_ub.append(col)
        b_ub.append(1.0)

    bounds = [(0.0, 1.0) if feasible else (0.0, 0.0) for feasible in instance.feasible.reshape(-1)]
    result = linprog(
        c,
        A_ub=np.vstack(a_ub),
        b_ub=np.asarray(b_ub),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"LP relaxation failed: {result.message}")

    soft = result.x.reshape(u_count, r_count)
    assignment = round_by_projection(soft, instance.feasible)
    elapsed = time.perf_counter() - start
    return evaluate_assignment(
        instance,
        assignment,
        "LP-Relax+Projection",
        elapsed,
        soft_assignment=soft,
        details={"lp_objective": float(result.fun), "linprog_status": int(result.status)},
    )


def solve_entropy_relaxation(
    instance: ProblemInstance,
    tau: float = 0.3,
    iterations: int = 200,
    rounding: str = "greedy",
) -> AllocationResult:
    start = time.perf_counter()
    soft = _entropy_soft_assignment(instance, tau=tau, iterations=iterations)

    if rounding == "projection":
        assignment = round_by_projection(soft, instance.feasible)
        suffix = "Projection"
    elif rounding == "greedy":
        assignment = round_by_greedy(soft, instance.feasible)
        suffix = "Greedy"
    else:
        raise ValueError(f"unknown rounding mode: {rounding}")

    elapsed = time.perf_counter() - start
    return evaluate_assignment(
        instance,
        assignment,
        f"Entropy-Relax(tau={tau:g})+{suffix}",
        elapsed,
        soft_assignment=soft,
        details={"tau": tau, "iterations": iterations, "rounding": rounding},
    )


def solve_hybrid_score_greedy(
    instance: ProblemInstance,
    alpha: float = 0.25,
    tau: float = 0.3,
    iterations: int = 120,
) -> AllocationResult:
    start = time.perf_counter()
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    soft = _entropy_soft_assignment(instance, tau=tau, iterations=iterations)
    utility = np.where(instance.feasible, -instance.cost, 0.0)
    score = alpha * _unit_normalize(soft, instance.feasible) + (1.0 - alpha) * _unit_normalize(utility, instance.feasible)
    assignment = round_by_greedy(score, instance.feasible)
    elapsed = time.perf_counter() - start
    return evaluate_assignment(
        instance,
        assignment,
        f"Hybrid-Score-Greedy(alpha={alpha:g})",
        elapsed,
        soft_assignment=soft,
        details={"alpha": alpha, "tau": tau, "iterations": iterations},
    )
