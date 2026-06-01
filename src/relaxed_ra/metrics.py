from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from .models import AllocationResult, ProblemInstance, SoftAllocationResult


def assignment_objective(instance: ProblemInstance, assignment: np.ndarray) -> float:
    z = assignment.astype(float)
    return float(instance.total_samples + np.sum(instance.cost * z))


def evaluate_assignment(
    instance: ProblemInstance,
    assignment: np.ndarray,
    method: str,
    runtime_seconds: float,
    soft_assignment: np.ndarray | None = None,
    details: dict | None = None,
) -> AllocationResult:
    z = assignment.astype(int)
    selected_mask = z.sum(axis=1) > 0
    selected_users = int(selected_mask.sum())
    selected_packet_errors = instance.packet_error[z == 1]
    mean_q = float(np.mean(selected_packet_errors)) if selected_packet_errors.size else 0.0
    expected_success = float(np.sum(instance.sample_counts[:, None] * (1.0 - instance.packet_error) * z))
    objective = assignment_objective(instance, z)
    fl_quality = expected_success / max(instance.total_samples, 1e-12)
    return AllocationResult(
        method=method,
        assignment=z,
        soft_assignment=z.astype(float) if soft_assignment is None else soft_assignment.astype(float),
        objective=objective,
        selected_users=selected_users,
        expected_successful_samples=expected_success,
        mean_packet_error=mean_q,
        convergence_gap=objective,
        fl_quality=fl_quality,
        runtime_seconds=float(runtime_seconds),
        details={} if details is None else details,
    )


def evaluate_soft_assignment(
    instance: ProblemInstance,
    soft_assignment: np.ndarray,
    method: str,
    runtime_seconds: float,
    entropy_objective: float | None = None,
    kkt_residual: float = 0.0,
    details: dict | None = None,
) -> SoftAllocationResult:
    x = np.where(instance.feasible, np.maximum(soft_assignment.astype(float), 0.0), 0.0)
    linear_objective = float(instance.total_samples + np.sum(instance.cost * x))
    soft_mass = float(np.sum(x))
    expected_success = float(np.sum(instance.sample_counts[:, None] * (1.0 - instance.packet_error) * x))
    weighted_errors = float(np.sum(instance.packet_error * x))
    mean_q = weighted_errors / max(soft_mass, 1e-12)
    fl_quality = expected_success / max(instance.total_samples, 1e-12)
    fractional_mass = float(np.sum(x[(x > 1e-6) & (x < 1.0 - 1e-6)]))
    fractional_mass_ratio = fractional_mass / max(soft_mass, 1e-12)
    return SoftAllocationResult(
        method=method,
        soft_assignment=x,
        linear_objective=linear_objective,
        entropy_objective=linear_objective if entropy_objective is None else float(entropy_objective),
        soft_mass=soft_mass,
        expected_successful_samples=expected_success,
        mean_packet_error=mean_q,
        fl_quality=fl_quality,
        fractional_mass_ratio=fractional_mass_ratio,
        kkt_residual=float(kkt_residual),
        runtime_seconds=float(runtime_seconds),
        details={} if details is None else details,
    )


def timed(callable_: Callable[[], AllocationResult]) -> AllocationResult:
    start = time.perf_counter()
    result = callable_()
    elapsed = time.perf_counter() - start
    return AllocationResult(
        method=result.method,
        assignment=result.assignment,
        soft_assignment=result.soft_assignment,
        objective=result.objective,
        selected_users=result.selected_users,
        expected_successful_samples=result.expected_successful_samples,
        mean_packet_error=result.mean_packet_error,
        convergence_gap=result.convergence_gap,
        fl_quality=result.fl_quality,
        runtime_seconds=elapsed,
        details=result.details,
    )
