from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from .models import AllocationResult, ProblemInstance


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
