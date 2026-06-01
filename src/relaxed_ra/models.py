from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProblemInstance:
    user_count: int
    rb_count: int
    sample_counts: np.ndarray
    packet_error: np.ndarray
    p_star: np.ndarray
    delay: np.ndarray
    energy: np.ndarray
    feasible: np.ndarray
    cost: np.ndarray
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_samples(self) -> float:
        return float(np.sum(self.sample_counts))


@dataclass(frozen=True)
class AllocationResult:
    method: str
    assignment: np.ndarray
    soft_assignment: np.ndarray
    objective: float
    selected_users: int
    expected_successful_samples: float
    mean_packet_error: float
    convergence_gap: float
    fl_quality: float
    runtime_seconds: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SoftAllocationResult:
    method: str
    soft_assignment: np.ndarray
    linear_objective: float
    entropy_objective: float
    soft_mass: float
    expected_successful_samples: float
    mean_packet_error: float
    fl_quality: float
    fractional_mass_ratio: float
    kkt_residual: float
    runtime_seconds: float
    details: dict[str, Any] = field(default_factory=dict)
