import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relaxed_ra.models import ProblemInstance
from relaxed_ra.solvers import (
    round_by_projection,
    solve_entropy_relaxation,
    solve_hybrid_score_greedy,
    solve_greedy_cost,
    solve_hungarian,
    solve_lp_relaxation,
)


def make_known_instance() -> ProblemInstance:
    sample_counts = np.array([10.0, 8.0, 6.0])
    packet_error = np.array(
        [
            [0.10, 0.20],
            [0.05, 0.60],
            [0.20, 0.10],
        ]
    )
    feasible = np.array(
        [
            [True, True],
            [True, True],
            [True, True],
        ]
    )
    cost = sample_counts[:, None] * (packet_error - 1.0)
    return ProblemInstance(
        user_count=3,
        rb_count=2,
        sample_counts=sample_counts,
        packet_error=packet_error,
        p_star=np.full((3, 2), 0.1),
        delay=np.full((3, 2), 0.01),
        energy=np.full((3, 2), 0.01),
        feasible=feasible,
        cost=cost,
        seed=7,
    )


class SolverTests(unittest.TestCase):
    def test_hungarian_finds_known_best_partial_matching(self) -> None:
        instance = make_known_instance()

        result = solve_hungarian(instance)

        expected = np.array(
            [
                [0, 1],
                [1, 0],
                [0, 0],
            ],
            dtype=int,
        )
        np.testing.assert_array_equal(result.assignment, expected)
        self.assertAlmostEqual(result.objective, 8.4)
        self.assertEqual(result.selected_users, 2)

    def test_lp_relaxation_projects_to_hungarian_quality(self) -> None:
        instance = make_known_instance()
        hungarian = solve_hungarian(instance)

        lp = solve_lp_relaxation(instance)

        self.assertTrue(np.all(lp.soft_assignment >= -1e-8))
        self.assertLessEqual(float(lp.soft_assignment.sum(axis=1).max()), 1.0 + 1e-7)
        self.assertLessEqual(float(lp.soft_assignment.sum(axis=0).max()), 1.0 + 1e-7)
        self.assertAlmostEqual(lp.objective, hungarian.objective, places=7)
        np.testing.assert_array_equal(lp.assignment, hungarian.assignment)

    def test_projection_rounding_returns_feasible_binary_assignment(self) -> None:
        instance = make_known_instance()
        soft = np.array(
            [
                [0.8, 0.7],
                [0.6, 0.2],
                [0.5, 0.4],
            ]
        )

        rounded = round_by_projection(soft, instance.feasible)

        self.assertTrue(np.array_equal(rounded, rounded.astype(bool)))
        self.assertLessEqual(int(rounded.sum(axis=1).max()), 1)
        self.assertLessEqual(int(rounded.sum(axis=0).max()), 1)
        self.assertEqual(int(rounded.sum()), 2)

    def test_entropy_projection_returns_feasible_projection_result(self) -> None:
        instance = make_known_instance()

        result = solve_entropy_relaxation(instance, tau=0.3, iterations=25, rounding="projection")

        self.assertEqual(result.method, "Entropy-Relax(tau=0.3)+Projection")
        self.assertLessEqual(int(result.assignment.sum(axis=1).max()), 1)
        self.assertLessEqual(int(result.assignment.sum(axis=0).max()), 1)
        self.assertTrue(np.all(result.assignment <= instance.feasible.astype(int)))

    def test_hybrid_score_greedy_alpha_zero_matches_cost_greedy(self) -> None:
        instance = make_known_instance()

        cost_greedy = solve_greedy_cost(instance)
        hybrid = solve_hybrid_score_greedy(instance, alpha=0.0, tau=0.3, iterations=25)

        self.assertEqual(hybrid.method, "Hybrid-Score-Greedy(alpha=0)")
        np.testing.assert_array_equal(hybrid.assignment, cost_greedy.assignment)
        self.assertAlmostEqual(hybrid.objective, cost_greedy.objective)


if __name__ == "__main__":
    unittest.main()
