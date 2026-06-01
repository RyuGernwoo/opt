import csv
import pathlib
import sys
import tempfile
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relaxed_ra.continuous_experiment import ContinuousExperimentSettings, run_continuous_experiment_suite
from relaxed_ra.solvers import solve_hungarian, solve_soft_entropy_kkt, solve_soft_lp
from tests.test_solvers import make_known_instance


class ContinuousSoftAllocationTests(unittest.TestCase):
    def test_soft_lp_matches_hungarian_linear_objective_without_rounding(self) -> None:
        instance = make_known_instance()
        hungarian = solve_hungarian(instance)

        soft_lp = solve_soft_lp(instance)

        self.assertEqual(soft_lp.method, "Continuous-LP(HiGHS)")
        self.assertAlmostEqual(soft_lp.linear_objective, hungarian.objective, places=7)
        self.assertLessEqual(float(soft_lp.soft_assignment.sum(axis=1).max()), 1.0 + 1e-7)
        self.assertLessEqual(float(soft_lp.soft_assignment.sum(axis=0).max()), 1.0 + 1e-7)

    def test_entropy_kkt_returns_fractional_feasible_soft_assignment(self) -> None:
        instance = make_known_instance()

        result = solve_soft_entropy_kkt(instance, tau=50.0, iterations=500)

        self.assertEqual(result.method, "Soft-Entropy-KKT(tau=50)")
        self.assertLessEqual(float(result.soft_assignment.sum(axis=1).max()), 1.0 + 1e-6)
        self.assertLessEqual(float(result.soft_assignment.sum(axis=0).max()), 1.0 + 1e-6)
        self.assertGreater(result.fractional_mass_ratio, 0.0)
        self.assertLess(result.kkt_residual, 1e-5)

    def test_continuous_experiment_writes_separate_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = ContinuousExperimentSettings(
                output_dir=pathlib.Path(tmp),
                seeds=2,
                user_sweep=(6,),
                rb_sweep=(3,),
                fixed_users=6,
                fixed_rbs=3,
                entropy_taus=(25.0,),
            )

            paths = run_continuous_experiment_suite(settings)

            self.assertTrue(paths.raw_csv.exists())
            self.assertTrue(paths.summary_csv.exists())
            self.assertTrue(paths.objective_plot.exists())
            self.assertTrue(paths.fractional_plot.exists())

            with paths.summary_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            methods = {row["method"] for row in rows}
            self.assertIn("Continuous-LP(HiGHS)", methods)
            self.assertIn("Soft-Entropy-KKT(tau=25)", methods)

            svg_text = paths.fractional_plot.read_text(encoding="utf-8")
            self.assertIn("<svg", svg_text)
            self.assertIn("Fractional Mass", svg_text)


if __name__ == "__main__":
    unittest.main()
