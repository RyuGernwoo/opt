import csv
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relaxed_ra.experiment import ExperimentSettings, run_experiment_suite


class ExperimentOutputTests(unittest.TestCase):
    def test_tiny_experiment_writes_csv_and_svg_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = pathlib.Path(tmp)
            settings = ExperimentSettings(
                output_dir=output_dir,
                seeds=2,
                user_sweep=(6,),
                rb_sweep=(3,),
                fixed_users=6,
                fixed_rbs=3,
                entropy_taus=(0.15,),
            )

            paths = run_experiment_suite(settings)

            self.assertTrue(paths.raw_csv.exists())
            self.assertTrue(paths.summary_csv.exists())
            self.assertTrue(paths.objective_plot.exists())
            self.assertTrue(paths.runtime_plot.exists())
            self.assertTrue(paths.convergence_plot.exists())

            with paths.summary_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            methods = {row["method"] for row in rows}
            self.assertIn("Hungarian", methods)
            self.assertIn("LP-Relax+Projection", methods)
            self.assertIn("Entropy-Relax(tau=0.15)+Greedy", methods)
            self.assertIn("Entropy-Relax(tau=0.15)+Projection", methods)
            self.assertIn("Hybrid-Score-Greedy(alpha=0.25)", methods)

            svg_text = paths.objective_plot.read_text(encoding="utf-8")
            self.assertIn("<svg", svg_text)
            self.assertIn("Objective Gap", svg_text)


if __name__ == "__main__":
    unittest.main()
