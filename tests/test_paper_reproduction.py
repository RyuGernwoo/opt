from __future__ import annotations

import csv
import pathlib
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relaxed_ra.paper_reproduction import PaperReproductionSettings, run_paper_reproduction_suite


class PaperReproductionTests(unittest.TestCase):
    def test_tiny_paper_reproduction_writes_csv_and_svg_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = PaperReproductionSettings(
                output_dir=Path(tmp),
                seeds=1,
                user_sweep=(6,),
                rb_sweep=(4,),
                sample_sweep=(8,),
                fixed_users=6,
                fixed_rbs=4,
                digit_rounds=2,
                iteration_points=(0, 1, 2),
            )
            paths = run_paper_reproduction_suite(settings)

            self.assertTrue(paths.raw_csv.exists())
            self.assertTrue(paths.summary_csv.exists())
            for plot in paths.plots:
                self.assertTrue(plot.exists())
                self.assertGreater(plot.stat().st_size, 200)

            with paths.summary_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            methods = {row["method"] for row in rows}
            self.assertIn("Paper-Hungarian", methods)
            self.assertIn("LP-Relax+Projection", methods)
            self.assertIn("Hybrid-Score-Greedy(alpha=0.25)", methods)
            self.assertIn("Continuous-LP(HiGHS)", methods)
            self.assertIn("Soft-Entropy-KKT(tau=1)", methods)


if __name__ == "__main__":
    unittest.main()
