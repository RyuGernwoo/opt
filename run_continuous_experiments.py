from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from relaxed_ra.continuous_experiment import ContinuousExperimentSettings, run_continuous_experiment_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run continuous soft RB allocation experiments.")
    parser.add_argument("--output", type=Path, default=Path("results/continuous"), help="Directory for CSV and SVG outputs.")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds per sweep point.")
    parser.add_argument("--quick", action="store_true", help="Run a smaller smoke experiment.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        settings = ContinuousExperimentSettings(
            output_dir=args.output,
            seeds=args.seeds,
            user_sweep=(10, 15),
            rb_sweep=(5, 9),
            fixed_users=15,
            fixed_rbs=9,
            entropy_taus=(75.0,),
        )
    else:
        settings = ContinuousExperimentSettings(output_dir=args.output, seeds=args.seeds)

    paths = run_continuous_experiment_suite(settings)
    print(f"Wrote continuous raw results: {paths.raw_csv}")
    print(f"Wrote continuous summary: {paths.summary_csv}")
    print(f"Wrote continuous plots: {paths.objective_plot.parent}")


if __name__ == "__main__":
    main()
