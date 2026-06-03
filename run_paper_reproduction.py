from __future__ import annotations

import argparse
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from relaxed_ra.paper_reproduction import PaperReproductionSettings, run_paper_reproduction_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-aligned FL reproduction experiments.")
    parser.add_argument("--output", type=Path, default=Path("results/paper"))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="Run a small smoke experiment.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        settings = PaperReproductionSettings(
            output_dir=args.output,
            seeds=args.seeds,
            user_sweep=(6, 12),
            matching_user_sweep=(5, 10),
            rb_sweep=(6, 12),
            sample_sweep=(10, 30),
            fixed_users=12,
            fixed_rbs=8,
            digit_rounds=4,
            digit_sweep_rounds=4,
            iteration_points=(0, 2, 4),
        )
    else:
        settings = PaperReproductionSettings(output_dir=args.output, seeds=args.seeds)

    paths = run_paper_reproduction_suite(settings)
    print(f"Wrote paper-style raw results: {paths.raw_csv}")
    print(f"Wrote paper-style summary: {paths.summary_csv}")
    print(f"Wrote paper-style plots: {paths.plots[0].parent}")


if __name__ == "__main__":
    main()
