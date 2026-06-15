"""Analyze fixed HGB(U+P+E) heterogeneity from frozen result CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audit_framework.hgb_heterogeneity import run_hgb_heterogeneity_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    merged = (
        "audit_framework/cache/diagnostic_results/"
        "strengthening_experiments/merged"
    )
    parser.add_argument(
        "--candidate-path",
        default=f"{merged}/candidate_budget_runs.csv",
    )
    parser.add_argument(
        "--selected-path",
        default=f"{merged}/selected_budget_runs.csv",
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "audit_framework/cache/diagnostic_results/"
            "strengthening_experiments/hgb_heterogeneity"
        ),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--precision", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_hgb_heterogeneity_analysis(
        candidate_path=args.candidate_path,
        selected_path=args.selected_path,
        out_dir=args.out_dir,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        precision=args.precision,
    )
    print(f"[hgb-heterogeneity] complete: {report}", flush=True)


if __name__ == "__main__":
    main()
