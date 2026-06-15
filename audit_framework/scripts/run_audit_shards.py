"""Run, validate, merge, and summarize the nine fixed PAUSE audit shards."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audit_framework.sharding import (
    build_specs,
    merge_shards,
    run_shard,
    summarize_merged,
    write_diagnostic_analysis,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["pace", "tapb", "drugban"],
        default=["pace", "tapb", "drugban"],
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["biosnap", "bindingdb", "human"],
        default=["biosnap", "bindingdb", "human"],
    )
    parser.add_argument("--seeds", nargs="+", default=["4", "5", "6", "7", "8"])
    parser.add_argument(
        "--group-axes",
        nargs="+",
        choices=["target", "drug"],
        default=["target", "drug"],
    )
    parser.add_argument(
        "--confidence-source",
        choices=["base", "calibrated"],
        default="base",
    )
    parser.add_argument(
        "--confidence-thresholds",
        nargs="+",
        type=float,
        default=[0.8, 0.9],
    )
    parser.add_argument(
        "--uncertainty-reject-fractions",
        nargs="+",
        type=float,
        default=[0.10, 0.20],
    )
    parser.add_argument("--review-fraction", type=float, default=0.20)
    parser.add_argument(
        "--calibration-root",
        default="audit_framework/cache/validation_audits",
    )
    parser.add_argument("--dataset-root", default="datasets")
    parser.add_argument(
        "--shard-root",
        default=(
            "audit_framework/cache/diagnostic_results/"
            "general_target_joint_shards"
        ),
    )
    parser.add_argument(
        "--merged-dir",
        default=(
            "audit_framework/cache/diagnostic_results/"
            "general_target_joint_merged"
        ),
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
    )
    parser.add_argument("--shard-bootstrap-resamples", type=int, default=20)
    parser.add_argument("--final-bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--precision", type=int, default=12)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--run-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run and validate the selected shards without merging.",
    )
    mode.add_argument(
        "--merge-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Validate and merge existing shards without running them.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    specs = build_specs(
        models=args.models,
        datasets=args.datasets,
        shard_root=args.shard_root,
    )
    validation_kwargs = {
        "seeds": list(args.seeds),
        "group_axes": list(args.group_axes),
        "confidence_source": args.confidence_source,
        "confidence_thresholds": list(args.confidence_thresholds),
        "reject_fractions": list(args.uncertainty_reject_fractions),
        "review_fraction": args.review_fraction,
    }
    if not args.merge_only:
        for spec in specs:
            run_shard(
                spec,
                python_executable=args.python_executable,
                calibration_root=args.calibration_root,
                dataset_root=args.dataset_root,
                seeds=list(args.seeds),
                group_axes=list(args.group_axes),
                confidence_source=args.confidence_source,
                confidence_thresholds=list(args.confidence_thresholds),
                reject_fractions=list(
                    args.uncertainty_reject_fractions
                ),
                review_fraction=args.review_fraction,
                shard_bootstrap_resamples=(
                    args.shard_bootstrap_resamples
                ),
                precision=args.precision,
            )
    if args.run_only:
        return
    merge_shards(
        specs,
        merged_dir=args.merged_dir,
        precision=args.precision,
        validation_kwargs=validation_kwargs,
    )
    summarize_merged(
        python_executable=args.python_executable,
        merged_dir=args.merged_dir,
        datasets=list(args.datasets),
        bootstrap_resamples=args.final_bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        precision=args.precision,
    )
    write_diagnostic_analysis(
        merged_dir=args.merged_dir,
        bootstrap_resamples=args.final_bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed + 60_000,
        precision=args.precision,
    )


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
