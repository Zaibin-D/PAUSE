"""Run the three predeclared PAUSE strengthening experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audit_framework.strengthening import (
    build_strengthening_shards,
    merge_and_summarize_strengthening,
    run_cross_domain_transfer,
    run_strengthening_shard,
    validate_strengthening_shard,
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
    parser.add_argument(
        "--calibration-root",
        default="audit_framework/cache/validation_audits",
    )
    parser.add_argument("--dataset-root", default="datasets")
    parser.add_argument(
        "--shard-root",
        default=(
            "audit_framework/cache/diagnostic_results/"
            "strengthening_experiments/shards"
        ),
    )
    parser.add_argument(
        "--merged-dir",
        default=(
            "audit_framework/cache/diagnostic_results/"
            "strengthening_experiments/merged"
        ),
    )
    parser.add_argument(
        "--transfer-dir",
        default=(
            "audit_framework/cache/diagnostic_results/"
            "strengthening_experiments/cross_domain_transfer"
        ),
    )
    parser.add_argument(
        "--formal-dir",
        default="audit_framework/results/audit",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--precision", type=int, default=12)
    parser.add_argument(
        "--merge-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--transfer-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def run(args: argparse.Namespace) -> None:
    if args.transfer_only:
        run_cross_domain_transfer(
            formal_dir=args.formal_dir,
            out_dir=args.transfer_dir,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
            precision=args.precision,
        )
        return

    shards = build_strengthening_shards(
        models=args.models,
        datasets=args.datasets,
        shard_root=args.shard_root,
    )
    if not args.merge_only:
        for shard in shards:
            try:
                report = validate_strengthening_shard(shard)
            except (FileNotFoundError, RuntimeError, ValueError):
                report = None
            if report is not None and (shard.out_dir / "COMPLETE").exists():
                print(
                    f"[strengthening] skip complete "
                    f"{shard.model_key}/{shard.dataset}: {report}",
                    flush=True,
                )
                continue
            run_strengthening_shard(
                shard,
                calibration_root=args.calibration_root,
                dataset_root=args.dataset_root,
                precision=args.precision,
            )
            report = validate_strengthening_shard(shard)
            print(
                f"[strengthening] validated "
                f"{shard.model_key}/{shard.dataset}: {report}",
                flush=True,
            )

    merge_and_summarize_strengthening(
        shards,
        merged_dir=args.merged_dir,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        precision=args.precision,
    )
    run_cross_domain_transfer(
        formal_dir=args.formal_dir,
        out_dir=args.transfer_dir,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed + 10_000,
        precision=args.precision,
    )


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
