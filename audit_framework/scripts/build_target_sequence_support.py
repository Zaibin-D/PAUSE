"""Build fixed label-free MMseqs2 target-family support for cluster audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audit_framework.data import (
    DEFAULT_DATASETS,
    MMSEQS_MAX_EVALUE,
    MMSEQS_MIN_IDENTITY,
    MMSEQS_MIN_QUERY_COVERAGE,
    TARGET_SEQUENCE_SUPPORT_NAME,
    compute_target_sequence_support_payload,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="datasets")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--split", default="cluster", choices=["cluster"])
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    dataset_root = Path(args.dataset_root).resolve()
    for dataset in args.datasets:
        payload = compute_target_sequence_support_payload(
            str(dataset_root),
            str(dataset),
            str(args.split),
        )
        if payload is None:
            raise RuntimeError(
                f"MMseqs2 target support unavailable for "
                f"{dataset}/{args.split}."
            )
        out_dir = dataset_root / dataset / args.split / "pime"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / TARGET_SEQUENCE_SUPPORT_NAME
        with out_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        distances = payload["target_distance"]
        meta = {
            "dataset": dataset,
            "split": args.split,
            "source_reference": (
                dataset_root
                / dataset
                / args.split
                / "source_train_with_id.csv"
            ).as_posix(),
            "outcome_labels_used": False,
            "metric": "MMseqs2 identity x min(query coverage, target coverage)",
            "minimum_identity_percent": MMSEQS_MIN_IDENTITY,
            "minimum_query_coverage": MMSEQS_MIN_QUERY_COVERAGE,
            "maximum_evalue": MMSEQS_MAX_EVALUE,
            "target_entities": len(distances),
            "family_unseen_entities": int(
                sum(payload["target_family_unseen"].values())
            ),
            "output": out_path.as_posix(),
        }
        out_path.with_suffix(".json").write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )
        print(
            f"[write] {out_path} targets={meta['target_entities']} "
            f"family_unseen={meta['family_unseen_entities']}"
        )


if __name__ == "__main__":
    main()
