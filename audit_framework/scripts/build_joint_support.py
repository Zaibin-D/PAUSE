"""Build fixed model-independent drug-conditioned source-pair support."""

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
    DIRECT_TARGET_SUPPORT_NAME,
    JOINT_SUPPORT_NAME,
    SUPPORT_NEIGHBOURS,
    compute_joint_support_payload,
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
    try:
        from rdkit import rdBase
    except ImportError as error:
        raise RuntimeError("RDKit is required to build joint support.") from error

    for dataset in args.datasets:
        direct_path = (
            dataset_root
            / dataset
            / args.split
            / "pime"
            / DIRECT_TARGET_SUPPORT_NAME
        )
        if not direct_path.exists():
            raise FileNotFoundError(
                f"Build direct target support first: {direct_path}"
            )
        compute_joint_support_payload.cache_clear()
        payload = compute_joint_support_payload(
            str(dataset_root),
            str(dataset),
            str(args.split),
        )
        if payload is None:
            raise RuntimeError(
                f"Joint support unavailable for {dataset}/{args.split}."
            )
        out_path = direct_path.with_name(JOINT_SUPPORT_NAME)
        with out_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        metadata = {
            "dataset": dataset,
            "split": args.split,
            "outcome_labels_used": False,
            "source_reference": (
                dataset_root
                / dataset
                / args.split
                / "source_train_with_id.csv"
            ).as_posix(),
            "direct_target_support": direct_path.as_posix(),
            "output": out_path.as_posix(),
            "neighbours": SUPPORT_NEIGHBOURS,
            "drug_metric": payload["drug_metric"],
            "target_metric": payload["target_metric"],
            "joint_metric": (
                "1-max(drug_similarity*target_similarity) over observed "
                "source-train neighbour pairs"
            ),
            "exact_query_pair_excluded": True,
            "rdkit_version": rdBase.rdkitVersion,
            "drug_entities": len(payload["drug_neighbours"]),
            "target_entities": len(payload["target_neighbours"]),
            "source_pairs": len(payload["source_pairs"]),
        }
        out_path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        print(
            f"[write] {out_path} drugs={metadata['drug_entities']} "
            f"targets={metadata['target_entities']} "
            f"pairs={metadata['source_pairs']}"
        )


if __name__ == "__main__":
    main()
