"""Build fixed label-free native source-domain support for cluster audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audit_framework.data import (
    DEFAULT_DATASETS,
    NATIVE_SUPPORT_NAME,
    compute_native_distance_payload,
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
        payload = compute_native_distance_payload(
            str(dataset_root),
            str(dataset),
            str(args.split),
        )
        if payload is None:
            raise RuntimeError(
                f"Native support unavailable for {dataset}/{args.split}. "
                "RDKit and the frozen ESM feature assets are required."
            )
        out_dir = dataset_root / dataset / args.split / "pime"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / NATIVE_SUPPORT_NAME
        with out_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            from rdkit import rdBase

            rdkit_version = rdBase.rdkitVersion
        except ImportError:
            rdkit_version = "unavailable"
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
            "drug_metric": "Morgan radius=2, 2048-bit, Tanimoto",
            "target_metric": "frozen ESM CLS cosine",
            "neighbours": 5,
            "rdkit_version": rdkit_version,
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
            "drug_entities": len(payload["drug_nearest"]),
            "target_entities": len(payload["target_nearest"]),
            "output": out_path.as_posix(),
        }
        meta_path = out_path.with_suffix(".json")
        meta_path.write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )
        print(
            f"[write] {out_path} "
            f"drugs={meta['drug_entities']} targets={meta['target_entities']}"
        )


if __name__ == "__main__":
    main()
