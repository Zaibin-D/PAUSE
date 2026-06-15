"""Build fixed direct target-to-source MMseqs2 support for PAUSE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import re
import subprocess
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audit_framework.data import (
    DEFAULT_DATASETS,
    DIRECT_TARGET_HITS_NAME,
    DIRECT_TARGET_SUPPORT_NAME,
    SUPPORT_NEIGHBOURS,
    compute_direct_target_support_payload,
)


MMSEQS_FORMAT = (
    "query,target,pident,alnlen,qstart,qend,tstart,tend,"
    "evalue,bits,qlen,tlen"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="datasets")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--split", default="cluster", choices=["cluster"])
    parser.add_argument("--mmseqs", default="mmseqs")
    parser.add_argument(
        "--docker-image",
        default=None,
        help=(
            "Optional MMseqs2 container image. When set, the dataset root is "
            "mounted read-write at /data and --mmseqs is ignored."
        ),
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--reuse-hits",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Parse an existing fixed hits TSV instead of invoking MMseqs2.",
    )
    return parser


def _canonical_id(value: object) -> str:
    text = str(value).strip()
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    return str(int(number)) if number.is_integer() else text


def _clean_sequence(value: object) -> str:
    sequence = re.sub(r"\s+", "", str(value)).upper()
    return sequence if sequence and sequence != "NAN" else ""


def _write_fastas(
    dataset_root: Path,
    dataset: str,
    split: str,
    mmseqs_dir: Path,
) -> tuple[Path, Path, int, int]:
    split_dir = dataset_root / dataset / split
    entity_path = split_dir / "pime" / "target_entity.csv"
    source_path = split_dir / "source_train_with_id.csv"
    entities = pd.read_csv(
        entity_path,
        usecols=["pr_id", "protein_sequence"],
    )
    source_ids = set(
        pd.read_csv(source_path, usecols=["pr_id"])["pr_id"].map(
            _canonical_id
        )
    )

    query_path = mmseqs_dir / "all_targets.fasta"
    source_fasta_path = mmseqs_dir / "source_targets.fasta"
    query_records: list[str] = []
    source_records: list[str] = []
    seen: set[str] = set()
    for entity_id, raw_sequence in entities.itertuples(index=False, name=None):
        key = _canonical_id(entity_id)
        sequence = _clean_sequence(raw_sequence)
        if not key or not sequence or key in seen:
            continue
        seen.add(key)
        record = f">pr_id={key}\n{sequence}\n"
        query_records.append(record)
        if key in source_ids:
            source_records.append(record)
    if not query_records or not source_records:
        raise RuntimeError(
            f"No usable target sequences for {dataset}/{split}: "
            f"queries={len(query_records)} source={len(source_records)}"
        )
    query_path.write_text("".join(query_records), encoding="ascii")
    source_fasta_path.write_text("".join(source_records), encoding="ascii")
    return (
        query_path,
        source_fasta_path,
        len(query_records),
        len(source_records),
    )


def _mmseqs_version(executable: str) -> str:
    result = subprocess.run(
        [executable, "version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return (result.stdout or result.stderr).strip()


def _docker_mmseqs_version(image: str) -> str:
    result = subprocess.run(
        ["docker", "run", "--rm", image, "version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return (result.stdout or result.stderr).strip()


def _container_path(path: Path, dataset_root: Path) -> str:
    return "/data/" + path.resolve().relative_to(dataset_root).as_posix()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    dataset_root = Path(args.dataset_root).resolve()
    version = "not_invoked"
    if not args.reuse_hits:
        version = (
            _docker_mmseqs_version(args.docker_image)
            if args.docker_image
            else _mmseqs_version(args.mmseqs)
        )

    for dataset in args.datasets:
        pime_dir = dataset_root / dataset / args.split / "pime"
        mmseqs_dir = pime_dir / "mmseqs"
        mmseqs_dir.mkdir(parents=True, exist_ok=True)
        query_path, source_path, query_n, source_n = _write_fastas(
            dataset_root,
            str(dataset),
            str(args.split),
            mmseqs_dir,
        )
        hits_path = mmseqs_dir / DIRECT_TARGET_HITS_NAME
        native_command = [
            args.mmseqs,
            "easy-search",
            str(query_path),
            str(source_path),
            str(hits_path),
            str(mmseqs_dir / "direct_tmp"),
            "--max-seqs",
            str(SUPPORT_NEIGHBOURS),
            "--threads",
            str(args.threads),
            "--format-output",
            MMSEQS_FORMAT,
        ]
        if args.docker_image:
            command = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{dataset_root}:/data",
                args.docker_image,
                "easy-search",
                _container_path(query_path, dataset_root),
                _container_path(source_path, dataset_root),
                _container_path(hits_path, dataset_root),
                _container_path(mmseqs_dir / "direct_tmp", dataset_root),
                "--max-seqs",
                str(SUPPORT_NEIGHBOURS),
                "--threads",
                str(args.threads),
                "--format-output",
                MMSEQS_FORMAT,
            ]
        else:
            command = native_command
        if not args.reuse_hits:
            subprocess.run(command, check=True)
        elif not hits_path.exists():
            raise FileNotFoundError(hits_path)

        compute_direct_target_support_payload.cache_clear()
        payload = compute_direct_target_support_payload(
            str(dataset_root),
            str(dataset),
            str(args.split),
        )
        if payload is None:
            raise RuntimeError(
                f"Direct target support unavailable for {dataset}/{args.split}."
            )
        out_path = pime_dir / DIRECT_TARGET_SUPPORT_NAME
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
            "query_fasta": query_path.as_posix(),
            "source_fasta": source_path.as_posix(),
            "hits": hits_path.as_posix(),
            "output": out_path.as_posix(),
            "mmseqs_version": version,
            "execution_backend": (
                f"docker:{args.docker_image}"
                if args.docker_image
                else f"native:{args.mmseqs}"
            ),
            "command": command,
            "neighbours": SUPPORT_NEIGHBOURS,
            "score": payload["score_definition"],
            "missing_hits_padded_with_zero": True,
            "query_targets": query_n,
            "source_targets": source_n,
            "supported_targets": int(
                sum(
                    distance < 1.0
                    for distance in payload["target_nearest"].values()
                )
            ),
        }
        out_path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        print(
            f"[write] {out_path} queries={query_n} source={source_n} "
            f"supported={metadata['supported_targets']}"
        )


if __name__ == "__main__":
    main()
