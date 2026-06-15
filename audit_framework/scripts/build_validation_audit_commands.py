"""Build or run validation-set PAUSE input-export commands.

The existing diagnostics directory contains test-set audit CSVs. This helper
reads those CSV headers to recover the exact data split, config, score branch,
and checkpoint, then emits matching validation-set audit commands under the
streamlined audit-framework cache.
"""

from __future__ import annotations

import argparse
import json
import os
import glob
import re
import subprocess
from pathlib import Path

import pandas as pd
import yaml


AUDIT_NAME = "pause_audit_inputs.csv"
SUMMARY_NAME = "pause_audit_inputs_summary.csv"
DEFAULT_TEST_ROOTS = (
    "audit_framework/cache/test_audits/pace",
    "audit_framework/cache/test_audits/tapb",
    "audit_framework/cache/test_audits/drugban",
)
DEFAULT_DATASETS = ("biosnap", "bindingdb", "human")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build validation-set audit commands from existing test audit CSVs."
    )
    parser.add_argument("--test-roots", nargs="+", default=list(DEFAULT_TEST_ROOTS))
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--out-root", default="audit_framework/cache/validation_audits")
    parser.add_argument("--phase", default="val", choices=["train", "val", "test"])
    parser.add_argument("--root", default="./datasets")
    parser.add_argument("--model-type", default="pime")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--diagnostic-seed", type=int, default=2026)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--python",
        default="python",
        help="Python executable used for audit jobs. Default uses the active shell environment.",
    )
    parser.add_argument("--command-file", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based command index to start executing from.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum commands to execute after writing the manifest.")
    parser.add_argument("--keep-going", action="store_true", help="Continue executing later commands if one command fails.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def root_name(path: Path) -> str:
    return path.name


def seed_from_dir(path: Path) -> str:
    return path.parent.name


def parse_config_value(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    matches = re.findall(r"PosixPath\('([^']+)'\)", text)
    if matches:
        return ",".join(matches)
    matches = re.findall(r"WindowsPath\('([^']+)'\)", text)
    if matches:
        return ",".join(matches)
    if text.startswith("[") and text.endswith("]"):
        stripped = text.strip("[]")
        parts = [part.strip().strip("'\"") for part in stripped.split(",") if part.strip()]
        return ",".join(parts)
    return text


def split_config_paths(config: str) -> list[str]:
    return [part.strip() for part in str(config or "").split(",") if part.strip()]


def seed_number(run_id: str) -> str:
    match = re.search(r"(\d+)$", str(run_id))
    if not match:
        raise ValueError(f"Could not infer numeric seed from run id: {run_id}")
    return match.group(1)


def load_external_checkpoint_pattern(config_path: str) -> str:
    if not str(config_path or "").strip():
        return ""
    path = Path(config_path)
    if not path.exists() or not path.is_file():
        return ""
    with path.open("r", encoding="utf-8") as fp:
        loaded = yaml.safe_load(fp) or {}
    base = loaded.get("MODEL", {}).get("BASE", {})
    return str(base.get("EXTERNAL_CHECKPOINT", "") or "").strip()


def resolve_seed_checkpoint_pattern(pattern: str, run_id: str) -> str:
    if not pattern:
        return ""
    value = str(pattern)
    seed = seed_number(run_id)
    value = value.replace("${TAPB_SEED}", seed).replace("${DRUGBAN_SEED}", seed)
    value = re.sub(r"\$\{[A-Z0-9_]*SEED\}", seed, value)
    value = os.path.expandvars(os.path.expanduser(value))
    return value


def write_checkpoint_override(args: argparse.Namespace, source_root: Path, dataset: str, run_id: str, config: str) -> str:
    override = ""
    for config_path in split_config_paths(config):
        pattern = load_external_checkpoint_pattern(config_path)
        resolved = resolve_seed_checkpoint_pattern(pattern, run_id)
        if not resolved or resolved == pattern:
            continue
        override_path = (
            Path(args.out_root)
            / "runtime_configs"
            / root_name(source_root)
            / dataset
            / run_id
            / "external_checkpoint.yaml"
        )
        override_path.parent.mkdir(parents=True, exist_ok=True)
        override_data = {
            "MODEL": {
                "BASE": {
                    "EXTERNAL_CHECKPOINT": resolved,
                }
            }
        }
        with override_path.open("w", encoding="utf-8") as fp:
            yaml.safe_dump(override_data, fp, sort_keys=False)
        override = override_path.as_posix()
        break
    return override


def checkpoint_pattern_has_match(pattern: str) -> bool:
    if not pattern:
        return False
    path = Path(pattern)
    if any(char in pattern for char in "*?[]"):
        return bool(glob.glob(pattern))
    return path.exists()


def discover(root: Path, datasets: list[str]) -> list[Path]:
    paths = []
    for dataset in datasets:
        paths.extend(sorted((root / dataset).glob(f"seed_*/{AUDIT_NAME}")))
    return paths


def read_first_row(path: Path) -> dict[str, object]:
    return pd.read_csv(path, nrows=1).iloc[0].to_dict()


def build_command(args: argparse.Namespace, audit_path: Path) -> dict[str, object]:
    first = read_first_row(audit_path)
    source_root = audit_path.parents[2]
    dataset = str(first.get("data") or audit_path.parents[1].name)
    split = str(first.get("split") or "cluster")
    seed = seed_from_dir(audit_path)
    out_dir = Path(args.out_root) / root_name(source_root) / dataset / seed
    out_csv = out_dir / AUDIT_NAME
    summary_csv = out_dir / SUMMARY_NAME
    checkpoint = str(first.get("checkpoint") or "").strip()
    checkpoint_exists = bool(checkpoint) and Path(checkpoint).exists()
    config = parse_config_value(first.get("config"))
    override_config = write_checkpoint_override(args, source_root, dataset, seed, config)
    external_override_pattern = load_external_checkpoint_pattern(override_config)
    if override_config:
        config = f"{config},{override_config}" if config else override_config
    cmd = [
        args.python,
        "diagnostics/export_audit_inputs.py",
        "--data",
        dataset,
        "--split",
        split,
        "--phase",
        args.phase,
        "--root",
        args.root,
        "--config",
        config,
        "--model-type",
        args.model_type,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.diagnostic_seed),
        "--progress-every",
        str(args.progress_every),
        "--out",
        str(out_csv),
        "--summary",
        str(summary_csv),
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    if checkpoint:
        cmd.extend(["--checkpoint", checkpoint])
    return {
        "source_root": str(source_root),
        "source_audit": str(audit_path),
        "audit_root": root_name(source_root),
        "dataset": dataset,
        "seed": seed,
        "split": split,
        "phase": args.phase,
        "score": "base",
        "config": config,
        "checkpoint": checkpoint,
        "checkpoint_exists": checkpoint_exists,
        "external_checkpoint_override": override_config,
        "external_checkpoint_exists": checkpoint_pattern_has_match(external_override_pattern),
        "out_csv": str(out_csv),
        "summary_csv": str(summary_csv),
        "exists": out_csv.exists() and summary_csv.exists(),
        "argv_json": json.dumps(cmd),
        "command": subprocess.list2cmdline(cmd),
    }


def run_command(row: dict[str, object], force: bool) -> bool:
    out_csv = Path(str(row["out_csv"]))
    summary_csv = Path(str(row["summary_csv"]))
    if not bool(row.get("checkpoint_exists")):
        print(f"[skip] missing checkpoint: {row.get('checkpoint')}")
        return True
    if out_csv.exists() and summary_csv.exists() and not force:
        print(f"[skip] exists: {out_csv}")
        return True
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = json.loads(str(row["argv_json"]))
    print(f"[run] {subprocess.list2cmdline(cmd)}", flush=True)
    completed = subprocess.run(cmd, check=False)
    if completed.returncode:
        print(f"[fail] exit={completed.returncode}: {out_csv}", flush=True)
        return False
    return True


def main() -> None:
    args = parse_args()
    rows = []
    for root_arg in args.test_roots:
        root = Path(root_arg)
        if not root.exists():
            print(f"[skip] missing root: {root}")
            continue
        for audit_path in discover(root, args.datasets):
            rows.append(build_command(args, audit_path))
    table = pd.DataFrame(rows)
    command_file = Path(args.command_file) if args.command_file else Path(args.out_root) / "validation_audit_commands.csv"
    command_file.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(command_file, index=False)
    print(f"[write] {command_file}")
    print(f"[commands] {len(table)}")
    if args.execute:
        start = max(0, int(args.start_index or 0))
        execute_rows = rows[start:]
        if args.limit is not None:
            execute_rows = execute_rows[: args.limit]
        for row in execute_rows:
            ok = run_command(row, args.force)
            if not ok and not args.keep_going:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
