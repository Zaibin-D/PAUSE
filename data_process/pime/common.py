import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data_process.pime.schema import PIME_FILES, SPLIT_FILES, FEATURE_SPECS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = Path(os.environ.get("PASMU_DATASET_ROOT", PROJECT_ROOT / "datasets")).expanduser()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def dataset_dir(dataset, split):
    return DATASET_ROOT / dataset / split


def pime_dir(dataset, split):
    path = dataset_dir(dataset, split) / "pime"
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_phase_frames(dataset, split):
    base = dataset_dir(dataset, split)
    if split not in SPLIT_FILES:
        raise ValueError(f"Unsupported split={split!r}; expected one of {sorted(SPLIT_FILES)}")

    frames = {}
    for phase, filename in SPLIT_FILES[split].items():
        path = base / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["pime_phase"] = phase
        frames[phase] = df
    if not frames:
        raise FileNotFoundError(f"No split CSV files found under {base}")
    return frames


def concat_phase_frames(dataset, split):
    frames = read_phase_frames(dataset, split)
    return pd.concat(frames.values(), ignore_index=True)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_pickle(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def read_pickle(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def manifest_path(dataset, split):
    return pime_dir(dataset, split) / PIME_FILES["manifest"]


def load_manifest(dataset, split):
    path = manifest_path(dataset, split)
    if path.exists():
        return read_json(path, default={})
    return {
        "schema_version": "pime-evidence-store-v1",
        "dataset": dataset,
        "split": split,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "feature_specs": [
            {
                "name": spec.name,
                "channel_permission": spec.channel_permission.value,
                "leakage_risk": spec.leakage_risk.value,
                "source": spec.source,
                "description": spec.description,
            }
            for spec in FEATURE_SPECS
        ],
        "artifacts": {},
    }


def update_manifest(dataset, split, artifact_name, payload):
    manifest = load_manifest(dataset, split)
    manifest["updated_at"] = utc_now()
    manifest.setdefault("artifacts", {})[artifact_name] = {
        "updated_at": utc_now(),
        **payload,
    }
    write_json(manifest_path(dataset, split), manifest)
    return manifest


def require_columns(df, columns, context):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")
