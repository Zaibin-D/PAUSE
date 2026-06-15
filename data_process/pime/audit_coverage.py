import argparse
from pathlib import Path

import pandas as pd

from data_process.pime.common import (
    pime_dir,
    read_pickle,
    utc_now,
    write_json,
)
from data_process.pime.schema import FEATURE_SPECS, PIME_FILES


CORE_ASSETS = (
    "drug_entity.csv",
    "target_entity.csv",
    "drug_prior_feat.pkl",
    "target_prior_feat.pkl",
    "native_source_support.pkl",
)

DIAGNOSTIC_ASSETS = (
    "target_sequence_support.pkl",
    "direct_target_support.pkl",
    "joint_source_support.pkl",
)


def _pickle_entries(path):
    if not path.exists():
        return 0
    payload = read_pickle(path)
    return len(payload) if hasattr(payload, "__len__") else 0


def audit_coverage(dataset, split):
    out_dir = pime_dir(dataset, split)
    drug_path = out_dir / PIME_FILES["drug_entity"]
    target_path = out_dir / PIME_FILES["target_entity"]
    drug_table = (
        pd.read_csv(drug_path)
        if drug_path.exists()
        else pd.DataFrame()
    )
    target_table = (
        pd.read_csv(target_path)
        if target_path.exists()
        else pd.DataFrame()
    )

    assets = {
        name: {
            "present": (out_dir / name).exists(),
            "bytes": (
                int((out_dir / name).stat().st_size)
                if (out_dir / name).is_file()
                else 0
            ),
        }
        for name in (*CORE_ASSETS, *DIAGNOSTIC_ASSETS)
    }
    stats = {
        "dataset": dataset,
        "split": split,
        "generated_at": utc_now(),
        "num_drugs": int(len(drug_table)),
        "num_targets": int(len(target_table)),
        "drug_prior_entries": _pickle_entries(
            out_dir / PIME_FILES["drug_prior_feat"]
        ),
        "target_prior_entries": _pickle_entries(
            out_dir / PIME_FILES["target_prior_feat"]
        ),
        "core_complete": all(
            assets[name]["present"] for name in CORE_ASSETS
        ),
        "assets": assets,
    }

    report_lines = [
        f"# PAUSE Evidence Store Audit: {dataset}/{split}",
        "",
        f"- Drugs: {stats['num_drugs']}",
        f"- Targets: {stats['num_targets']}",
        f"- Drug prior entries: {stats['drug_prior_entries']}",
        f"- Target prior entries: {stats['target_prior_entries']}",
        f"- Core complete: {stats['core_complete']}",
        "",
        "## Assets",
        "",
    ]
    report_lines.extend(
        f"- `{name}`: {'present' if value['present'] else 'missing'}"
        for name, value in assets.items()
    )

    report_path = out_dir / PIME_FILES["audit_report"]
    stats_path = out_dir / PIME_FILES["audit_stats"]
    manifest_path = out_dir / PIME_FILES["manifest"]
    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    write_json(stats_path, stats)
    write_json(
        manifest_path,
        {
            "schema_version": "pause-evidence-store-v2",
            "dataset": dataset,
            "split": split,
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
            "core_assets": list(CORE_ASSETS),
            "fixed_diagnostic_assets": list(DIAGNOSTIC_ASSETS),
            "audit": {
                "report": report_path.name,
                "stats": stats_path.name,
                "core_complete": stats["core_complete"],
            },
        },
    )
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Audit the PAUSE prior and support evidence store."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="cluster", choices=["cluster"])
    args = parser.parse_args()
    audit_coverage(args.dataset, args.split)


if __name__ == "__main__":
    main()
