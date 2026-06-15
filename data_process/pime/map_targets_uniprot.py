import argparse
from pathlib import Path

import pandas as pd

from data_process.pime.common import pime_dir, update_manifest
from data_process.pime.schema import PIME_FILES


def export_target_fasta(dataset, split):
    out_dir = pime_dir(dataset, split)
    entity_path = out_dir / PIME_FILES["target_entity"]
    if not entity_path.exists():
        raise FileNotFoundError(f"Run build_entity_registry first: {entity_path}")
    target_df = pd.read_csv(entity_path)
    fasta_path = out_dir / "target_sequences_for_uniprot_mapping.fasta"
    with fasta_path.open("w", encoding="utf-8") as f:
        for row in target_df.itertuples(index=False):
            sequence = str(row.protein_sequence or "").strip()
            if not sequence:
                continue
            f.write(f">pr_id={int(row.pr_id)}\n")
            for idx in range(0, len(sequence), 80):
                f.write(sequence[idx: idx + 80] + "\n")

    update_manifest(
        dataset,
        split,
        "target_uniprot_fasta",
        {
            "file": str(fasta_path.relative_to(out_dir)),
            "num_targets": int(len(target_df)),
            "channel_permission": "index_only",
            "leakage_risk": "low",
            "note": "Use this FASTA with UniProt BLAST, MMseqs2, or an institutional mapping workflow; import the resulting accession table with this script.",
        },
    )
    return fasta_path


def import_uniprot_mapping(dataset, split, mapping_csv):
    out_dir = pime_dir(dataset, split)
    mapping_csv = Path(mapping_csv)
    mapping_df = pd.read_csv(mapping_csv)
    required = {"pr_id", "uniprot_id"}
    missing = required - set(mapping_df.columns)
    if missing:
        raise ValueError(f"Mapping CSV missing required columns: {sorted(missing)}")

    keep_cols = [
        col
        for col in [
            "pr_id",
            "uniprot_id",
            "mapping_method",
            "identity",
            "coverage",
            "evalue",
            "reviewed",
            "organism",
            "note",
        ]
        if col in mapping_df.columns
    ]
    out = mapping_df[keep_cols].copy()
    out["pr_id"] = out["pr_id"].astype(int)
    out["uniprot_id"] = out["uniprot_id"].astype(str).str.strip()
    out = out[out["uniprot_id"] != ""].drop_duplicates(subset=["pr_id", "uniprot_id"])
    out_path = out_dir / "target_uniprot_map.csv"
    out.to_csv(out_path, index=False)

    target_path = out_dir / PIME_FILES["target_entity"]
    num_targets = len(pd.read_csv(target_path)) if target_path.exists() else 0
    mapped_targets = out["pr_id"].nunique()
    update_manifest(
        dataset,
        split,
        "target_uniprot_map",
        {
            "file": str(out_path.relative_to(out_dir)),
            "source_file": str(mapping_csv),
            "num_targets": int(num_targets),
            "num_mapped_targets": int(mapped_targets),
            "coverage": float(mapped_targets / num_targets) if num_targets else 0.0,
            "channel_permission": "diagnostic_index",
            "leakage_risk": "low",
            "note": "UniProt accessions support label-free target-family diagnostics.",
        },
    )
    return out


def main():
    parser = argparse.ArgumentParser(description="Export/import target to UniProt mapping for PIME.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="cluster", choices=["cluster"])
    parser.add_argument("--export-fasta", action="store_true")
    parser.add_argument("--mapping-csv", default=None, help="CSV with pr_id,uniprot_id and optional quality columns.")
    args = parser.parse_args()
    if args.export_fasta:
        export_target_fasta(args.dataset, args.split)
    if args.mapping_csv:
        import_uniprot_mapping(args.dataset, args.split, args.mapping_csv)
    if not args.export_fasta and not args.mapping_csv:
        raise SystemExit("Specify --export-fasta and/or --mapping-csv.")


if __name__ == "__main__":
    main()
