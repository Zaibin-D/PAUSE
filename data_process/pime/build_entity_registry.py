import argparse
from collections import defaultdict

import pandas as pd

from data_process.pime.common import (
    concat_phase_frames,
    pime_dir,
    require_columns,
    update_manifest,
)
from data_process.pime.chem_utils import standardize_smiles
from data_process.pime.schema import PIME_FILES


def _phase_summary(df, id_col):
    phases = defaultdict(list)
    for entity_id, group in df.groupby(id_col):
        phases[int(entity_id)] = sorted(str(v) for v in group["pime_phase"].dropna().unique())
    return phases


def build_entity_registry(dataset, split):
    df = concat_phase_frames(dataset, split)
    require_columns(df, ["SMILES", "Protein", "dr_id", "pr_id"], "entity registry")

    out_dir = pime_dir(dataset, split)
    drug_path = out_dir / PIME_FILES["drug_entity"]
    target_path = out_dir / PIME_FILES["target_entity"]

    drug_phases = _phase_summary(df, "dr_id")
    target_phases = _phase_summary(df, "pr_id")

    drug_rows = []
    for dr_id, group in df[["dr_id", "SMILES"]].drop_duplicates().groupby("dr_id"):
        smiles_values = sorted({str(v).strip() for v in group["SMILES"].dropna() if str(v).strip()})
        smiles = smiles_values[0] if smiles_values else ""
        standardized = standardize_smiles(smiles)
        drug_rows.append(
            {
                "dr_id": int(dr_id),
                "smiles": smiles,
                "canonical_smiles": standardized.canonical_smiles,
                "selected_component_smiles": standardized.selected_component_smiles,
                "num_components": int(standardized.num_components),
                "standardization_note": standardized.standardization_note,
                "smiles_status": standardized.status,
                "phases": "|".join(drug_phases[int(dr_id)]),
            }
        )

    target_rows = []
    for pr_id, group in df[["pr_id", "Protein"]].drop_duplicates().groupby("pr_id"):
        seq_values = sorted({str(v).strip() for v in group["Protein"].dropna() if str(v).strip()})
        sequence = seq_values[0] if seq_values else ""
        target_rows.append(
            {
                "pr_id": int(pr_id),
                "protein_sequence": sequence,
                "sequence_length": len(sequence),
                "sequence_status": "ok" if sequence else "missing",
                "phases": "|".join(target_phases[int(pr_id)]),
            }
        )

    drug_df = pd.DataFrame(drug_rows).sort_values("dr_id").reset_index(drop=True)
    target_df = pd.DataFrame(target_rows).sort_values("pr_id").reset_index(drop=True)
    drug_df.to_csv(drug_path, index=False)
    target_df.to_csv(target_path, index=False)

    update_manifest(
        dataset,
        split,
        "entity_registry",
        {
            "files": [str(drug_path.relative_to(out_dir)), str(target_path.relative_to(out_dir))],
            "num_drugs": int(len(drug_df)),
            "num_targets": int(len(target_df)),
            "invalid_or_missing_smiles": int((drug_df["smiles_status"] != "ok").sum()),
            "missing_sequences": int((target_df["sequence_status"] != "ok").sum()),
            "channel_permission": "index_only",
            "leakage_risk": "low",
        },
    )
    return drug_df, target_df


def main():
    parser = argparse.ArgumentParser(description="Build PIME drug/target entity registries.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="cluster", choices=["cluster", "random"])
    args = parser.parse_args()
    build_entity_registry(args.dataset, args.split)


if __name__ == "__main__":
    main()
