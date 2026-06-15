import argparse

import numpy as np
import pandas as pd

from data_process.pime.common import dataset_dir, pime_dir, read_pickle, update_manifest, write_json, write_pickle
from data_process.pime.schema import PIME_FILES


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def _sequence_global_features(sequence):
    seq = str(sequence or "").strip().upper()
    length = max(len(seq), 1)
    counts = np.asarray([seq.count(aa) / length for aa in AMINO_ACIDS], dtype=np.float32)
    biochemical = np.asarray(
        [
            len(seq),
            sum(seq.count(aa) for aa in "DE") / length,
            sum(seq.count(aa) for aa in "KRH") / length,
            sum(seq.count(aa) for aa in "AILMFWVY") / length,
            sum(seq.count(aa) for aa in "STNQ") / length,
            sum(seq.count(aa) for aa in "FWYH") / length,
            seq.count("C") / length,
            seq.count("P") / length,
        ],
        dtype=np.float32,
    )
    return np.concatenate([biochemical, counts]).astype(np.float32)


def build_target_prior(dataset, split):
    out_dir = pime_dir(dataset, split)
    entity_path = out_dir / PIME_FILES["target_entity"]
    if not entity_path.exists():
        raise FileNotFoundError(f"Run build_entity_registry first: {entity_path}")
    entity_df = pd.read_csv(entity_path)

    cls_path = dataset_dir(dataset, split) / "prot_cls_feat.pkl"
    prot_cls = read_pickle(cls_path) if cls_path.exists() else {}

    seq_feature_names = [
        "sequence_length",
        "acidic_fraction",
        "basic_fraction",
        "hydrophobic_fraction",
        "polar_fraction",
        "aromatic_fraction",
        "cysteine_fraction",
        "proline_fraction",
    ] + [f"aa_comp_{aa}" for aa in AMINO_ACIDS]

    payload = {}
    rows = []
    for row in entity_df.itertuples(index=False):
        seq_vec = _sequence_global_features(row.protein_sequence)
        cls_vec = prot_cls.get(int(row.pr_id))
        if cls_vec is not None:
            cls_vec = np.asarray(cls_vec, dtype=np.float32).reshape(-1)
            vec = np.concatenate([cls_vec, seq_vec]).astype(np.float32)
            status = "ok"
            cls_dim = int(cls_vec.shape[0])
        else:
            vec = seq_vec.astype(np.float32)
            status = "sequence_only_missing_esm_cls"
            cls_dim = 0
        payload[int(row.pr_id)] = vec
        rows.append(
            {
                "pr_id": int(row.pr_id),
                "status": status,
                "feature_dim": int(vec.shape[0]),
                "esm_cls_dim": cls_dim,
                "sequence_feature_dim": int(seq_vec.shape[0]),
            }
        )

    table = pd.DataFrame(rows).sort_values("pr_id").reset_index(drop=True)
    feat_path = out_dir / PIME_FILES["target_prior_feat"]
    meta_path = out_dir / PIME_FILES["target_prior_meta"]
    table_path = out_dir / "target_prior_table.csv"
    write_pickle(feat_path, payload)
    table.to_csv(table_path, index=False)
    write_json(
        meta_path,
        {
            "channel_permission": "prior",
            "leakage_risk": "low",
            "source": "Existing ESM CLS features plus non-label sequence composition features.",
            "esm_cls_source": str(cls_path) if cls_path.exists() else None,
            "sequence_feature_names": seq_feature_names,
            "note": "No train-label degree, positive ratio, or known DTI labels are included.",
            "num_entities": int(len(entity_df)),
            "num_covered": int(len(payload)),
        },
    )

    update_manifest(
        dataset,
        split,
        "target_prior_feat",
        {
            "files": [
                str(feat_path.relative_to(out_dir)),
                str(meta_path.relative_to(out_dir)),
                str(table_path.relative_to(out_dir)),
            ],
            "num_entities": int(len(entity_df)),
            "num_covered": int(len(payload)),
            "coverage": float(len(payload) / len(entity_df)) if len(entity_df) else 0.0,
            "channel_permission": "prior",
            "leakage_risk": "low",
        },
    )
    return payload, table


def main():
    parser = argparse.ArgumentParser(description="Build non-label target prior features for PIME.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="cluster", choices=["cluster", "random"])
    args = parser.parse_args()
    build_target_prior(args.dataset, args.split)


if __name__ == "__main__":
    main()
