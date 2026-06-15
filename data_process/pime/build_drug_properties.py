import argparse

import numpy as np
import pandas as pd

from data_process.pime.common import pime_dir, update_manifest, write_json, write_pickle
from data_process.pime.schema import PIME_FILES


GLOBAL_DESCRIPTOR_NAMES = [
    "mol_wt",
    "logp",
    "tpsa",
    "num_h_donors",
    "num_h_acceptors",
    "num_rotatable_bonds",
    "ring_count",
    "num_aromatic_rings",
    "num_aliphatic_rings",
    "fraction_csp3",
    "heavy_atom_count",
    "formal_charge",
]


def _load_rdkit():
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import Crippen, Descriptors, Lipinski, MACCSkeys, rdMolDescriptors
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as exc:
        raise RuntimeError(
            "RDKit is required for PIME drug prior features. Activate the project "
            "environment from environment.yml or install rdkit before running this builder."
        ) from exc

    return Chem, DataStructs, Crippen, Descriptors, Lipinski, MACCSkeys, rdMolDescriptors, MurckoScaffold


def _descriptor_vector(mol, rdkit_modules):
    Chem, _, Crippen, Descriptors, Lipinski, _, rdMolDescriptors, _ = rdkit_modules
    formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    return np.asarray(
        [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumRotatableBonds(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcNumAliphaticRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            mol.GetNumHeavyAtoms(),
            formal_charge,
        ],
        dtype=np.float32,
    )


def _maccs_bits(mol, rdkit_modules):
    _, DataStructs, _, _, _, MACCSkeys, _, _ = rdkit_modules
    bitvect = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((bitvect.GetNumBits(),), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr


def _scaffold_smiles(mol, rdkit_modules):
    Chem, _, _, _, _, _, _, MurckoScaffold = rdkit_modules
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(scaffold, canonical=True)


def build_drug_properties(dataset, split):
    rdkit_modules = _load_rdkit()
    Chem = rdkit_modules[0]
    out_dir = pime_dir(dataset, split)
    entity_path = out_dir / PIME_FILES["drug_entity"]
    if not entity_path.exists():
        raise FileNotFoundError(f"Run build_entity_registry first: {entity_path}")

    entity_df = pd.read_csv(entity_path)
    features = {}
    rows = []
    maccs_names = [f"maccs_{idx}" for idx in range(167)]
    feature_names = GLOBAL_DESCRIPTOR_NAMES + maccs_names

    for row in entity_df.itertuples(index=False):
        canonical = "" if pd.isna(row.canonical_smiles) else str(row.canonical_smiles).strip()
        raw = "" if pd.isna(row.smiles) else str(row.smiles).strip()
        smiles = canonical or raw
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            rows.append(
                {
                    "dr_id": int(row.dr_id),
                    "status": "invalid_or_missing",
                    "canonical_smiles": smiles,
                    "scaffold_smiles": "",
                    "num_features": 0,
                }
            )
            continue
        desc = _descriptor_vector(mol, rdkit_modules)
        bits = _maccs_bits(mol, rdkit_modules)
        vec = np.concatenate([desc, bits]).astype(np.float32)
        features[int(row.dr_id)] = vec
        rows.append(
            {
                "dr_id": int(row.dr_id),
                "status": "ok",
                "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
                "scaffold_smiles": _scaffold_smiles(mol, rdkit_modules),
                "num_features": int(vec.shape[0]),
            }
        )

    table = pd.DataFrame(rows).sort_values("dr_id").reset_index(drop=True)
    feat_path = out_dir / PIME_FILES["drug_prior_feat"]
    meta_path = out_dir / PIME_FILES["drug_prior_meta"]
    table_path = out_dir / "drug_prior_table.csv"
    write_pickle(feat_path, features)
    table.to_csv(table_path, index=False)
    write_json(
        meta_path,
        {
            "channel_permission": "prior",
            "leakage_risk": "low",
            "source": "RDKit descriptors and MACCS keys; no label-derived statistics.",
            "feature_names": feature_names,
            "feature_dim": len(feature_names),
            "num_entities": int(len(entity_df)),
            "num_covered": int(len(features)),
        },
    )

    update_manifest(
        dataset,
        split,
        "drug_prior_feat",
        {
            "files": [
                str(feat_path.relative_to(out_dir)),
                str(meta_path.relative_to(out_dir)),
                str(table_path.relative_to(out_dir)),
            ],
            "num_entities": int(len(entity_df)),
            "num_covered": int(len(features)),
            "coverage": float(len(features) / len(entity_df)) if len(entity_df) else 0.0,
            "channel_permission": "prior",
            "leakage_risk": "low",
        },
    )
    return features, table


def main():
    parser = argparse.ArgumentParser(description="Build non-label drug prior features for PIME.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="cluster", choices=["cluster", "random"])
    args = parser.parse_args()
    build_drug_properties(args.dataset, args.split)


if __name__ == "__main__":
    main()
