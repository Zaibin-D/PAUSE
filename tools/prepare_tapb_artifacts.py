"""Generate TAPB preprocessing artifacts without retraining TAPB.

Run this from the PaSMU-DTI repository and point it to a separate TAPB checkout.
It calls TAPB's own preparation functions in the TAPB working directory so the
generated files keep the layout expected by TAPB checkpoints.
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def main():
    parser = argparse.ArgumentParser(description="Prepare TAPB pr_f/C artifacts for PaSMU audit.")
    parser.add_argument("--tapb-root", required=True, help="Path to the external TAPB repository.")
    parser.add_argument("--data", required=True, help="Dataset name, e.g. biosnap or bindingdb.")
    parser.add_argument("--split", default="cluster", help="TAPB split name, e.g. cluster, random, cold.")
    parser.add_argument("--force", action="store_true", help="Regenerate artifacts even if they already exist.")
    args = parser.parse_args()

    tapb_root = resolve_path(args.tapb_root)
    if not tapb_root.exists():
        raise FileNotFoundError(tapb_root)
    if str(tapb_root) not in sys.path:
        sys.path.insert(0, str(tapb_root))

    old_cwd = Path.cwd()
    os.chdir(tapb_root)
    try:
        from preparation import generate_esm2_feature, kmeans_for_c
        from utils.utils import load_config_file, set_seed

        config = load_config_file("configs/train_config.yaml")
        set_seed(int(config.TRAIN.SEED))
        data_root = Path("datasets") / args.data / args.split
        if not data_root.exists():
            raise FileNotFoundError(data_root)

        protein_path = data_root / str(config.TRAIN.PR_PATH)
        confounder_path = data_root / str(config.TRAIN.C_PATH)

        if args.force or not protein_path.exists():
            generate_esm2_feature(config, args.data, args.split)
        else:
            print(f"[Skip] {protein_path} already exists.")

        train_csv = data_root / ("source_train_with_id.csv" if args.split == "cluster" else "train_with_id.csv")
        if not train_csv.exists():
            raise FileNotFoundError(train_csv)

        if args.force or not confounder_path.exists():
            train_df = pd.read_csv(train_csv)
            kmeans_for_c(config, train_df, str(data_root))
        else:
            print(f"[Skip] {confounder_path} already exists.")

        print(f"[Done] TAPB artifacts ready under {data_root.resolve()}")
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
