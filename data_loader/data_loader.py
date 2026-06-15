"""
DTI dataset and dataloader.

Unified token-based pipeline:
- drug CLS + drug token features
- protein CLS + protein token features
- no auxiliary protein view
- no drug/protein graph features
"""
import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, Dataset


PIME_FILE_DRUG_PRIOR = "drug_prior_feat.pkl"
PIME_FILE_TARGET_PRIOR = "target_prior_feat.pkl"
PIME_FILE_DRUG_ENTITY = "drug_entity.csv"
PIME_FILE_TARGET_ENTITY = "target_entity.csv"


def _phase_file_map(config, split):
    split_cfg = getattr(getattr(config, "DATA", None), "SPLIT_FILES", None)
    if split == "cluster":
        return {
            "train": str(getattr(split_cfg, "CLUSTER_TRAIN", "source_train")),
            "val": str(getattr(split_cfg, "CLUSTER_VAL", "target_test")),
            "test": str(getattr(split_cfg, "CLUSTER_TEST", "target_test")),
        }
    return {
        "train": str(getattr(split_cfg, "RANDOM_TRAIN", "train")),
        "val": str(getattr(split_cfg, "RANDOM_VAL", "val")),
        "test": str(getattr(split_cfg, "RANDOM_TEST", "test")),
    }


class DTIDataset(Dataset):
    def __init__(self, dataset, split, phase, config):
        data_paths_cfg = config.DATA.PATHS
        data_files_cfg = config.DATA.FILES
        env_cfg = config.DATA.ENVIRONMENT

        self.dataset = dataset
        self.split = split
        self.phase = phase

        data_dir = os.path.join(data_paths_cfg.ROOT_DIR, dataset, split)
        self.data_dir = data_dir
        file_map = _phase_file_map(config, split)

        csv_path = os.path.join(data_dir, f"{file_map[phase]}_with_id.csv")
        self.df = pd.read_csv(csv_path)
        self.original_len = len(self.df)
        print(f"  [{phase}] Loaded {self.original_len} samples from {csv_path}")

        self.drug_macro = self._require_pkl(data_dir, data_files_cfg.FILE_DRUG_MACRO)
        self.drug_token = self._require_pkl(data_dir, data_files_cfg.FILE_DRUG_TOKEN)
        self.prot_macro = self._require_pkl(data_dir, data_files_cfg.FILE_PROT_MACRO)
        self.prot_token = self._require_pkl(data_dir, data_files_cfg.FILE_PROT_TOKEN)

        before_feature_filter = len(self.df)
        drug_macro_ids = set(self.drug_macro)
        drug_token_ids = set(self.drug_token)
        prot_macro_ids = set(self.prot_macro)
        prot_token_ids = set(self.prot_token)

        valid_mask = (
            self.df["dr_id"].astype(int).isin(drug_macro_ids)
            & self.df["dr_id"].astype(int).isin(drug_token_ids)
            & self.df["pr_id"].astype(int).isin(prot_macro_ids)
            & self.df["pr_id"].astype(int).isin(prot_token_ids)
        )

        self.df = self.df.loc[valid_mask].reset_index(drop=True)
        skipped = before_feature_filter - len(self.df)
        print(f"    Feature-covered samples: {len(self.df)}/{before_feature_filter}")
        if skipped > 0:
            print(f"    [WARNING] {phase}: Skipped {skipped} samples due to missing features")
        if len(self.df) == 0:
            raise RuntimeError(f"No usable samples remain for {dataset}/{split}/{phase}. Check feature extraction output.")

        num_env_buckets = int(env_cfg.PSEUDO_ENV_BUCKETS)
        env_seed = int(env_cfg.PSEUDO_ENV_SEED)
        env_mode = str(env_cfg.PSEUDO_ENV_MODE).strip().lower()
        env_ids = self._build_pseudo_env_ids(
            self.df,
            self.drug_macro,
            self.prot_macro,
            num_env_buckets,
            env_seed,
            env_mode,
        )
        self.df["env_id"] = env_ids.astype(np.int64)
        env_counts = self.df["env_id"].value_counts().sort_index()
        print(
            f"    Pseudo environments[{env_mode}]: {len(env_counts)} active / {num_env_buckets} configured "
            f"(min={int(env_counts.min())}, max={int(env_counts.max())})"
        )

    @staticmethod
    def _require_pkl(data_dir, filename):
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required feature file not found: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"    Loaded {filename} ({len(data)} entries)")
        return data

    @staticmethod
    def _normalize_rows(features):
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return features / np.clip(norms, a_min=1e-6, a_max=None)

    @classmethod
    def _cluster_features(cls, features, num_buckets, seed):
        num_rows = int(features.shape[0])
        active_buckets = max(1, min(int(num_buckets), num_rows))
        if active_buckets == 1:
            return np.zeros(num_rows, dtype=np.int64)
        clusterer = MiniBatchKMeans(
            n_clusters=active_buckets,
            random_state=seed,
            batch_size=min(max(active_buckets * 8, 32), num_rows),
            n_init=10,
            reassignment_ratio=0.01,
        )
        return clusterer.fit_predict(features).astype(np.int64)

    @classmethod
    def _build_pseudo_env_ids(cls, df, drug_macro, prot_macro, num_buckets, seed, env_mode):
        env_mode = str(env_mode).strip().lower()
        if env_mode == "protein_macro":
            unique_pr = sorted(df["pr_id"].astype(int).unique().tolist())
            if not unique_pr:
                return np.zeros(len(df), dtype=np.int64)
            prot_features = np.stack([np.asarray(prot_macro[pid], dtype=np.float32) for pid in unique_pr], axis=0)
            prot_features = cls._normalize_rows(prot_features)
            env_ids = cls._cluster_features(prot_features, num_buckets, seed)
            env_lookup = {pid: int(env_id) for pid, env_id in zip(unique_pr, env_ids)}
            return df["pr_id"].astype(int).map(env_lookup).astype(np.int64).to_numpy()

        if env_mode != "pair_macro":
            raise ValueError(
                f"Unsupported PSEUDO_ENV_MODE={env_mode!r}. Expected one of: 'pair_macro', 'protein_macro'."
            )

        pair_df = df[["dr_id", "pr_id"]].astype(int).drop_duplicates().reset_index(drop=True)
        if pair_df.empty:
            return np.zeros(len(df), dtype=np.int64)

        drug_features = np.stack(
            [np.asarray(drug_macro[dr_id], dtype=np.float32) for dr_id in pair_df["dr_id"].tolist()],
            axis=0,
        )
        prot_features = np.stack(
            [np.asarray(prot_macro[pr_id], dtype=np.float32) for pr_id in pair_df["pr_id"].tolist()],
            axis=0,
        )
        drug_features = cls._normalize_rows(drug_features)
        prot_features = cls._normalize_rows(prot_features)
        pair_features = np.concatenate([drug_features, prot_features], axis=1)
        pair_features = cls._normalize_rows(pair_features)
        env_ids = cls._cluster_features(pair_features, num_buckets, seed)
        env_lookup = {
            (int(dr_id), int(pr_id)): int(env_id)
            for dr_id, pr_id, env_id in zip(pair_df["dr_id"], pair_df["pr_id"], env_ids)
        }
        pair_keys = list(zip(df["dr_id"].astype(int).tolist(), df["pr_id"].astype(int).tolist()))
        return np.asarray([env_lookup[key] for key in pair_keys], dtype=np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dr_id = int(row["dr_id"])
        pr_id = int(row["pr_id"])
        label = float(row["Y"])
        env_id = int(row["env_id"])
        return {
            "macro_drug": torch.as_tensor(self.drug_macro[dr_id], dtype=torch.float32),
            "macro_target": torch.as_tensor(self.prot_macro[pr_id], dtype=torch.float32),
            "drug_tokens": torch.as_tensor(self.drug_token[dr_id], dtype=torch.float32),
            "prot_tokens": torch.as_tensor(self.prot_token[pr_id], dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
            "env_id": torch.tensor(env_id, dtype=torch.long),
            "dr_id": torch.tensor(dr_id, dtype=torch.long),
            "pr_id": torch.tensor(pr_id, dtype=torch.long),
            "smiles": "" if "SMILES" not in row or pd.isna(row["SMILES"]) else str(row["SMILES"]),
            "protein_sequence": "" if "Protein" not in row or pd.isna(row["Protein"]) else str(row["Protein"]),
        }


def _as_feature_matrix(value, feature_dim=0):
    if isinstance(value, dict):
        value = value.get("features", value.get("feature", value))
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        width = int(feature_dim) if feature_dim else 0
        return np.zeros((0, width), dtype=np.float32)
    return array.astype(np.float32, copy=False)


def _infer_matrix_dim(mapping, nested=False):
    for value in mapping.values():
        if nested:
            if not value:
                continue
            for item in value:
                array = _as_feature_matrix(item)
                if array.ndim == 2 and array.shape[1] > 0:
                    return int(array.shape[1])
        else:
            array = _as_feature_matrix(value)
            if array.ndim == 2 and array.shape[1] > 0:
                return int(array.shape[1])
    return 0


class PIMEDTIDataset(DTIDataset):
    """DTI dataset with the prior evidence required by PAUSE.

    The base predictor inputs are kept intact. PAUSE adds non-label drug and
    target prior features plus raw entity text required by external adapters.
    """

    def __init__(self, dataset, split, phase, config):
        super().__init__(dataset, split, phase, config)

        pime_dir = os.path.join(self.data_dir, "pime")
        self.pime_dir = pime_dir
        self.drug_prior = self._require_pime_pkl(pime_dir, PIME_FILE_DRUG_PRIOR)
        self.target_prior = self._require_pime_pkl(pime_dir, PIME_FILE_TARGET_PRIOR)
        self.drug_smiles = self._load_entity_text(
            pime_dir,
            PIME_FILE_DRUG_ENTITY,
            id_col="dr_id",
            value_cols=("canonical_smiles", "selected_component_smiles", "smiles"),
        )
        self.target_sequences = self._load_entity_text(
            pime_dir,
            PIME_FILE_TARGET_ENTITY,
            id_col="pr_id",
            value_cols=("protein_sequence",),
        )

        self.drug_prior_dim = _infer_matrix_dim(self.drug_prior)
        self.target_prior_dim = _infer_matrix_dim(self.target_prior)
        self._filter_pime_coverage()

    @staticmethod
    def _require_pime_pkl(pime_dir, filename):
        path = os.path.join(pime_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required PIME feature file not found: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"    Loaded pime/{filename} ({len(data)} entries)")
        return data

    @staticmethod
    def _load_entity_text(pime_dir, filename, id_col, value_cols):
        path = os.path.join(pime_dir, filename)
        if not os.path.exists(path):
            print(f"    Optional pime/{filename} not found; raw external-model inputs unavailable")
            return {}
        table = pd.read_csv(path)
        if id_col not in table.columns:
            print(f"    Optional pime/{filename} missing {id_col}; raw external-model inputs unavailable")
            return {}
        selected_col = next((col for col in value_cols if col in table.columns), None)
        if selected_col is None:
            print(f"    Optional pime/{filename} missing {value_cols}; raw external-model inputs unavailable")
            return {}
        out = {}
        for row in table[[id_col, selected_col]].itertuples(index=False):
            try:
                entity_id = int(row[0])
            except (TypeError, ValueError):
                continue
            value = "" if pd.isna(row[1]) else str(row[1])
            out[entity_id] = value
        print(f"    Loaded pime/{filename}:{selected_col} ({len(out)} entries)")
        return out

    def _filter_pime_coverage(self):
        before = len(self.df)
        required_mask = (
            self.df["dr_id"].astype(int).isin(set(self.drug_prior))
            & self.df["pr_id"].astype(int).isin(set(self.target_prior))
        )

        self.df = self.df.loc[required_mask].reset_index(drop=True)
        skipped = before - len(self.df)
        print(
            f"    PIME-covered samples: {len(self.df)}/{before} "
            f"(skipped={skipped})"
        )
        print(
            "    PIME dims: "
            f"drug_prior={self.drug_prior_dim}, "
            f"target_prior={self.target_prior_dim}"
        )
        if len(self.df) == 0:
            raise RuntimeError(
                f"No PIME-usable samples remain for {self.dataset}/{self.split}/{self.phase}. "
                "Check PIME Evidence Store coverage."
            )

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        dr_id = int(batch["dr_id"].item())
        pr_id = int(batch["pr_id"].item())

        drug_prior = np.asarray(self.drug_prior[dr_id], dtype=np.float32).reshape(-1)
        target_prior = np.asarray(self.target_prior[pr_id], dtype=np.float32).reshape(-1)

        batch.update(
            {
                "pime_drug_prior": torch.as_tensor(drug_prior, dtype=torch.float32),
                "pime_target_prior": torch.as_tensor(target_prior, dtype=torch.float32),
                "smiles": batch.get("smiles") or self.drug_smiles.get(dr_id, ""),
                "protein_sequence": batch.get("protein_sequence") or self.target_sequences.get(pr_id, ""),
            }
        )
        return batch


def dti_collate_fn(batch_list):
    batch_size = len(batch_list)
    macro_drug = torch.stack([b["macro_drug"] for b in batch_list])
    macro_target = torch.stack([b["macro_target"] for b in batch_list])
    labels = torch.stack([b["label"] for b in batch_list])

    drug_token_list = [b["drug_tokens"] for b in batch_list]
    max_d_len = max(t.size(0) for t in drug_token_list)
    drug_dim = drug_token_list[0].size(1)
    drug_tokens_padded = torch.zeros(batch_size, max_d_len, drug_dim)
    drug_tokens_mask = torch.zeros(batch_size, max_d_len, dtype=torch.bool)
    for i, tokens in enumerate(drug_token_list):
        length = tokens.size(0)
        drug_tokens_padded[i, :length, :] = tokens
        drug_tokens_mask[i, :length] = True

    prot_token_list = [b["prot_tokens"] for b in batch_list]
    max_p_len = max(t.size(0) for t in prot_token_list)
    prot_dim = prot_token_list[0].size(1)
    prot_tokens_padded = torch.zeros(batch_size, max_p_len, prot_dim)
    prot_tokens_mask = torch.zeros(batch_size, max_p_len, dtype=torch.bool)
    for i, tokens in enumerate(prot_token_list):
        length = tokens.size(0)
        prot_tokens_padded[i, :length, :] = tokens
        prot_tokens_mask[i, :length] = True

    batch = {
        "macro_drug": macro_drug,
        "macro_target": macro_target,
        "drug_tokens": drug_tokens_padded,
        "drug_tokens_mask": drug_tokens_mask,
        "prot_tokens": prot_tokens_padded,
        "prot_tokens_mask": prot_tokens_mask,
        "label": labels,
        "env_id": torch.stack([b["env_id"] for b in batch_list]),
        "dr_id": torch.stack([b["dr_id"] for b in batch_list]),
        "pr_id": torch.stack([b["pr_id"] for b in batch_list]),
    }
    if "smiles" in batch_list[0]:
        batch["smiles"] = [b.get("smiles", "") for b in batch_list]
    if "protein_sequence" in batch_list[0]:
        batch["protein_sequence"] = [b.get("protein_sequence", "") for b in batch_list]
    return batch


def pime_collate_fn(batch_list):
    batch = dti_collate_fn(batch_list)
    batch["pime_drug_prior"] = torch.stack([b["pime_drug_prior"] for b in batch_list])
    batch["pime_target_prior"] = torch.stack([b["pime_target_prior"] for b in batch_list])
    return batch


def get_dataloader(
    dataset,
    split,
    phase,
    config,
    batch_size=32,
    shuffle=True,
    num_workers=None,
):
    loader_cfg = config.DATA.LOADER
    if num_workers is None:
        num_workers = int(loader_cfg.NUM_WORKERS)
    ds = PIMEDTIDataset(dataset, split, phase, config)
    loader_kwargs = {
        "dataset": ds,
        "collate_fn": pime_collate_fn,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    loader_kwargs.update(
        {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "drop_last": False,
        }
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(loader_cfg.PERSISTENT_WORKERS)
        loader_kwargs["prefetch_factor"] = max(int(loader_cfg.PREFETCH_FACTOR), 1)
    return DataLoader(**loader_kwargs)


def get_pime_dataloader(
    dataset,
    split,
    phase,
    config,
    batch_size=32,
    shuffle=True,
    num_workers=None,
):
    return get_dataloader(
        dataset,
        split,
        phase,
        config,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

