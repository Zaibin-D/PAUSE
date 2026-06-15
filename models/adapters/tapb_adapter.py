import importlib.util
import glob
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

from models.adapters.base import BaseAdapterCapabilities, ExternalBaseAdapter


ROOT = Path(__file__).resolve().parents[2]


class TAPBBaseAdapter(ExternalBaseAdapter):
    """Frozen TAPB predictor adapter for PaSMU structural audit diagnostics.

    TAPB keeps its own repository layout and expects MolFormer-tokenized SMILES
    plus precomputed ESM protein features. This adapter maps PaSMU audit batches
    to those inputs using raw SMILES/protein sequences preserved by the dataset.
    """

    adapter_name = "tapb"
    model_family = "tapb"
    capabilities = BaseAdapterCapabilities(
        consumes_explicit_prior=False,
        exposes_branch_components=False,
        requires_raw_inputs=True,
        external_code=True,
    )

    def __init__(self, config):
        super().__init__()
        base_cfg = config.MODEL.BASE
        self.code_root = self._resolve_required_path(getattr(base_cfg, "EXTERNAL_CODE_ROOT", ""), "EXTERNAL_CODE_ROOT")
        self.checkpoint_path = self._resolve_required_path(
            getattr(base_cfg, "EXTERNAL_CHECKPOINT", ""),
            "EXTERNAL_CHECKPOINT",
        )
        print(f"[TAPB adapter] loading frozen checkpoint: {self.checkpoint_path}", flush=True)
        self.model_config_path = self._resolve_path(
            getattr(base_cfg, "EXTERNAL_CONFIG", ""),
            self.code_root / "configs" / "model_config.yaml",
        )
        self.train_config_path = self._resolve_path(
            getattr(base_cfg, "EXTERNAL_TRAIN_CONFIG", ""),
            self.code_root / "configs" / "train_config.yaml",
        )
        self.data_root = self._resolve_data_root(getattr(base_cfg, "EXTERNAL_DATA_ROOT", ""))
        self.tokenizer_path = self._resolve_path(
            getattr(base_cfg, "EXTERNAL_TOKENIZER", ""),
            self.code_root / "models" / "drug" / "molformer",
        )

        self.model_config = self._load_yaml(self.model_config_path)
        train_config = self._load_yaml(self.train_config_path)
        train_cfg = train_config.get("TRAIN", {})
        self.protein_feature_path = self.data_root / train_cfg.get("PR_PATH", "pr_f_1280_2000.pkl")
        self.confounder_path = self.data_root / train_cfg.get("C_PATH", "C_1280_2000_8.pkl")

        self._ensure_tapb_artifacts()
        self.tapb_cls = self._load_tapb_class()
        self.tokenizer = self._load_tokenizer()
        self.protein_features = self._load_pickle(self.protein_feature_path)
        self.protein_index = self._build_protein_index()

        confounders = self._load_pickle(self.confounder_path)
        c = torch.from_numpy(confounders["cluster_centers"]).permute(1, 0).to(dtype=torch.float32)
        p_ci = confounders["prior"].to(dtype=torch.float32)

        self.model = self.tapb_cls(model_configs=self.model_config, c=c, p_ci=p_ci)
        state = torch.load(self.checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _resolve_path(value, default=None):
        raw = os.path.expandvars(os.path.expanduser(str(value or "").strip()))
        if not raw and default is None:
            return None
        path = Path(raw) if raw else Path(default)
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        return path

    @classmethod
    def _resolve_required_path(cls, value, name):
        path = cls._resolve_path(value)
        if path is None:
            raise ValueError(f"MODEL.BASE.{name} must be set for TAPB.")
        if any(char in str(path) for char in "*?[]"):
            matches = sorted(
                (Path(match).resolve() for match in glob.glob(str(path))),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
            if not matches:
                raise FileNotFoundError(f"MODEL.BASE.{name} pattern matched no files: {path}")
            path = matches[0]
        if not path.exists():
            raise FileNotFoundError(f"MODEL.BASE.{name} does not exist: {path}")
        return path

    def _resolve_data_root(self, value):
        path = self._resolve_path(value) if str(value or "").strip() else self._infer_data_root()
        if not path.exists():
            raise FileNotFoundError(f"MODEL.BASE.EXTERNAL_DATA_ROOT does not exist: {path}")
        return path

    def _infer_data_root(self):
        parts = list(self.checkpoint_path.parts)
        if "results" in parts:
            idx = parts.index("results")
            if len(parts) > idx + 2:
                data, split = parts[idx + 1], parts[idx + 2]
                return self.code_root / "datasets" / data / split
        raise ValueError(
            "Set MODEL.BASE.EXTERNAL_DATA_ROOT because it could not be inferred "
            "from EXTERNAL_CHECKPOINT. Expected a TAPB path like results/<data>/<split>/seed_*/best_epoch_*.pth."
        )

    @staticmethod
    def _load_yaml(path):
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as fp:
            return yaml.safe_load(fp) or {}

    @staticmethod
    def _load_pickle(path):
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("rb") as fp:
            return pickle.load(fp)

    def _ensure_tapb_artifacts(self):
        missing = [path for path in [self.protein_feature_path, self.confounder_path] if not path.exists()]
        if missing:
            missing_text = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(
                "TAPB audit requires the checkpoint plus TAPB preprocessing artifacts. "
                f"Missing: {missing_text}. Generate or copy them from TAPB before running PaSMU diagnostics."
            )

    def _load_tapb_class(self):
        code_root = str(self.code_root)
        if code_root not in sys.path:
            sys.path.insert(0, code_root)
        self._load_module_alias("models.attention", self.code_root / "models" / "attention.py")
        self._load_module_alias("models.transformer", self.code_root / "models" / "transformer.py")
        module = self._load_module_alias("models.tapb", self.code_root / "models" / "tapb.py")
        return getattr(module, "TAPB")

    @staticmethod
    def _load_module_alias(module_name, path):
        existing = sys.modules.get(module_name)
        if existing is not None and getattr(existing, "__file__", None) == str(path):
            return existing
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load {module_name} from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _load_tokenizer(self):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError("TAPB adapter requires transformers in the active Python environment.") from exc
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"TAPB MolFormer tokenizer path does not exist: {self.tokenizer_path}")
        return AutoTokenizer.from_pretrained(str(self.tokenizer_path), trust_remote_code=True)

    def _build_protein_index(self):
        frames = []
        cluster_files = ["source_train_with_id.csv", "target_train_with_id.csv", "target_test_with_id.csv"]
        random_files = ["train_with_id.csv", "val_with_id.csv", "test_with_id.csv"]
        filenames = cluster_files if all((self.data_root / name).exists() for name in cluster_files) else random_files
        for filename in filenames:
            path = self.data_root / filename
            if path.exists():
                frames.append(pd.read_csv(path, usecols=["Protein", "pr_id"]))
        if not frames:
            raise FileNotFoundError(f"No TAPB *_with_id.csv files found under {self.data_root}")

        table = pd.concat(frames, ignore_index=True).drop_duplicates("Protein")
        index = {str(row.Protein): int(row.pr_id) for row in table.itertuples(index=False)}
        if not index:
            raise ValueError(f"No protein sequence to TAPB pr_id mapping could be built from {self.data_root}")
        return index

    @staticmethod
    def _tensor_to_int(value):
        if hasattr(value, "detach"):
            return int(value.detach().cpu().item())
        return int(value)

    def _protein_tensor_for_sequences(self, sequences, device, batch=None):
        missing = []
        tensors = []
        pr_ids = batch.get("pr_id") if isinstance(batch, dict) else None
        for row, seq in enumerate(sequences):
            seq = str(seq)
            pr_id = self.protein_index.get(seq)
            if pr_id is None:
                if pr_ids is None:
                    missing.append(seq[:30])
                    continue
                pr_id = self._tensor_to_int(pr_ids[row])
            tensor = torch.as_tensor(self.protein_features[pr_id], dtype=torch.float32).clone()
            if "X" in seq:
                # TAPB ESM features include a CLS token at index 0, while the
                # input sequence is residue-indexed. Unknown residues have no
                # reliable token feature and are therefore zeroed.
                masked_positions = [idx + 1 for idx, aa in enumerate(seq) if aa == "X" and idx + 1 < tensor.size(0)]
                if masked_positions:
                    tensor[torch.as_tensor(masked_positions, dtype=torch.long)] = 0.0
            tensors.append(tensor)
        if missing:
            preview = ", ".join(missing[:3])
            raise KeyError(
                f"{len(missing)} protein sequences are absent from TAPB EXTERNAL_DATA_ROOT mapping. "
                f"Examples: {preview}"
            )
        return self._pad_proteins(tensors, device)

    @staticmethod
    def _pad_proteins(tensors, device):
        max_len = max(tensor.size(0) for tensor in tensors)
        dim = tensors[0].size(-1)
        padded = torch.zeros(len(tensors), max_len, dim, dtype=torch.float32, device=device)
        mask = torch.zeros(len(tensors), max_len, dtype=torch.float32, device=device)
        for idx, tensor in enumerate(tensors):
            length = tensor.size(0)
            padded[idx, :length] = tensor.to(device)
            mask[idx, :length] = 1.0
        return padded, mask

    @staticmethod
    def _prob_to_logit(prob):
        prob = prob.clamp(1e-6, 1.0 - 1e-6)
        return torch.log(prob / (1.0 - prob))

    def forward_external(self, batch):
        if "smiles" not in batch or "protein_sequence" not in batch:
            raise KeyError("TAPB adapter requires batch['smiles'] and batch['protein_sequence'].")
        device = next(self.model.parameters()).device
        self.model.c = self.model.c.to(device)
        self.model.p_ci = self.model.p_ci.to(device)
        smiles = [str(item) for item in batch["smiles"]]
        sequences = [str(item) for item in batch["protein_sequence"]]

        drug_inputs = self.tokenizer(
            smiles,
            padding="longest",
            return_tensors="pt",
            truncation=True,
            max_length=200,
        ).to(device)
        protein_inputs, protein_mask = self._protein_tensor_for_sequences(sequences, device, batch=batch)

        output = self.model(drug_inputs, protein_inputs, pr_mask=protein_mask)
        probs = output["logits"][:, 1]
        return {
            "s_base": self._prob_to_logit(probs),
            "tapb_prob": probs,
        }
