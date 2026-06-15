import glob
import importlib.util
import os
import sys
from collections import OrderedDict
from functools import partial
from pathlib import Path

import numpy as np
import torch
from models.adapters.base import BaseAdapterCapabilities, ExternalBaseAdapter


ROOT = Path(__file__).resolve().parents[2]


CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}


class DrugBANBaseAdapter(ExternalBaseAdapter):
    """Frozen DrugBAN predictor adapter for PAUSE diagnostics.

    DrugBAN keeps its own DGL/dgllife preprocessing path. This adapter maps the
    PAUSE audit batch raw SMILES and protein sequences to DrugBAN graph/sequence
    inputs and returns a base logit without changing the checkpoint.
    """

    adapter_name = "drugban"
    model_family = "drugban"
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
        self.model_config_path = self._resolve_path(
            getattr(base_cfg, "EXTERNAL_CONFIG", ""),
            self.code_root / "configs" / "DrugBAN.yaml",
        )
        print(f"[DrugBAN adapter] loading frozen checkpoint: {self.checkpoint_path}", flush=True)

        self.cfg = self._load_drugban_cfg(self.model_config_path)
        self.max_drug_nodes = int(self.cfg.DRUG.MAX_NODES)
        self.out_binary = int(self.cfg.DECODER.BINARY)
        self.drugban_cls = self._load_drugban_class()

        self.model = self.drugban_cls(**self.cfg)
        state = torch.load(self.checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        try:
            import dgl
            from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph
        except ImportError as exc:
            raise ImportError(
                "DrugBAN adapter requires dgl and dgllife in the active Python environment. "
                "Run PAUSE diagnostics from the same environment used for DrugBAN, or install DrugBAN dependencies."
            ) from exc
        self._dgl = dgl
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.smiles_to_graph = partial(smiles_to_bigraph, add_self_loop=True)
        self.graph_cache = OrderedDict()
        self.graph_cache_size = int(os.environ.get("DRUGBAN_GRAPH_CACHE_SIZE", "20000"))

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
            raise ValueError(f"MODEL.BASE.{name} must be set for DrugBAN.")
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

    def _load_drugban_cfg(self, config_path):
        configs_module = self._load_module_alias("drugban_external_configs", self.code_root / "configs.py")
        cfg = configs_module.get_cfg_defaults()
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        cfg.merge_from_file(str(config_path))
        cfg.freeze()
        return cfg

    def _load_drugban_class(self):
        code_root = str(self.code_root)
        if code_root not in sys.path:
            sys.path.insert(0, code_root)
        module = self._load_module_alias("drugban_external_models", self.code_root / "models.py")
        return getattr(module, "DrugBAN")

    @staticmethod
    def _integer_label_protein(sequence, max_length=1200):
        encoding = np.zeros(max_length, dtype=np.int64)
        for idx, letter in enumerate(str(sequence)[:max_length]):
            encoding[idx] = CHARPROTSET.get(str(letter).upper(), 0)
        return encoding

    def _build_graph_from_smiles(self, smiles):
        graph = self.smiles_to_graph(
            smiles=str(smiles),
            node_featurizer=self.atom_featurizer,
            edge_featurizer=self.bond_featurizer,
        )
        actual_node_feats = graph.ndata.pop("h")
        num_actual_nodes = int(actual_node_feats.shape[0])
        if num_actual_nodes > self.max_drug_nodes:
            raise ValueError(
                f"DrugBAN graph has {num_actual_nodes} nodes, exceeding configured max {self.max_drug_nodes}."
            )
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1], dtype=actual_node_feats.dtype)
        graph.ndata["h"] = torch.cat((actual_node_feats, virtual_node_bit), dim=1)
        if num_virtual_nodes > 0:
            virtual_node_feat = torch.cat(
                (
                    torch.zeros(num_virtual_nodes, 74, dtype=actual_node_feats.dtype),
                    torch.ones(num_virtual_nodes, 1, dtype=actual_node_feats.dtype),
                ),
                dim=1,
            )
            graph.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        return graph.add_self_loop()

    def _graph_from_smiles(self, smiles):
        smiles = str(smiles)
        if self.graph_cache_size <= 0:
            return self._build_graph_from_smiles(smiles)
        graph = self.graph_cache.get(smiles)
        if graph is not None:
            self.graph_cache.move_to_end(smiles)
            return graph.clone()
        graph = self._build_graph_from_smiles(smiles)
        self.graph_cache[smiles] = graph
        if len(self.graph_cache) > self.graph_cache_size:
            self.graph_cache.popitem(last=False)
        return graph.clone()

    @staticmethod
    def _binary_logits_to_base(score, out_binary):
        if out_binary == 1:
            return score.view(-1)
        if out_binary == 2:
            return (score[:, 1] - score[:, 0]).view(-1)
        raise ValueError(f"Unsupported DrugBAN DECODER.BINARY={out_binary}; expected 1 or 2.")

    def forward_external(self, batch):
        if "smiles" not in batch or "protein_sequence" not in batch:
            raise KeyError("DrugBAN adapter requires batch['smiles'] and batch['protein_sequence'].")
        device = next(self.model.parameters()).device
        smiles = [str(item) for item in batch["smiles"]]
        sequences = [str(item) for item in batch["protein_sequence"]]

        graphs = [self._graph_from_smiles(item) for item in smiles]
        drug_graph = self._dgl.batch(graphs).to(device)
        protein = torch.as_tensor(
            np.stack([self._integer_label_protein(seq) for seq in sequences], axis=0),
            dtype=torch.long,
            device=device,
        )

        _, _, _, score = self.model(drug_graph, protein)
        s_base = self._binary_logits_to_base(score, self.out_binary)
        return {
            "s_base": s_base,
            "drugban_prob": torch.sigmoid(s_base),
        }
