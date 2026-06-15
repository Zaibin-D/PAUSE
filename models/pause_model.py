import torch
import torch.nn as nn

from configs import Config
from models.adapters import create_base_adapter
from models.prior_features import DRUG_PRIOR_DIM, TARGET_PRIOR_DIM


class PriorEvidenceHead(nn.Module):
    """Explicit prior channel used by the frozen-predictor audit."""

    def __init__(self, config: Config):
        super().__init__()
        hidden_dim = int(config.MODEL.BACKBONE.HIDDEN_DIM)
        dropout = float(config.MODEL.BACKBONE.DROPOUT)
        input_dim = (
            int(config.MODEL.BACKBONE.MACRO_DRUG_INPUT_DIM)
            + DRUG_PRIOR_DIM
            + TARGET_PRIOR_DIM
        )
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, batch):
        features = torch.cat(
            [
                batch["macro_drug"],
                batch["pime_drug_prior"],
                batch["pime_target_prior"],
            ],
            dim=-1,
        )
        return self.net(features).view(-1)


class PAUSEModel(nn.Module):
    """Frozen DTI predictor wrapper exposing base and prior logits."""

    def __init__(self, config: Config):
        super().__init__()
        self.base = create_base_adapter(config)
        self.base_model_type = getattr(self.base, "model_family", "unknown")
        self.base_capabilities = getattr(self.base, "capabilities", None)
        self.prior_head = PriorEvidenceHead(config)

    @staticmethod
    def _neutral_base_components(s_base):
        zero = torch.zeros_like(s_base)
        return {
            "s_base": s_base,
            "s_basic": s_base,
            "s_cls": zero,
            "s_token": zero,
            "s_evidence_base": zero,
            "s_consensus": s_base,
            "evidence_branch_logits": s_base.unsqueeze(-1),
            "evidence_consensus_weight": torch.zeros(
                s_base.size(0),
                3,
                device=s_base.device,
                dtype=s_base.dtype,
            ),
            "evidence_consensus_disagreement": zero,
            "evidence_drug_prior_weight": zero,
            "evidence_target_prior_weight": zero,
        }

    def _base_components(self, batch):
        if hasattr(self.base, "forward_components"):
            return self.base.forward_components(batch)
        return self._neutral_base_components(self.base(batch).view(-1))

    def freeze_base(self):
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()

    def enforce_frozen_eval(self):
        self.base.eval()

    def train_base_only(self):
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.base.parameters():
            parameter.requires_grad_(True)

    def train_prior_only(self):
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.prior_head.parameters():
            parameter.requires_grad_(True)

    def forward_base(self, batch):
        components = self._base_components(batch)
        s_base = components["s_base"].view(-1)
        return {
            **components,
            "s_base": s_base,
            "s_full": s_base,
            "s_global": s_base,
        }

    def forward(self, batch):
        output = self.forward_base(batch)
        output["s_prior"] = self.prior_head(batch).view(-1)
        return output


__all__ = ["PAUSEModel", "PriorEvidenceHead"]
