"""
CAN-only DTI model.

The model keeps macro encoders, token projections, CAN layers, and a direct
classifier over pooled CAN token representations plus macro pair features.
"""
import torch
import torch.nn as nn

from configs import Config
from models.encoders import CANLayer, MacroEncoder


class CANDTIModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        backbone_cfg = config.MODEL.BACKBONE
        dim = backbone_cfg.HIDDEN_DIM
        dropout = backbone_cfg.DROPOUT

        self.macro_drug_enc = MacroEncoder(
            input_dim=backbone_cfg.MACRO_DRUG_INPUT_DIM,
            hidden_dim=dim,
            output_dim=dim,
            dropout=dropout,
        )
        self.macro_target_enc = MacroEncoder(
            input_dim=backbone_cfg.MACRO_TARGET_INPUT_DIM,
            hidden_dim=dim,
            output_dim=dim,
            dropout=dropout,
        )
        self.macro_pair_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.drug_token_proj = nn.Linear(backbone_cfg.DRUG_TOKEN_DIM, dim)
        self.prot_token_proj = nn.Linear(backbone_cfg.PROT_TOKEN_DIM, dim)
        self.can_layers = nn.ModuleList(
            [
                CANLayer(
                    hidden_dim=dim,
                    num_heads=backbone_cfg.CAN_NUM_HEADS,
                    group_size=backbone_cfg.CAN_GROUP_SIZE,
                    dropout=dropout,
                )
                for _ in range(backbone_cfg.CAN_NUM_LAYERS)
            ]
        )

        self.can_classifier = nn.Sequential(
            nn.Linear(dim * 5, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1),
        )

    @staticmethod
    def _masked_mean(states, mask):
        denom = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        return (states * mask.unsqueeze(-1).float()).sum(dim=1) / denom

    def forward(self, batch):
        v_drug = self.macro_drug_enc(batch["macro_drug"])
        v_prot = self.macro_target_enc(batch["macro_target"])
        v_pair = self.macro_pair_fusion(torch.cat([v_drug, v_prot], dim=-1))

        d_tok = self.drug_token_proj(batch["drug_tokens"])
        p_tok = self.prot_token_proj(batch["prot_tokens"])
        drug_tokens_mask = batch["drug_tokens_mask"]
        prot_tokens_mask = batch["prot_tokens_mask"]

        for layer in self.can_layers:
            d_tok, p_tok = layer(d_tok, p_tok, drug_tokens_mask, prot_tokens_mask)

        drug_summary = self._masked_mean(d_tok, drug_tokens_mask)
        prot_summary = self._masked_mean(p_tok, prot_tokens_mask)
        pair_features = torch.cat(
            [
                v_pair,
                drug_summary,
                prot_summary,
                torch.abs(drug_summary - prot_summary),
                drug_summary * prot_summary,
            ],
            dim=-1,
        )
        return self.can_classifier(pair_features)
