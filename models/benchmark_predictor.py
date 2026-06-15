import torch
import torch.nn as nn

from configs import Config
from models.encoders import MacroEncoder
from models.prior_features import DRUG_PRIOR_DIM, TARGET_PRIOR_DIM


def projection(in_dim, hidden_dim, dropout):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    )


class ResidualConditioner(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, evidence, condition):
        delta = self.net(torch.cat([evidence, condition], dim=-1))
        return self.norm(evidence + delta)


class TokenPriorAdapter(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens, condition):
        delta = self.net(condition).unsqueeze(1)
        return self.norm(tokens + delta)


class LearnedEvidenceCompressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_queries, num_heads, dropout):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _valid_mask(mask):
        mask = mask.bool()
        if mask.size(1) == 0:
            return mask
        empty = ~mask.any(dim=1)
        if empty.any():
            mask = mask.clone()
            mask[empty, 0] = True
        return mask

    def forward(self, tokens, mask):
        tokens = self.proj(tokens)
        mask = self._valid_mask(mask)
        queries = self.queries.unsqueeze(0).expand(tokens.size(0), -1, -1)
        attended, _ = self.attn(queries, tokens, tokens, key_padding_mask=(~mask).bool(), need_weights=False)
        evidence = self.norm_attn(queries + attended)
        return self.norm_ffn(evidence + self.ffn(evidence))


class PooledEvidenceCompressor(nn.Module):
    """Non-attentive token compressor for the w/o learned compression ablation."""

    def __init__(self, input_dim, hidden_dim, num_queries, num_heads, dropout):
        super().__init__()
        del num_heads
        self.num_queries = int(num_queries)
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _valid_mask(mask):
        mask = mask.bool()
        if mask.size(1) == 0:
            return mask
        empty = ~mask.any(dim=1)
        if empty.any():
            mask = mask.clone()
            mask[empty, 0] = True
        return mask

    def forward(self, tokens, mask):
        tokens = self.dropout(self.norm(self.proj(tokens)))
        mask = self._valid_mask(mask)
        mask_f = mask.unsqueeze(-1).to(dtype=tokens.dtype)
        global_count = mask_f.sum(dim=1).clamp(min=1.0)
        global_mean = (tokens * mask_f).sum(dim=1) / global_count

        length = tokens.size(1)
        pooled = []
        for idx in range(self.num_queries):
            start = idx * length // self.num_queries
            end = max((idx + 1) * length // self.num_queries, start + 1)
            end = min(end, length)
            segment = tokens[:, start:end]
            segment_mask = mask[:, start:end].unsqueeze(-1).to(dtype=tokens.dtype)
            count = segment_mask.sum(dim=1)
            segment_mean = (segment * segment_mask).sum(dim=1) / count.clamp(min=1.0)
            segment_mean = torch.where(count > 0.0, segment_mean, global_mean)
            pooled.append(segment_mean)
        return torch.stack(pooled, dim=1)


class CrossEvidenceBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.drug_to_target = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.target_to_drug = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.drug_norm_attn = nn.LayerNorm(hidden_dim)
        self.target_norm_attn = nn.LayerNorm(hidden_dim)
        self.drug_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.target_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.drug_norm_ffn = nn.LayerNorm(hidden_dim)
        self.target_norm_ffn = nn.LayerNorm(hidden_dim)

    def forward(self, drug_evidence, target_evidence):
        drug_ctx, _ = self.drug_to_target(drug_evidence, target_evidence, target_evidence, need_weights=False)
        target_ctx, _ = self.target_to_drug(target_evidence, drug_evidence, drug_evidence, need_weights=False)
        drug_evidence = self.drug_norm_attn(drug_evidence + drug_ctx)
        target_evidence = self.target_norm_attn(target_evidence + target_ctx)
        drug_evidence = self.drug_norm_ffn(drug_evidence + self.drug_ffn(drug_evidence))
        target_evidence = self.target_norm_ffn(target_evidence + self.target_ffn(target_evidence))
        return drug_evidence, target_evidence


class EvidenceConsensusRouter(nn.Module):
    """Route global, token, and prior evidence through a small consensus bottleneck."""

    def __init__(self, hidden_dim, num_heads, dropout, view_dropout):
        super().__init__()
        self.view_dropout = float(view_dropout)
        self.view_embedding = nn.Parameter(torch.randn(3, hidden_dim) * 0.02)
        self.view_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.route = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def _apply_view_dropout(self, views):
        if not self.training or self.view_dropout <= 0.0:
            return views
        keep_prob = max(1.0 - self.view_dropout, 1.0e-6)
        keep = torch.rand(views.size(0), views.size(1), 1, device=views.device) < keep_prob
        empty = ~keep.any(dim=1, keepdim=True)
        keep = torch.where(empty, torch.ones_like(keep), keep)
        return views * keep.float() / keep_prob

    def forward(self, cls_pair, token_pair, prior_pair):
        raw_views = torch.stack([cls_pair, token_pair, prior_pair], dim=1)
        views = self._apply_view_dropout(raw_views)
        x = views + self.view_embedding.unsqueeze(0)
        attended, _ = self.view_attn(x, x, x, need_weights=False)
        x = self.norm_attn(x + attended)
        x = self.norm_ffn(x + self.ffn(x))

        context = x.mean(dim=1, keepdim=True).expand_as(x)
        route_logits = self.route(torch.cat([x, context, torch.abs(x - context)], dim=-1)).squeeze(-1)
        route_weights = torch.softmax(route_logits, dim=-1)
        consensus = torch.sum(route_weights.unsqueeze(-1) * x, dim=1)
        consensus = self.out_norm(consensus)
        disagreement = torch.sum(
            route_weights.unsqueeze(-1) * torch.abs(raw_views - consensus.unsqueeze(1)),
            dim=(1, 2),
        ) / max(consensus.size(-1), 1)
        return consensus, route_weights, disagreement


class BenchmarkPredictor(nn.Module):
    """Prior-conditioned benchmark predictor for cluster generalization.

    Base prediction uses pair-level compatible evidence: global CLS
    embeddings, compressed token interaction evidence, and drug/target prior
    features.
    """

    DRUG_EVIDENCE_TOKENS = 8
    TARGET_EVIDENCE_TOKENS = 16

    def __init__(self, config: Config):
        super().__init__()
        cfg = config.MODEL.BACKBONE
        base_cfg = getattr(config.MODEL, "BASE", None)
        dim = int(cfg.HIDDEN_DIM)
        dropout = float(cfg.DROPOUT)
        num_heads = int(cfg.CAN_NUM_HEADS)
        self.hidden_dim = dim
        self.prior_branch_enabled = bool(getattr(base_cfg, "PRIOR_BRANCH_ENABLED", True))
        self.prior_aware_conditioning_enabled = bool(
            getattr(base_cfg, "PRIOR_AWARE_CONDITIONING_ENABLED", True)
        )
        self.learned_evidence_compression_enabled = bool(
            getattr(base_cfg, "LEARNED_EVIDENCE_COMPRESSION_ENABLED", True)
        )
        self.compressed_token_interaction_enabled = bool(
            getattr(base_cfg, "COMPRESSED_TOKEN_INTERACTION_ENABLED", True)
        )
        self.consensus_enabled = bool(getattr(base_cfg, "CONSENSUS_ENABLED", True))

        self.macro_drug_enc = MacroEncoder(int(cfg.MACRO_DRUG_INPUT_DIM), dim, dim, dropout)
        self.macro_target_enc = MacroEncoder(int(cfg.MACRO_TARGET_INPUT_DIM), dim, dim, dropout)
        self.cls_pair_fusion = projection(dim * 4, dim, dropout)

        self.drug_prior_proj = projection(DRUG_PRIOR_DIM, dim, dropout)
        self.target_prior_proj = projection(TARGET_PRIOR_DIM, dim, dropout)
        self.prior_pair_fusion = projection(dim * 4, dim, dropout)

        self.cls_prior_conditioner = ResidualConditioner(dim, dropout)
        self.drug_token_prior_adapter = TokenPriorAdapter(dim, dropout)
        self.target_token_prior_adapter = TokenPriorAdapter(dim, dropout)

        compressor_cls = (
            LearnedEvidenceCompressor
            if self.learned_evidence_compression_enabled
            else PooledEvidenceCompressor
        )
        self.drug_compressor = compressor_cls(
            int(cfg.DRUG_TOKEN_DIM),
            dim,
            self.DRUG_EVIDENCE_TOKENS,
            num_heads,
            dropout,
        )
        self.target_compressor = compressor_cls(
            int(cfg.PROT_TOKEN_DIM),
            dim,
            self.TARGET_EVIDENCE_TOKENS,
            num_heads,
            dropout,
        )
        self.cross_evidence = CrossEvidenceBlock(dim, num_heads, dropout)
        self.token_pair_fusion = projection(dim * 8, dim, dropout)

        self.consensus_router = EvidenceConsensusRouter(
            dim,
            num_heads,
            dropout,
            float(getattr(base_cfg, "CONSENSUS_VIEW_DROPOUT", 0.05)),
        )
        self.cls_head = nn.Linear(dim, 1)
        self.token_head = nn.Linear(dim, 1)
        self.prior_head = nn.Linear(dim, 1)
        self.consensus_head = nn.Linear(dim, 1)
        self.core_head = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )
        self.legacy_fusion_head = nn.Sequential(
            nn.Linear(dim * 9 + 4, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(dim * 13 + 9, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    @staticmethod
    def _pair_features(left, right):
        return torch.cat([left, right, torch.abs(left - right), left * right], dim=-1)

    @staticmethod
    def _pool_evidence(tokens):
        mean = tokens.mean(dim=1)
        max_value = tokens.max(dim=1).values
        return mean, max_value

    def _cls_branch(self, batch):
        drug = self.macro_drug_enc(batch["macro_drug"])
        target = self.macro_target_enc(batch["macro_target"])
        return self.cls_pair_fusion(self._pair_features(drug, target))

    def _prior_branch(self, batch):
        if not self.prior_branch_enabled:
            batch_size = batch["macro_drug"].size(0)
            zero_pair = batch["macro_drug"].new_zeros(batch_size, self.hidden_dim)
            disabled = batch["macro_drug"].new_zeros(batch_size)
            return zero_pair, disabled, disabled
        drug = self.drug_prior_proj(batch["pime_drug_prior"])
        target = self.target_prior_proj(batch["pime_target_prior"])
        pair = self.prior_pair_fusion(self._pair_features(drug, target))
        enabled = torch.ones(batch["macro_drug"].size(0), device=batch["macro_drug"].device)
        return pair, enabled, enabled

    def _token_branch(self, batch, prior_pair):
        drug_evidence = self.drug_compressor(batch["drug_tokens"], batch["drug_tokens_mask"])
        target_evidence = self.target_compressor(batch["prot_tokens"], batch["prot_tokens_mask"])
        if self.prior_branch_enabled and self.prior_aware_conditioning_enabled:
            drug_evidence = self.drug_token_prior_adapter(drug_evidence, prior_pair)
            target_evidence = self.target_token_prior_adapter(target_evidence, prior_pair)
        if self.compressed_token_interaction_enabled:
            drug_evidence, target_evidence = self.cross_evidence(drug_evidence, target_evidence)
        drug_mean, drug_max = self._pool_evidence(drug_evidence)
        target_mean, target_max = self._pool_evidence(target_evidence)
        return self.token_pair_fusion(
            torch.cat(
                [
                    drug_mean,
                    target_mean,
                    drug_max,
                    target_max,
                    torch.abs(drug_mean - target_mean),
                    drug_mean * target_mean,
                    torch.abs(drug_max - target_max),
                    drug_max * target_max,
                ],
                dim=-1,
            )
        )

    def forward_components(self, batch):
        prior_pair, drug_prior_enabled, target_prior_enabled = self._prior_branch(batch)
        cls_pair = self._cls_branch(batch)
        if self.prior_branch_enabled and self.prior_aware_conditioning_enabled:
            cls_pair = self.cls_prior_conditioner(cls_pair, prior_pair)
        token_pair = self._token_branch(batch, prior_pair)

        s_cls = self.cls_head(cls_pair).view(-1)
        s_token = self.token_head(token_pair).view(-1)
        if self.prior_branch_enabled:
            s_prior = self.prior_head(prior_pair).view(-1)
        else:
            s_prior = cls_pair.new_zeros(cls_pair.size(0))
        core_features = self._pair_features(cls_pair, token_pair)
        s_core = self.core_head(core_features).view(-1)
        zero_pair = torch.zeros_like(cls_pair)
        if self.prior_branch_enabled:
            token_prior_diff = torch.abs(token_pair - prior_pair)
            token_prior_prod = token_pair * prior_pair
            cls_prior_diff = torch.abs(cls_pair - prior_pair)
            cls_prior_prod = cls_pair * prior_pair
        else:
            token_prior_diff = zero_pair
            token_prior_prod = zero_pair
            cls_prior_diff = zero_pair
            cls_prior_prod = zero_pair

        if self.consensus_enabled:
            if self.prior_branch_enabled:
                consensus_pair, route_weights, disagreement = self.consensus_router(cls_pair, token_pair, prior_pair)
                consensus_prior_diff = torch.abs(consensus_pair - prior_pair)
            else:
                consensus_pair = 0.5 * (cls_pair + token_pair)
                route_weights = torch.zeros(cls_pair.size(0), 3, dtype=cls_pair.dtype, device=cls_pair.device)
                route_weights[:, 0] = 0.5
                route_weights[:, 1] = 0.5
                disagreement = torch.abs(cls_pair - token_pair).mean(dim=-1)
                consensus_prior_diff = zero_pair
            s_consensus = self.consensus_head(consensus_pair).view(-1)
            branch_logits = torch.stack([s_core, s_cls, s_token, s_prior, s_consensus], dim=-1)
            fusion_features = torch.cat(
                [
                    cls_pair,
                    token_pair,
                    prior_pair,
                    consensus_pair,
                    torch.abs(cls_pair - token_pair),
                    cls_pair * token_pair,
                    token_prior_diff,
                    token_prior_prod,
                    cls_prior_diff,
                    cls_prior_prod,
                    torch.abs(consensus_pair - cls_pair),
                    torch.abs(consensus_pair - token_pair),
                    consensus_prior_diff,
                    branch_logits,
                    route_weights,
                    disagreement.unsqueeze(-1),
                ],
                dim=-1,
            )
            s_base = self.fusion_head(fusion_features).view(-1)
        else:
            if self.prior_branch_enabled:
                consensus_pair = (cls_pair + token_pair + prior_pair) / 3.0
            else:
                consensus_pair = 0.5 * (cls_pair + token_pair)
            if self.prior_branch_enabled:
                route_weights = torch.full(
                    (cls_pair.size(0), 3),
                    1.0 / 3.0,
                    dtype=cls_pair.dtype,
                    device=cls_pair.device,
                )
                disagreement = torch.zeros(cls_pair.size(0), dtype=cls_pair.dtype, device=cls_pair.device)
            else:
                route_weights = torch.zeros(cls_pair.size(0), 3, dtype=cls_pair.dtype, device=cls_pair.device)
                route_weights[:, 0] = 0.5
                route_weights[:, 1] = 0.5
                disagreement = torch.abs(cls_pair - token_pair).mean(dim=-1)
            s_consensus = self.consensus_head(consensus_pair).view(-1)
            branch_logits = torch.stack([s_core, s_cls, s_token, s_prior], dim=-1)
            fusion_features = torch.cat(
                [
                    cls_pair,
                    token_pair,
                    prior_pair,
                    torch.abs(cls_pair - token_pair),
                    cls_pair * token_pair,
                    token_prior_diff,
                    token_prior_prod,
                    cls_prior_diff,
                    cls_prior_prod,
                    branch_logits,
                ],
                dim=-1,
            )
            s_base = self.legacy_fusion_head(fusion_features).view(-1)

        return {
            "s_base": s_base,
            "s_basic": s_core,
            "s_cls": s_cls,
            "s_token": s_token,
            "s_evidence_base": s_prior,
            "s_consensus": s_consensus,
            "evidence_branch_logits": branch_logits,
            "evidence_consensus_weight": route_weights,
            "evidence_consensus_disagreement": disagreement,
            "evidence_drug_prior_weight": drug_prior_enabled,
            "evidence_target_prior_weight": target_prior_enabled,
        }

    def forward(self, batch):
        return self.forward_components(batch)["s_base"]
