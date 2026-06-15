import math

import torch
import torch.nn as nn


class CANLayer(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=8, group_size=1, dropout=0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.group_size = max(int(group_size), 1)

        self.drug_q = nn.Linear(hidden_dim, hidden_dim)
        self.drug_k = nn.Linear(hidden_dim, hidden_dim)
        self.drug_v = nn.Linear(hidden_dim, hidden_dim)
        self.prot_q = nn.Linear(hidden_dim, hidden_dim)
        self.prot_k = nn.Linear(hidden_dim, hidden_dim)
        self.prot_v = nn.Linear(hidden_dim, hidden_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.drug_out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.prot_out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.drug_attn_ln = nn.LayerNorm(hidden_dim)
        self.prot_attn_ln = nn.LayerNorm(hidden_dim)

        ffn_inner = hidden_dim * 4
        self.drug_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_inner, hidden_dim),
            nn.Dropout(dropout),
        )
        self.prot_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_inner, hidden_dim),
            nn.Dropout(dropout),
        )
        self.drug_ffn_ln = nn.LayerNorm(hidden_dim)
        self.prot_ffn_ln = nn.LayerNorm(hidden_dim)

    def _group_embeddings(self, x, mask):
        mask = mask.bool()
        if self.group_size == 1:
            return x, mask

        batch_size, seq_len, hidden_dim = x.shape
        groups = (seq_len + self.group_size - 1) // self.group_size
        padded_len = groups * self.group_size
        pad_len = padded_len - seq_len

        if pad_len > 0:
            x = torch.cat([x, x.new_zeros(batch_size, pad_len, hidden_dim)], dim=1)
            mask = torch.cat([mask, mask.new_zeros(batch_size, pad_len)], dim=1)

        x_group = x.view(batch_size, groups, self.group_size, hidden_dim)
        mask_group = mask.view(batch_size, groups, self.group_size)
        weights = mask_group.unsqueeze(-1).float()
        x_grouped = (x_group * weights).sum(dim=2) / weights.sum(dim=2).clamp(min=1.0)
        mask_grouped = mask_group.any(dim=2)
        return x_grouped, mask_grouped

    def _ungroup_embeddings(self, x_grouped, original_len):
        if self.group_size == 1:
            return x_grouped

        batch_size, groups, hidden_dim = x_grouped.shape
        x_ungrouped = x_grouped.unsqueeze(2).expand(batch_size, groups, self.group_size, hidden_dim).contiguous()
        x_ungrouped = x_ungrouped.view(batch_size, groups * self.group_size, hidden_dim)
        return x_ungrouped[:, :original_len, :]

    def _attend(self, q, k, v, mask_row, mask_col):
        mask_row = mask_row.bool()
        mask_col = mask_col.bool()
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        batch_size, num_heads, len_row, _ = q.size()
        len_col = k.size(-2)

        row_mask = mask_row.unsqueeze(1).unsqueeze(3).expand(batch_size, num_heads, len_row, len_col)
        col_mask = mask_col.unsqueeze(1).unsqueeze(2).expand(batch_size, num_heads, len_row, len_col)
        pair_mask = row_mask & col_mask

        scores = scores.masked_fill(~pair_mask, torch.finfo(scores.dtype).min)
        alpha = torch.softmax(scores, dim=-1)
        alpha = self.attn_dropout(alpha)
        alpha = torch.where(row_mask, alpha, torch.zeros_like(alpha))
        return torch.matmul(alpha, v)

    def forward(self, drug_tokens, prot_tokens, mask_drug, mask_prot):
        batch_size, drug_len, hidden_dim = drug_tokens.size()
        prot_len = prot_tokens.size(1)
        num_heads = self.num_heads
        head_dim = self.head_dim

        drug_grouped, drug_group_mask = self._group_embeddings(drug_tokens, mask_drug)
        prot_grouped, prot_group_mask = self._group_embeddings(prot_tokens, mask_prot)

        drug_group_len = drug_grouped.size(1)
        prot_group_len = prot_grouped.size(1)

        dq = self.drug_q(drug_grouped).view(batch_size, drug_group_len, num_heads, head_dim).transpose(1, 2)
        dk = self.drug_k(drug_grouped).view(batch_size, drug_group_len, num_heads, head_dim).transpose(1, 2)
        dv = self.drug_v(drug_grouped).view(batch_size, drug_group_len, num_heads, head_dim).transpose(1, 2)

        pq = self.prot_q(prot_grouped).view(batch_size, prot_group_len, num_heads, head_dim).transpose(1, 2)
        pk = self.prot_k(prot_grouped).view(batch_size, prot_group_len, num_heads, head_dim).transpose(1, 2)
        pv = self.prot_v(prot_grouped).view(batch_size, prot_group_len, num_heads, head_dim).transpose(1, 2)

        drug_self = self._attend(dq, dk, dv, drug_group_mask, drug_group_mask)
        drug_cross = self._attend(dq, pk, pv, drug_group_mask, prot_group_mask)
        prot_self = self._attend(pq, pk, pv, prot_group_mask, prot_group_mask)
        prot_cross = self._attend(pq, dk, dv, prot_group_mask, drug_group_mask)

        drug_combined = 0.5 * (drug_self + drug_cross)
        prot_combined = 0.5 * (prot_self + prot_cross)

        drug_combined = drug_combined.transpose(1, 2).contiguous().view(batch_size, drug_group_len, hidden_dim)
        prot_combined = prot_combined.transpose(1, 2).contiguous().view(batch_size, prot_group_len, hidden_dim)

        drug_combined = self._ungroup_embeddings(drug_combined, drug_len)
        prot_combined = self._ungroup_embeddings(prot_combined, prot_len)

        drug_attn = self.drug_attn_ln(drug_tokens + self.proj_drop(self.drug_out_proj(drug_combined)))
        prot_attn = self.prot_attn_ln(prot_tokens + self.proj_drop(self.prot_out_proj(prot_combined)))

        drug_updated = self.drug_ffn_ln(drug_attn + self.drug_ffn(drug_attn))
        prot_updated = self.prot_ffn_ln(prot_attn + self.prot_ffn(prot_attn))
        return drug_updated, prot_updated
