from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BaseAdapterCapabilities:
    """Describe which audit probes are semantically available for a base."""

    consumes_explicit_prior: bool = False
    exposes_branch_components: bool = False
    requires_raw_inputs: bool = False
    external_code: bool = False


class BaseAdapterMixin:
    """Common helpers for base predictors used by the audit wrapper.

    Local adapters inherit the original model classes directly so existing
    checkpoints keep the same ``base.*`` state-dict keys.
    """

    adapter_name = "unknown"
    model_family = "unknown"
    capabilities = BaseAdapterCapabilities()

    @staticmethod
    def neutral_components(s_base):
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


class ExternalBaseAdapter(BaseAdapterMixin, torch.nn.Module):
    """Base class for third-party frozen DTI predictors.

    Subclasses should implement ``forward_external(batch)`` and return either a
    tensor of raw logits or a dict containing ``s_base``. The audit wrapper will
    receive neutral branch components for models that do not expose internals.
    """

    adapter_name = "external"
    model_family = "external"
    capabilities = BaseAdapterCapabilities(
        consumes_explicit_prior=False,
        exposes_branch_components=False,
        requires_raw_inputs=True,
        external_code=True,
    )

    def forward_external(self, batch):
        raise NotImplementedError

    def forward_components(self, batch):
        output = self.forward_external(batch)
        if isinstance(output, dict):
            if "s_base" not in output:
                raise KeyError(f"External adapter output must contain 's_base'. Got keys: {sorted(output)}")
            s_base = output["s_base"].view(-1)
            components = self.neutral_components(s_base)
            components.update(output)
            components["s_base"] = s_base
            return components
        return self.neutral_components(output.view(-1))

    def forward(self, batch):
        return self.forward_components(batch)["s_base"]
