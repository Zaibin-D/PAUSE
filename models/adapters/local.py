from models.adapters.base import BaseAdapterCapabilities, BaseAdapterMixin
from models.benchmark_predictor import BenchmarkPredictor
from models.can_dti_model import CANDTIModel


class BenchmarkBaseAdapter(BaseAdapterMixin, BenchmarkPredictor):
    """Full PIME benchmark base exposed through the adapter contract."""

    adapter_name = "full_base"
    model_family = "benchmark"
    capabilities = BaseAdapterCapabilities(
        consumes_explicit_prior=True,
        exposes_branch_components=True,
        requires_raw_inputs=False,
        external_code=False,
    )

    def __init__(self, config):
        super().__init__(config)


class CANOnlyBaseAdapter(BaseAdapterMixin, CANDTIModel):
    """CAN-only base exposed through the same component contract."""

    adapter_name = "can_only"
    model_family = "can_only"
    capabilities = BaseAdapterCapabilities(
        consumes_explicit_prior=False,
        exposes_branch_components=False,
        requires_raw_inputs=False,
        external_code=False,
    )

    def __init__(self, config):
        super().__init__(config)

    def forward_components(self, batch):
        s_base = super().forward(batch).view(-1)
        return self.neutral_components(s_base)
