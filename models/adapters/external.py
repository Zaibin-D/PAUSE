from models.adapters.base import BaseAdapterCapabilities


EXTERNAL_MODEL_TYPES = {
    "tapb",
    "drugban",
    "deepdta",
    "deepconvdti",
    "graphdta",
    "psichic",
}


EXTERNAL_CAPABILITIES = {
    "tapb": BaseAdapterCapabilities(
        consumes_explicit_prior=False,
        exposes_branch_components=False,
        requires_raw_inputs=True,
        external_code=True,
    ),
    "drugban": BaseAdapterCapabilities(
        consumes_explicit_prior=False,
        exposes_branch_components=False,
        requires_raw_inputs=True,
        external_code=True,
    ),
    "deepdta": BaseAdapterCapabilities(
        consumes_explicit_prior=False,
        exposes_branch_components=False,
        requires_raw_inputs=True,
        external_code=True,
    ),
    "deepconvdti": BaseAdapterCapabilities(
        consumes_explicit_prior=False,
        exposes_branch_components=False,
        requires_raw_inputs=True,
        external_code=True,
    ),
    "graphdta": BaseAdapterCapabilities(
        consumes_explicit_prior=False,
        exposes_branch_components=False,
        requires_raw_inputs=True,
        external_code=True,
    ),
    "psichic": BaseAdapterCapabilities(
        consumes_explicit_prior=False,
        exposes_branch_components=True,
        requires_raw_inputs=True,
        external_code=True,
    ),
}


def external_adapter_error(model_type):
    return NotImplementedError(
        f"MODEL.BASE.MODEL_TYPE={model_type!r} is reserved for an external base adapter, "
        "but no local adapter implementation is installed yet. Keep third-party model "
        "code in its own repository or vendor directory, then add a thin adapter under "
        "models/adapters/ that maps the audit batch to that model's inputs and returns "
        "a dict containing at least {'s_base': raw_logits}."
    )
