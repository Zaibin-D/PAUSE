import importlib
import sys
from pathlib import Path

from models.adapters.external import EXTERNAL_MODEL_TYPES, external_adapter_error
from models.adapters.local import BenchmarkBaseAdapter, CANOnlyBaseAdapter


LOCAL_ADAPTERS = {
    "benchmark": BenchmarkBaseAdapter,
    "full": BenchmarkBaseAdapter,
    "full_base": BenchmarkBaseAdapter,
    "pime_base": BenchmarkBaseAdapter,
    "can": CANOnlyBaseAdapter,
    "can_only": CANOnlyBaseAdapter,
    "can-only": CANOnlyBaseAdapter,
    "candti": CANOnlyBaseAdapter,
}


def normalize_base_model_type(model_type):
    return str(model_type or "benchmark").strip().lower()


def _custom_adapter_spec(base_cfg):
    module_name = str(getattr(base_cfg, "ADAPTER_MODULE", "") or "").strip()
    class_name = str(getattr(base_cfg, "ADAPTER_CLASS", "") or "").strip()
    if bool(module_name) != bool(class_name):
        raise ValueError("MODEL.BASE.ADAPTER_MODULE and MODEL.BASE.ADAPTER_CLASS must be set together.")
    if not module_name:
        return None
    return module_name, class_name


def _create_custom_adapter(config, base_cfg):
    spec = _custom_adapter_spec(base_cfg)
    if spec is None:
        return None
    module_name, class_name = spec
    code_root = str(getattr(base_cfg, "EXTERNAL_CODE_ROOT", "") or "").strip()
    if code_root:
        root = Path(code_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"MODEL.BASE.EXTERNAL_CODE_ROOT does not exist: {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    module = importlib.import_module(module_name)
    adapter_cls = getattr(module, class_name)
    return adapter_cls(config)


def create_base_adapter(config):
    base_cfg = getattr(config.MODEL, "BASE", None)
    custom_adapter = _create_custom_adapter(config, base_cfg)
    if custom_adapter is not None:
        return custom_adapter
    model_type = normalize_base_model_type(getattr(base_cfg, "MODEL_TYPE", "benchmark"))
    adapter_cls = LOCAL_ADAPTERS.get(model_type)
    if adapter_cls is not None:
        return adapter_cls(config)
    if model_type == "tapb":
        from models.adapters.tapb_adapter import TAPBBaseAdapter

        return TAPBBaseAdapter(config)
    if model_type == "drugban":
        from models.adapters.drugban_adapter import DrugBANBaseAdapter

        return DrugBANBaseAdapter(config)
    if model_type in EXTERNAL_MODEL_TYPES:
        raise external_adapter_error(model_type)
    known = sorted([*LOCAL_ADAPTERS, *EXTERNAL_MODEL_TYPES])
    raise ValueError(f"Unknown base MODEL_TYPE: {model_type}. Known adapter types: {known}")
