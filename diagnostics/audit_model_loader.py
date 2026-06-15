"""Shared model/config loading utilities for PAUSE input generation.

The diagnostic scripts should not know whether the frozen base is Full Base,
CAN-only, or an external adapter such as TAPB. They load a single audit model
wrapper and ask it for named score logits.
"""
import importlib
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings(
    "ignore",
    message=r"Converting mask without torch\.bool dtype to bool.*",
    category=UserWarning,
)


MODEL_SPECS = {
    "can": {
        "config_module": "configs",
        "config_class": "Config",
        "model_module": "models.can_dti_model",
        "model_class": "CANDTIModel",
    },
    "pime": {
        "config_module": "configs",
        "config_class": "Config",
        "model_module": "models.pause_model",
        "model_class": "PAUSEModel",
    },
    "pause": {
        "config_module": "configs",
        "config_class": "Config",
        "model_module": "models.pause_model",
        "model_class": "PAUSEModel",
    },
}


@dataclass
class AuditModelBundle:
    config: object
    model: torch.nn.Module
    device: torch.device
    checkpoint: object
    loaded_config: object
    model_root: Path


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_config_overrides(target, overrides, prefix=""):
    if overrides is None:
        return
    if not isinstance(overrides, dict):
        raise TypeError(f"Config override at {prefix or '<root>'} must be a mapping.")
    for key, value in overrides.items():
        key = str(key)
        attr_name = key if hasattr(target, key) else key.upper()
        path = f"{prefix}.{attr_name}" if prefix else attr_name
        if not hasattr(target, attr_name):
            if isinstance(value, dict):
                setattr(target, attr_name, SimpleNamespace())
            elif isinstance(target, SimpleNamespace) or prefix.endswith("SPLIT_FILES"):
                setattr(target, attr_name, value)
                continue
            else:
                raise KeyError(f"Unknown config key: {path}")
        current_value = getattr(target, attr_name)
        if isinstance(value, dict):
            apply_config_overrides(current_value, value, path)
        else:
            setattr(target, attr_name, value)


def resolve_model_root(args):
    root_arg = getattr(args, "model_root", None)
    root = Path(root_arg) if root_arg else ROOT
    if not root.exists():
        raise FileNotFoundError(f"Model root not found: {root}")
    return root.resolve()


def _import_from_root(module_name, symbol_name, root):
    root = Path(root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def load_config_and_model_class(args):
    model_type = str(getattr(args, "model_type", "pime") or "pime").lower()
    if model_type not in MODEL_SPECS:
        raise ValueError(f"Unknown diagnostic model_type={model_type!r}. Known: {sorted(MODEL_SPECS)}")
    model_root = resolve_model_root(args)
    spec = MODEL_SPECS[model_type]
    ConfigClass = _import_from_root(spec["config_module"], spec["config_class"], model_root)
    ModelClass = _import_from_root(spec["model_module"], spec["model_class"], model_root)
    return ConfigClass(), ModelClass, model_root


def resolve_config_path(config_arg, model_root=None):
    if config_arg is None:
        return None
    config_path = Path(config_arg)
    if config_path.exists():
        return config_path
    candidates = [
        Path.cwd() / str(config_arg),
        ROOT / str(config_arg),
        ROOT / "config_files" / str(config_arg),
    ]
    if model_root is not None:
        model_root = Path(model_root)
        candidates.extend([model_root / str(config_arg), model_root / "config_files" / str(config_arg)])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Config file not found: {config_arg}")


def load_yaml_config(config, config_arg, model_root=None):
    if config_arg and "," in str(config_arg):
        loaded = []
        for item in str(config_arg).split(","):
            item = item.strip()
            if item:
                found = load_yaml_config(config, item, model_root=model_root)
                if found is not None:
                    loaded.append(found)
        return loaded
    config_path = resolve_config_path(config_arg, model_root=model_root)
    if config_path is None:
        return None
    with config_path.open("r", encoding="utf-8") as fp:
        overrides = yaml.safe_load(fp) or {}
    apply_config_overrides(config, overrides)
    return config_path


def _is_disabled_checkpoint(value):
    return str(value or "").strip().lower() in {"", "none", "null", "false", "no"}


def resolve_checkpoint(args, required=True):
    checkpoint_arg = getattr(args, "checkpoint", None)
    if checkpoint_arg and not _is_disabled_checkpoint(checkpoint_arg):
        checkpoint = Path(checkpoint_arg)
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        return checkpoint
    if checkpoint_arg and _is_disabled_checkpoint(checkpoint_arg):
        return None

    run_dir_arg = getattr(args, "run_dir", None)
    if not run_dir_arg:
        if required:
            raise ValueError("Provide either --checkpoint or --run-dir.")
        return None
    run_dir = Path(run_dir_arg)
    candidates = sorted(run_dir.glob("best_*.pth"), key=lambda path: path.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    fallback = run_dir / "last_epoch.pth"
    if fallback.exists():
        return fallback
    if not required:
        return None
    raise FileNotFoundError(f"No best_*.pth or last_epoch.pth found in {run_dir}")


def load_checkpoint_state(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"]
    return state


def load_audit_model(args):
    config, model_class, model_root = load_config_and_model_class(args)
    loaded_config = load_yaml_config(config, getattr(args, "config", None), model_root=model_root)

    if hasattr(config, "DATA"):
        if hasattr(config.DATA, "PATHS") and hasattr(args, "root"):
            config.DATA.PATHS.ROOT_DIR = args.root
        if hasattr(config.DATA, "LOADER") and hasattr(args, "num_workers"):
            config.DATA.LOADER.NUM_WORKERS = int(args.num_workers)
    if hasattr(config, "TRAIN") and hasattr(config.TRAIN, "OPTIM") and hasattr(args, "batch_size"):
        config.TRAIN.OPTIM.BATCH_SIZE = int(args.batch_size)

    device = torch.device(getattr(args, "device", None) if getattr(args, "device", None) else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model_class(config).to(device)
    base_cfg = getattr(getattr(config, "MODEL", None), "BASE", None)
    custom_adapter = bool(str(getattr(base_cfg, "ADAPTER_MODULE", "") or "").strip())
    external_checkpoint = bool(str(getattr(base_cfg, "EXTERNAL_CHECKPOINT", "") or "").strip())
    checkpoint_required = not (custom_adapter or external_checkpoint)
    checkpoint = resolve_checkpoint(args, required=checkpoint_required)
    if checkpoint is not None:
        model.load_state_dict(load_checkpoint_state(checkpoint, device), strict=True)
    else:
        print(
            "[audit_model_loader] no PAUSE wrapper checkpoint loaded; "
            "assuming the external adapter loads its own frozen weights. "
            "Prior logits require a separately trained prior head."
        )
    model.eval()
    return AuditModelBundle(
        config=config,
        model=model,
        device=device,
        checkpoint=checkpoint,
        loaded_config=loaded_config,
        model_root=model_root,
    )


def select_logits(model_output, score_name):
    if isinstance(model_output, dict):
        key = f"s_{score_name}"
        if key not in model_output:
            raise KeyError(f"Model output does not contain {key!r}. Available keys: {sorted(model_output)}")
        return model_output[key].view(-1)
    if score_name != "full":
        raise ValueError(f"Tensor-output models only support --score full, got {score_name!r}")
    return model_output.view(-1)
