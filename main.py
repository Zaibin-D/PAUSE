import argparse
import os
import random
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from configs import Config
from data_loader.data_loader import get_dataloader
from models.pause_model import PAUSEModel
from train import (
    evaluate_final,
    load_base_checkpoint,
    load_checkpoint,
    train_stage,
)


EXTERNAL_BASE_TYPES = {
    "tapb",
    "drugban",
    "deepdta",
    "deepconvdti",
    "graphdta",
    "psichic",
}

warnings.filterwarnings(
    "ignore",
    message=r"Converting mask without torch\.bool dtype to bool.*",
    category=UserWarning,
)


def setup_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def apply_config_overrides(target, overrides, prefix=""):
    if overrides is None:
        return
    if not isinstance(overrides, dict):
        raise TypeError(
            f"Config override at {prefix or '<root>'} must be a mapping."
        )
    for key, value in overrides.items():
        attr_name = key if hasattr(target, key) else str(key).upper()
        path = f"{prefix}.{attr_name}" if prefix else attr_name
        if not hasattr(target, attr_name):
            if isinstance(value, dict):
                setattr(target, attr_name, SimpleNamespace())
            elif isinstance(target, SimpleNamespace) or prefix.endswith(
                "SPLIT_FILES"
            ):
                setattr(target, attr_name, value)
                continue
            else:
                raise KeyError(f"Unknown config key: {path}")
        current = getattr(target, attr_name)
        if isinstance(value, dict):
            apply_config_overrides(current, value, path)
        else:
            setattr(target, attr_name, value)


def load_config(config, path):
    if not path:
        return None
    if isinstance(path, (list, tuple)):
        return [
            loaded
            for item in path
            if (loaded := load_config(config, item)) is not None
        ]
    if "," in str(path):
        return load_config(
            config,
            [item.strip() for item in str(path).split(",") if item.strip()],
        )
    for candidate in (Path(path), Path("config_files") / path):
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as handle:
                apply_config_overrides(config, yaml.safe_load(handle) or {})
            return candidate
    raise FileNotFoundError(path)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train or evaluate PAUSE base and prior evidence channels."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="cluster", choices=["cluster"])
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--stage",
        default="all",
        choices=["base", "prior", "all", "eval"],
    )
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--base-epochs", type=int, default=None)
    parser.add_argument("--prior-epochs", type=int, default=None)
    parser.add_argument("--base-lr", type=float, default=None)
    parser.add_argument("--prior-lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--prior-checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    setup_seed(args.seed, deterministic=args.deterministic)
    config = Config()
    loaded = load_config(config, args.config)

    optim_cfg = config.TRAIN.OPTIM
    if args.batch_size is not None:
        optim_cfg.BATCH_SIZE = int(args.batch_size)
    args.base_epochs = int(args.base_epochs or optim_cfg.BASE_EPOCHS)
    args.prior_epochs = int(args.prior_epochs or optim_cfg.PRIOR_EPOCHS)
    args.base_lr = float(args.base_lr or optim_cfg.BASE_LR)
    args.prior_lr = float(args.prior_lr or optim_cfg.PRIOR_LR)
    if args.num_workers is None:
        args.num_workers = int(config.DATA.LOADER.NUM_WORKERS)
    config.DATA.LOADER.NUM_WORKERS = int(args.num_workers)
    if args.data_root is not None:
        config.DATA.PATHS.ROOT_DIR = args.data_root

    batch_size = int(optim_cfg.BATCH_SIZE)
    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir = (
        Path(args.output_root)
        / args.data
        / args.split
        / f"seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[PAUSE] data={args.data} split={args.split} seed={args.seed} "
        f"device={device.type} output={output_dir}"
    )
    if loaded is not None:
        print(f"[PAUSE] config={loaded}")

    train_loader = get_dataloader(
        args.data,
        args.split,
        "train",
        config,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    validation_loader = get_dataloader(
        args.data,
        args.split,
        "val",
        config,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = get_dataloader(
        args.data,
        args.split,
        "test",
        config,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    loaders = (train_loader, validation_loader, test_loader)

    model = PAUSEModel(config).to(device)
    (output_dir / "model_architecture.txt").write_text(
        str(model),
        encoding="utf-8",
    )

    base_cfg = config.MODEL.BASE
    base_model_type = str(base_cfg.MODEL_TYPE).lower()
    external_base = (
        base_model_type in EXTERNAL_BASE_TYPES
        or bool(str(base_cfg.EXTERNAL_CHECKPOINT).strip())
        or bool(str(base_cfg.ADAPTER_MODULE).strip())
    )

    if args.stage == "base" and external_base:
        raise ValueError("External base adapters are frozen and not retrained.")
    if args.stage in {"base", "all"} and not external_base:
        train_stage(
            args,
            config,
            model,
            loaders,
            device,
            output_dir,
            "base",
        )

    if args.stage in {"prior", "all", "eval"} and not external_base:
        base_checkpoint = Path(
            args.base_checkpoint or output_dir / "base_best.pth"
        )
        if base_checkpoint.exists():
            load_base_checkpoint(base_checkpoint, model, device)
        elif args.stage != "all":
            raise FileNotFoundError(base_checkpoint)

    if args.stage in {"prior", "all"}:
        train_stage(
            args,
            config,
            model,
            loaders,
            device,
            output_dir,
            "prior",
        )
    elif args.stage == "eval":
        prior_checkpoint = Path(
            args.prior_checkpoint or output_dir / "prior_best.pth"
        )
        if not prior_checkpoint.exists():
            raise FileNotFoundError(prior_checkpoint)
        load_checkpoint(prior_checkpoint, model, device, strict=True)

    evaluate_final(model, loaders, device, output_dir, args=args)


if __name__ == "__main__":
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )
    main()
