import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)

try:
    from prettytable import PrettyTable
except ImportError:
    class PrettyTable:
        def __init__(self, header):
            self.header = [str(value) for value in header]
            self.rows = []

        def add_row(self, row):
            self.rows.append([str(value) for value in row])

        def __str__(self):
            rows = [self.header, *self.rows]
            widths = [
                max(len(row[index]) for row in rows)
                for index in range(len(self.header))
            ]
            return "\n".join(
                " | ".join(
                    value.ljust(widths[index])
                    for index, value in enumerate(row)
                )
                for row in rows
            )

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from utils.torch_utils import batch_to_device


def safe_metrics(labels, logits, threshold=0.5):
    labels = np.asarray(labels, dtype=np.float32)
    logits = np.asarray(logits, dtype=np.float32)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    predictions = (probabilities >= float(threshold)).astype(np.int64)
    metrics = {}
    if len(np.unique(labels)) > 1:
        metrics["auroc"] = float(roc_auc_score(labels, probabilities))
        metrics["auprc"] = float(average_precision_score(labels, probabilities))
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
    tn, fp, fn, tp = confusion_matrix(
        labels.astype(np.int64),
        predictions,
        labels=[0, 1],
    ).ravel()
    metrics["sensitivity"] = float(tp / max(tp + fn, 1))
    metrics["specificity"] = float(tn / max(tn + fp, 1))
    metrics["precision"] = float(tp / max(tp + fp, 1))
    metrics["accuracy"] = float((tp + tn) / max(tp + tn + fp + fn, 1))
    metrics["f1"] = float(
        2
        * metrics["precision"]
        * metrics["sensitivity"]
        / max(metrics["precision"] + metrics["sensitivity"], 1.0e-8)
    )
    metrics["mcc"] = float(
        matthews_corrcoef(labels.astype(np.int64), predictions)
    )
    return metrics


def progress_iter(iterable, args, description):
    if args is not None and getattr(args, "no_progress", False):
        return iterable
    try:
        total = len(iterable)
    except TypeError:
        total = None
    return tqdm(
        iterable,
        total=total,
        desc=description,
        leave=False,
        dynamic_ncols=True,
    )


@torch.no_grad()
def collect_logits(
    model,
    loader,
    device,
    *,
    score,
    base_only=False,
    args=None,
    description="eval",
    eval_state_dict=None,
):
    backup_state = None
    if eval_state_dict is not None:
        backup_state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        model.load_state_dict(eval_state_dict, strict=True)
    model.eval()
    try:
        labels = []
        logits = []
        for batch in progress_iter(loader, args, description):
            batch = batch_to_device(batch, device)
            output = (
                model.forward_base(batch)
                if base_only
                else model(batch)
            )
            labels.extend(
                batch["label"].detach().cpu().numpy().reshape(-1).tolist()
            )
            logits.extend(
                output[score].detach().cpu().numpy().reshape(-1).tolist()
            )
        return np.asarray(labels), np.asarray(logits)
    finally:
        if backup_state is not None:
            model.load_state_dict(backup_state, strict=True)


def find_threshold(labels, logits):
    probabilities = 1.0 / (
        1.0 + np.exp(-np.clip(np.asarray(logits), -40.0, 40.0))
    )
    labels = np.asarray(labels).astype(np.int64)
    best_threshold = 0.5
    best_mcc = -2.0
    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = (probabilities >= threshold).astype(np.int64)
        mcc = matthews_corrcoef(labels, predictions)
        if mcc > best_mcc:
            best_threshold = float(threshold)
            best_mcc = float(mcc)
    return best_threshold


def write_metric_table(path, header, rows):
    table = PrettyTable(header)
    for row in rows:
        table.add_row(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(table), encoding="utf-8")
    print(table)


def pairwise_ranking_loss(logits, labels, margin=0.20, temperature=1.0):
    positive = logits[labels > 0.5]
    negative = logits[labels <= 0.5]
    if positive.numel() == 0 or negative.numel() == 0:
        return logits.sum() * 0.0
    differences = positive.unsqueeze(1) - negative.unsqueeze(0)
    temperature = max(float(temperature), 1.0e-6)
    return temperature * F.softplus(
        (float(margin) - differences) / temperature
    ).mean()


def base_loss(output, labels, pos_weight, config):
    base_cfg = config.MODEL.BASE
    logits = output["s_base"]
    loss = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=pos_weight,
    )
    rank_weight = float(getattr(base_cfg, "RANK_LOSS_WEIGHT", 0.0))
    if rank_weight > 0.0:
        loss = loss + rank_weight * pairwise_ranking_loss(
            logits,
            labels,
            margin=float(getattr(base_cfg, "RANK_LOSS_MARGIN", 0.20)),
            temperature=float(
                getattr(base_cfg, "RANK_LOSS_TEMPERATURE", 1.0)
            ),
        )
    agreement_weight = float(
        getattr(base_cfg, "CONSENSUS_AGREEMENT_WEIGHT", 0.0)
    )
    branch_logits = output.get("evidence_branch_logits")
    if agreement_weight > 0.0 and torch.is_tensor(branch_logits):
        target = logits.detach().unsqueeze(-1).expand_as(branch_logits)
        loss = loss + agreement_weight * F.smooth_l1_loss(
            branch_logits,
            target,
        )
    return loss


def prior_loss(output, labels, pos_weight):
    return F.binary_cross_entropy_with_logits(
        output["s_prior"],
        labels,
        pos_weight=pos_weight,
    )


def build_stage_optimizer(model, learning_rate, config):
    optim_cfg = config.TRAIN.OPTIM
    groups = {
        "backbone_decay": [],
        "backbone_no_decay": [],
        "head_decay": [],
        "head_no_decay": [],
    }
    trainable_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable_parameters.append(parameter)
        is_head = "head" in name
        no_decay = name.endswith("bias") or parameter.ndim == 1
        key = ("head_" if is_head else "backbone_") + (
            "no_decay" if no_decay else "decay"
        )
        groups[key].append(parameter)
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters for this stage.")

    optimizer_groups = []
    for name, scale, weight_decay in (
        (
            "backbone_decay",
            float(optim_cfg.BACKBONE_LR_SCALE),
            float(optim_cfg.WEIGHT_DECAY),
        ),
        ("backbone_no_decay", float(optim_cfg.BACKBONE_LR_SCALE), 0.0),
        (
            "head_decay",
            float(optim_cfg.HEAD_LR_SCALE),
            float(optim_cfg.WEIGHT_DECAY),
        ),
        ("head_no_decay", float(optim_cfg.HEAD_LR_SCALE), 0.0),
    ):
        if groups[name]:
            optimizer_groups.append(
                {
                    "params": groups[name],
                    "name": name,
                    "weight_decay": weight_decay,
                    "lr": float(learning_rate) * scale,
                }
            )
    optimizer = optim.AdamW(optimizer_groups)
    return (
        optimizer,
        trainable_parameters,
        [group["lr"] for group in optimizer.param_groups],
    )


def apply_epoch_lr(optimizer, base_lrs, epoch, warmup_epochs):
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        factor = float(epoch) / float(warmup_epochs)
        for base_lr, parameter_group in zip(
            base_lrs,
            optimizer.param_groups,
        ):
            parameter_group["lr"] = float(base_lr) * factor
    elif warmup_epochs > 0 and epoch == warmup_epochs + 1:
        for base_lr, parameter_group in zip(
            base_lrs,
            optimizer.param_groups,
        ):
            parameter_group["lr"] = float(base_lr)


def init_ema_state(model):
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }


def update_ema_state(model, ema_state, decay):
    with torch.no_grad():
        for key, value in model.state_dict().items():
            shadow = ema_state[key]
            if value.is_floating_point():
                shadow.mul_(float(decay)).add_(
                    value.detach(),
                    alpha=1.0 - float(decay),
                )
            else:
                shadow.copy_(value.detach())


def cpu_state_dict_from(model, state_dict=None):
    source = model.state_dict() if state_dict is None else state_dict
    return {
        key: value.detach().cpu().clone()
        for key, value in source.items()
    }


def compute_pos_weight(loader, device):
    dataset = getattr(loader, "dataset", None)
    if dataset is not None and hasattr(dataset, "df") and "Y" in dataset.df:
        labels = np.asarray(dataset.df["Y"], dtype=np.float32)
    else:
        labels = np.concatenate(
            [
                batch["label"].numpy().reshape(-1)
                for batch in loader
            ]
        ).astype(np.float32)
    positives = max(float((labels > 0.5).sum()), 1.0)
    negatives = max(float((labels <= 0.5).sum()), 1.0)
    return torch.tensor(
        negatives / positives,
        dtype=torch.float32,
        device=device,
    )


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    config,
    pos_weight,
    stage,
    trainable_parameters,
    *,
    args,
    epoch,
    ema_state,
):
    model.train()
    if stage == "prior":
        model.enforce_frozen_eval()
    losses = []
    iterator = progress_iter(loader, args, f"{stage} epoch {epoch}")
    for batch in iterator:
        batch = batch_to_device(batch, device)
        labels = batch["label"].view(-1).float()
        optimizer.zero_grad(set_to_none=True)
        output = (
            model.forward_base(batch)
            if stage == "base"
            else model(batch)
        )
        loss = (
            base_loss(output, labels, pos_weight, config)
            if stage == "base"
            else prior_loss(output, labels, pos_weight)
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {stage} loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            float(config.TRAIN.OPTIM.GRAD_CLIP_NORM),
        )
        optimizer.step()
        if ema_state is not None:
            eval_cfg = config.TRAIN.EVAL
            if epoch >= int(eval_cfg.MODEL_EMA_START_EPOCH):
                update_ema_state(
                    model,
                    ema_state,
                    float(eval_cfg.MODEL_EMA_DECAY),
                )
        losses.append(float(loss.detach().cpu()))
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                loss=f"{np.mean(losses):.4f}",
                refresh=False,
            )
    return float(np.mean(losses)) if losses else 0.0


def save_checkpoint(path, model, metadata, state_dict=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": cpu_state_dict_from(model, state_dict),
            "meta": metadata,
        },
        path,
    )


def load_checkpoint(path, model, device, strict=True):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=bool(strict))


def load_base_checkpoint(path, model, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    base_state = {
        key[len("base.") :]: value
        for key, value in state.items()
        if str(key).startswith("base.")
    }
    if not base_state:
        raise RuntimeError(f"No base.* weights found in checkpoint: {path}")
    model.base.load_state_dict(base_state, strict=True)


def train_stage(args, config, model, loaders, device, output_dir, stage):
    train_loader, validation_loader, _ = loaders
    if stage == "base":
        model.train_base_only()
        epochs = int(args.base_epochs)
        learning_rate = float(args.base_lr)
        checkpoint_name = "base_best.pth"
        patience = int(
            args.patience
            if args.patience is not None
            else config.TRAIN.OPTIM.BASE_PATIENCE
        )
        score_name = "s_base"
        base_only = True
    elif stage == "prior":
        model.freeze_base()
        model.train_prior_only()
        epochs = int(args.prior_epochs)
        learning_rate = float(args.prior_lr)
        checkpoint_name = "prior_best.pth"
        patience = int(
            args.patience
            if args.patience is not None
            else config.TRAIN.OPTIM.PRIOR_PATIENCE
        )
        score_name = "s_prior"
        base_only = False
    else:
        raise ValueError(f"Unsupported training stage: {stage}")

    optimizer, trainable_parameters, base_lrs = build_stage_optimizer(
        model,
        learning_rate,
        config,
    )
    optim_cfg = config.TRAIN.OPTIM
    warmup_epochs = int(optim_cfg.WARMUP_EPOCHS)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs - warmup_epochs, 1),
        eta_min=float(optim_cfg.ETA_MIN),
    )
    eval_cfg = config.TRAIN.EVAL
    use_ema = bool(eval_cfg.USE_MODEL_EMA)
    ema_state = init_ema_state(model) if use_ema else None
    pos_weight = compute_pos_weight(train_loader, device)

    best_score = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    rows = []
    for epoch in range(1, epochs + 1):
        apply_epoch_lr(optimizer, base_lrs, epoch, warmup_epochs)
        loss = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            config,
            pos_weight,
            stage,
            trainable_parameters,
            args=args,
            epoch=epoch,
            ema_state=ema_state,
        )
        eval_state = (
            ema_state
            if use_ema and bool(eval_cfg.EMA_USE_FOR_EVAL)
            else None
        )
        labels, logits = collect_logits(
            model,
            validation_loader,
            device,
            score=score_name,
            base_only=base_only,
            args=args,
            description=f"{stage} validation {epoch}",
            eval_state_dict=eval_state,
        )
        metrics = safe_metrics(labels, logits)
        selection_score = metrics["auroc"] + metrics["auprc"]
        rows.append(
            [
                epoch,
                f"{loss:.4f}",
                f"{metrics['auroc']:.4f}",
                f"{metrics['auprc']:.4f}",
            ]
        )
        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(
                output_dir / checkpoint_name,
                model,
                {
                    "stage": stage,
                    "epoch": epoch,
                    "score": selection_score,
                },
                state_dict=eval_state,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
        if epoch > warmup_epochs:
            scheduler.step()

    write_metric_table(
        output_dir / f"{stage}_train_markdowntable.txt",
        ["Epoch", "Loss", "Val AUROC", "Val AUPRC"],
        rows,
    )
    print(
        f"[{stage}] best_epoch={best_epoch} best_score={best_score:.4f} "
        f"saved={output_dir / checkpoint_name}"
    )
    load_checkpoint(
        output_dir / checkpoint_name,
        model,
        device,
        strict=True,
    )


def _evaluation_row(name, labels, logits, threshold):
    metrics = safe_metrics(labels, logits, threshold=threshold)
    return [
        name,
        f"{metrics['auroc']:.4f}",
        f"{metrics['auprc']:.4f}",
        f"{metrics['sensitivity']:.4f}",
        f"{metrics['precision']:.4f}",
        f"{metrics['f1']:.4f}",
        f"{metrics['mcc']:.4f}",
        f"{metrics['specificity']:.4f}",
        f"{metrics['accuracy']:.4f}",
        f"{threshold:.3f}",
    ], metrics


def evaluate_final(model, loaders, device, output_dir, args=None):
    _, validation_loader, test_loader = loaders
    rows = []
    saved_metrics = {}
    for name, score, base_only in (
        ("frozen_base", "s_base", True),
        ("prior_evidence", "s_prior", False),
    ):
        validation_labels, validation_logits = collect_logits(
            model,
            validation_loader,
            device,
            score=score,
            base_only=base_only,
            args=args,
            description=f"final {name} validation",
        )
        threshold = find_threshold(validation_labels, validation_logits)
        test_labels, test_logits = collect_logits(
            model,
            test_loader,
            device,
            score=score,
            base_only=base_only,
            args=args,
            description=f"final {name} test",
        )
        row, metrics = _evaluation_row(
            name,
            test_labels,
            test_logits,
            threshold,
        )
        rows.append(row)
        saved_metrics[name] = {
            "test_metrics": metrics,
            "threshold": threshold,
            "score": score,
        }

    write_metric_table(
        output_dir / "test_markdowntable.txt",
        [
            "Model",
            "AUROC",
            "AUPRC",
            "Sensitivity",
            "Precision",
            "F1",
            "MCC",
            "Specificity",
            "Accuracy",
            "Threshold",
        ],
        rows,
    )
    torch.save(saved_metrics, output_dir / "result_metrics.pt")
