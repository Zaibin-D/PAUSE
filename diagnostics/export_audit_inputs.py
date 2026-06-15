"""Export frozen base and prior logits consumed by the PAUSE audit."""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit_framework.data import attach_source_support
from data_loader.data_loader import get_dataloader
from diagnostics.audit_model_loader import load_audit_model
from utils.torch_utils import batch_to_device


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _metrics(labels, probabilities):
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    valid = np.isfinite(labels) & np.isfinite(probabilities)
    labels = labels[valid]
    probabilities = probabilities[valid]
    if len(labels) == 0 or np.unique(labels).size < 2:
        return {"auroc": np.nan, "auprc": np.nan}
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
    }


def _to_numpy(tensor):
    return tensor.detach().cpu().numpy().reshape(-1)


def run(args):
    setup_seed(args.seed)
    bundle = load_audit_model(args)
    loader = get_dataloader(
        args.data,
        args.split,
        args.phase,
        bundle.config,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    rows = []
    processed = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            batch = batch_to_device(batch, bundle.device)
            output = bundle.model(batch)
            s_base = _to_numpy(output["s_base"])
            s_prior = _to_numpy(output["s_prior"])
            labels = _to_numpy(batch["label"])
            drug_ids = _to_numpy(batch["dr_id"])
            target_ids = _to_numpy(batch["pr_id"])
            p_base = _sigmoid(s_base)
            p_prior = _sigmoid(s_prior)
            base_pred = (p_base >= 0.5).astype(float)

            for index in range(len(labels)):
                rows.append(
                    {
                        "data": args.data,
                        "split": args.split,
                        "phase": args.phase,
                        "score": "base",
                        "config": str(bundle.loaded_config or ""),
                        "checkpoint": str(bundle.checkpoint or ""),
                        "dr_id": int(drug_ids[index]),
                        "pr_id": int(target_ids[index]),
                        "label": float(labels[index]),
                        "s_score": float(s_base[index]),
                        "s_base": float(s_base[index]),
                        "s_full": float(s_base[index]),
                        "s_prior": float(s_prior[index]),
                        "p_base": float(p_base[index]),
                        "p_prior": float(p_prior[index]),
                        "base_pred": float(base_pred[index]),
                        "base_correct": float(
                            base_pred[index] == labels[index]
                        ),
                        "base_confidence": float(
                            2.0 * abs(p_base[index] - 0.5)
                        ),
                        "base_uncertainty": float(
                            1.0 - 2.0 * abs(p_base[index] - 0.5)
                        ),
                        "base_prior_conflict": float(
                            abs(p_base[index] - p_prior[index])
                        ),
                    }
                )
            processed += len(labels)
            if args.progress_every and batch_index % args.progress_every == 0:
                print(
                    f"[export] {args.data}/{args.phase} "
                    f"batches={batch_index} samples={processed}",
                    flush=True,
                )

    table = attach_source_support(
        pd.DataFrame(rows),
        dataset=args.data,
        dataset_root=args.root,
    )
    summary_rows = []
    for name, column in (("base", "p_base"), ("prior", "p_prior")):
        metrics = _metrics(table["label"], table[column])
        summary_rows.append(
            {
                "data": args.data,
                "split": args.split,
                "phase": args.phase,
                "channel": name,
                "rows": int(len(table)),
                **metrics,
            }
        )
    summary = pd.DataFrame(summary_rows)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = (
        Path(args.summary)
        if args.summary
        else output_path.with_name(output_path.stem + "_summary.csv")
    )
    table.to_csv(output_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"[write] {output_path}")
    print(f"[write] {summary_path}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="cluster", choices=["cluster"])
    parser.add_argument(
        "--phase",
        default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument("--root", default="./datasets")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-root", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-type", default="pime", choices=["pime"])
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", default=None)
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
