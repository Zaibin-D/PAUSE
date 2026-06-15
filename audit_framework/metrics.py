"""Calibration metrics for the PAUSE audit framework."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calibration_metrics(
    table: pd.DataFrame,
    error_risk: pd.Series | np.ndarray,
    *,
    mask: pd.Series | np.ndarray | None = None,
    bins: int = 10,
) -> dict[str, float | int]:
    """Evaluate whether a score estimates frozen-prediction error probability."""

    labels = pd.to_numeric(table.get("base_wrong"), errors="coerce")
    scores = pd.Series(np.asarray(error_risk, dtype=float), index=table.index)
    if mask is None:
        selected = pd.Series(True, index=table.index)
    else:
        selected = pd.Series(np.asarray(mask, dtype=bool), index=table.index)
    valid = selected & labels.notna() & scores.notna()
    y = labels.loc[valid].to_numpy(dtype=float)
    p = scores.loc[valid].clip(0.0, 1.0).to_numpy(dtype=float)
    if not len(y):
        return {
            "calibration_n": 0,
            "error_count": 0.0,
            "error_rate": np.nan,
            "mean_predicted_error_risk": np.nan,
            "brier_error_risk": np.nan,
            "ece_error_risk": np.nan,
        }

    bin_ids = np.minimum((p * int(bins)).astype(int), int(bins) - 1)
    ece = 0.0
    for bin_id in range(int(bins)):
        in_bin = bin_ids == bin_id
        if in_bin.any():
            ece += (
                float(in_bin.mean())
                * abs(float(y[in_bin].mean()) - float(p[in_bin].mean()))
            )
    return {
        "calibration_n": int(len(y)),
        "error_count": float(y.sum()),
        "error_rate": float(y.mean()),
        "mean_predicted_error_risk": float(p.mean()),
        "brier_error_risk": float(np.mean((p - y) ** 2)),
        "ece_error_risk": float(ece),
    }
