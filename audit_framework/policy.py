"""Decision policy, metrics, and conservative model selection for PAUSE."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .components import calibrated_error_risk


def _numeric(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table:
        return pd.Series(np.nan, index=table.index, dtype=float)
    return pd.to_numeric(table[column], errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )


def apply_uncertainty_policy(
    table: pd.DataFrame,
    calibrated_probability: pd.Series,
    *,
    confidence_threshold: float,
    reject_fraction: float,
    confidence_source: str = "base",
) -> dict[str, pd.Series | int | float]:
    """Apply frozen-positive restriction and uncertainty-first deferment."""

    probability = calibrated_probability.astype(float).clip(0.0, 1.0)
    error_risk = calibrated_error_risk(table, probability)
    calibrated_confidence = 1.0 - error_risk
    if confidence_source == "base":
        confidence = _numeric(table, "base_confidence")
        if confidence.isna().all():
            base_probability = _numeric(table, "p_base")
            confidence = (2.0 * (base_probability - 0.5).abs()).clip(0.0, 1.0)
    elif confidence_source == "calibrated":
        confidence = calibrated_confidence
    else:
        raise ValueError(f"Unsupported confidence source: {confidence_source}")
    base_pred = _numeric(table, "base_pred")
    if base_pred.isna().all():
        base_pred = (_numeric(table, "p_base") >= 0.5).astype(float)
    candidate = (
        probability.notna()
        & base_pred.ge(0.5)
        & confidence.gt(float(confidence_threshold))
    )

    candidate_positions = np.flatnonzero(candidate.to_numpy(dtype=bool))
    reject_n = int(round(len(candidate_positions) * float(reject_fraction)))
    reject_n = min(max(reject_n, 0), len(candidate_positions))
    deferred = pd.Series(False, index=table.index)
    if reject_n:
        candidate_risk = error_risk.iloc[candidate_positions]
        order = candidate_risk.sort_values(
            ascending=False,
            kind="mergesort",
        )
        deferred.loc[order.index[:reject_n]] = True
    residual = candidate & ~deferred
    return {
        "candidate_mask": candidate,
        "deferred_mask": deferred,
        "residual_mask": residual,
        "candidate_confidence": confidence,
        "calibrated_confidence": calibrated_confidence,
        "calibrated_error_risk": error_risk,
        "calibrated_uncertainty": error_risk,
        "candidate_n": int(candidate.sum()),
        "deferred_n": int(deferred.sum()),
        "residual_n": int(residual.sum()),
    }


def _expected_top_k(
    labels: np.ndarray,
    scores: np.ndarray,
    review_n: int,
) -> float:
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    boundary_score = sorted_scores[review_n - 1]
    above = sorted_scores > boundary_score
    boundary = sorted_scores == boundary_score
    above_n = int(above.sum())
    boundary_slots = int(review_n - above_n)
    boundary_rate = float(sorted_labels[boundary].mean())
    return float(sorted_labels[above].sum() + boundary_slots * boundary_rate)


def evaluate_ranking(
    table: pd.DataFrame,
    residual_mask: pd.Series,
    scores: pd.Series | np.ndarray,
    *,
    review_fraction: float,
    min_universe_n: int,
    deferred_mask: pd.Series | None = None,
) -> dict[str, float | int]:
    labels_all = _numeric(table, "base_wrong")
    score_series = pd.Series(np.asarray(scores, dtype=float), index=table.index)
    valid = residual_mask & labels_all.notna() & score_series.notna()
    labels = labels_all.loc[valid].to_numpy(dtype=float)
    score_values = score_series.loc[valid].to_numpy(dtype=float)
    universe_n = int(len(labels))
    error_count = float(labels.sum()) if universe_n else 0.0
    prevalence = float(labels.mean()) if universe_n else np.nan

    if universe_n and np.unique(labels).size == 2:
        auroc = float(roc_auc_score(labels, score_values))
        auprc = float(average_precision_score(labels, score_values))
    else:
        auroc = auprc = np.nan

    if universe_n >= int(min_universe_n):
        review_n = int(round(universe_n * float(review_fraction)))
        review_n = (
            min(max(review_n, 1), universe_n - 1)
            if universe_n > 1
            else universe_n
        )
        review_errors = _expected_top_k(labels, score_values, review_n)
        review_error_rate = review_errors / max(review_n, 1)
        review_lift = review_error_rate / max(prevalence, 1.0e-8)
        residual_recall = review_errors / max(error_count, 1.0e-8)
        retained_n = universe_n - review_n
        retained_error_count = error_count - review_errors
        retained_accuracy = (
            1.0 - retained_error_count / retained_n
            if retained_n
            else np.nan
        )
        retained_gain = retained_accuracy - (1.0 - prevalence)
    else:
        review_n = 0
        review_errors = review_error_rate = review_lift = np.nan
        residual_recall = retained_accuracy = retained_gain = np.nan

    deferred_error_count = 0.0
    all_candidate_error_count = error_count
    if deferred_mask is not None:
        deferred_labels = labels_all.loc[deferred_mask & labels_all.notna()]
        deferred_error_count = float(deferred_labels.sum())
        all_candidate_error_count += deferred_error_count
    combined_recall = (
        (deferred_error_count + review_errors)
        / max(all_candidate_error_count, 1.0e-8)
        if np.isfinite(review_errors)
        else np.nan
    )
    return {
        "universe_n": universe_n,
        "universe_error_count": error_count,
        "universe_error_rate": prevalence,
        "error_detection_auroc": auroc,
        "error_detection_auprc": auprc,
        "reviewed_n": int(review_n),
        "review_error_count": review_errors,
        "review_error_rate": review_error_rate,
        "review_lift": review_lift,
        "review_recall_of_residual_errors": residual_recall,
        "combined_recall_of_candidate_errors": combined_recall,
        "retained_accuracy_after_review": retained_accuracy,
        "retained_accuracy_gain_vs_residual": retained_gain,
    }


def select_one_standard_error(
    candidates: pd.DataFrame,
    *,
    fallback_profile: str = "uncertainty",
    min_validation_gain: float = 0.0,
) -> dict[str, object]:
    """Select a simple profile within one SE of the best grouped-CV result."""

    if candidates.empty:
        return {
            "selected_profile": fallback_profile,
            "selection_fallback": True,
            "selection_reason": "empty_candidate_table",
        }
    work = candidates.loc[candidates["primary_candidate"].astype(bool)].copy()
    work["validation_fold_mean_auprc"] = pd.to_numeric(
        work["validation_fold_mean_auprc"],
        errors="coerce",
    )
    finite = work.loc[work["validation_fold_mean_auprc"].notna()].copy()
    if "validation_finite_folds" in finite:
        fold_counts = pd.to_numeric(
            finite["validation_finite_folds"],
            errors="coerce",
        )
        finite = finite.loc[fold_counts.ge(2)].copy()
    fallback = work.loc[work["profile"].astype(str) == fallback_profile]

    if finite.empty:
        selected = fallback.iloc[0] if not fallback.empty else work.iloc[0]
        reason = "no_finite_group_cv_candidate"
        return {
            **selected.to_dict(),
            "selected_profile": selected["profile"],
            "selection_fallback": True,
            "selection_reason": reason,
        }

    best = finite.sort_values(
        ["validation_fold_mean_auprc", "complexity", "profile"],
        ascending=[False, True, True],
        kind="mergesort",
    ).iloc[0]
    best_se = pd.to_numeric(
        pd.Series([best.get("validation_fold_se_auprc", np.nan)]),
        errors="coerce",
    ).iloc[0]
    if not np.isfinite(best_se):
        best_se = 0.0
    threshold = float(best["validation_fold_mean_auprc"]) - float(best_se)
    eligible = finite.loc[
        finite["validation_fold_mean_auprc"] >= threshold
    ].sort_values(
        ["complexity", "validation_fold_mean_auprc", "profile"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    selected = eligible.iloc[0]

    fallback_metric = np.nan
    if not fallback.empty:
        fallback_metric = pd.to_numeric(
            pd.Series([fallback.iloc[0]["validation_fold_mean_auprc"]]),
            errors="coerce",
        ).iloc[0]
    selected_metric = float(selected["validation_fold_mean_auprc"])
    gain = selected_metric - fallback_metric if np.isfinite(fallback_metric) else np.nan
    if (
        selected["profile"] != fallback_profile
        and np.isfinite(gain)
        and gain < float(min_validation_gain)
        and not fallback.empty
    ):
        selected = fallback.iloc[0]
        return {
            **selected.to_dict(),
            "selected_profile": fallback_profile,
            "selection_fallback": True,
            "selection_reason": "gain_below_minimum",
            "best_profile": best["profile"],
            "best_validation_fold_mean_auprc": best[
                "validation_fold_mean_auprc"
            ],
            "one_se_threshold": threshold,
            "validation_gain_vs_uncertainty": gain,
        }

    return {
        **selected.to_dict(),
        "selected_profile": selected["profile"],
        "selection_fallback": selected["profile"] == fallback_profile,
        "selection_reason": (
            "uncertainty_selected"
            if selected["profile"] == fallback_profile
            else "one_standard_error"
        ),
        "best_profile": best["profile"],
        "best_validation_fold_mean_auprc": best[
            "validation_fold_mean_auprc"
        ],
        "one_se_threshold": threshold,
        "validation_gain_vs_uncertainty": gain,
    }
