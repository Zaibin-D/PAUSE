"""Run PAUSE-v2 Phase 1 diagnostics: U2 plus E2-clean.

This script is intentionally independent from the frozen formal PAUSE audit.
It reads cached validation/test ``pause_audit_inputs.csv`` files, reuses the
same high-confidence positive plus uncertainty-defer policy, builds label-free
U2 and E2-clean features for residual candidates, and writes debug CSV outputs
under ``audit_framework/cache/diagnostic_results`` by default.

No P2, E2-label, source-label local positive rate, test-label feature
construction, or formal-result writes are implemented here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from audit_framework import (
    ProfileResult,
    apply_uncertainty_policy,
    calibrated_error_risk,
    cross_fitted_calibration,
    evaluate_ranking,
    fit_calibrator,
)
from audit_framework.data import (
    DEFAULT_DATASETS,
    DEFAULT_TEST_ROOTS,
    prepare_table,
    write_csv,
)
from audit_framework.modeling import (
    _fit_hybrid_ranker,
    _numeric,
    _predict_ranker,
    _usable_features,
    make_folds,
)
from audit_framework.scripts.run_audit import (
    METRICS,
    POLICY_KEYS,
    _coverage,
    _discover_pairs,
    _policy_metadata,
)
from audit_framework.sharding import MODEL_ROOTS


U2_FEATURES = (
    "u2_calibrated_error_risk",
    "u2_confidence_percentile",
    "u2_risk_percentile",
    "u2_defer_boundary_margin",
    "u2_defer_boundary_proximity",
)

E2_FEATURES = (
    "e2_drug_marginal_support",
    "e2_target_marginal_support",
    "e2_joint_nn_sim",
    "e2_joint_top5_mean_sim",
    "e2_joint_top10_density",
    "e2_joint_support_gap",
    "e2_drug_target_support_imbalance",
)

FORMAL_PROFILE_MAP = {
    "uncertainty_support": "U+E",
    "uncertainty_prior_support": "U+P+E",
}


def _normalise_seed(value: str) -> str:
    text = str(value)
    return text if text.startswith("seed_") else f"seed_{text}"


def _canonical_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    if np.isfinite(number) and number.is_integer():
        return str(int(number))
    return text


def _hmean(left: pd.Series | float, right: pd.Series | float, eps: float = 1.0e-8):
    return (2.0 * left * right) / (left + right + eps)


def _rank_percentile(values: pd.Series, mask: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=values.index, dtype=float)
    scoped = pd.to_numeric(values.loc[mask], errors="coerce")
    valid = scoped.notna()
    n = int(valid.sum())
    if n <= 0:
        return out
    ranks = scoped.loc[valid].rank(method="average", ascending=True)
    out.loc[ranks.index] = ranks / float(n)
    return out


def _support_like(values: pd.Series, *, high_is_support: bool) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = pd.Series(np.nan, index=values.index, dtype=float)
    valid = numeric.notna()
    n = int(valid.sum())
    if n <= 0:
        return out
    if numeric.loc[valid].nunique() <= 1:
        out.loc[valid] = 0.5
        return out
    ranks = numeric.loc[valid].rank(method="average", ascending=True) / float(n)
    out.loc[valid] = ranks if high_is_support else 1.0 - ranks
    return out.clip(0.0, 1.0)


def _support_columns(table: pd.DataFrame, axis: str) -> tuple[list[str], list[str]]:
    """Return support-positive and distance/novelty columns for one entity axis."""

    axis_tokens = {
        "drug": ("drug", "dr_"),
        "target": ("target", "pr_"),
    }[axis]
    positive_terms = ("support", "count", "density")
    risk_terms = ("distance", "novel", "unseen", "sparsity", "shift")
    positive: list[str] = []
    risk: list[str] = []
    for column in table.columns:
        lower = column.lower()
        if not any(token in lower for token in axis_tokens):
            continue
        if any(term in lower for term in positive_terms):
            positive.append(column)
        if any(term in lower for term in risk_terms):
            risk.append(column)
    return sorted(set(positive)), sorted(set(risk))


def _marginal_support(table: pd.DataFrame, axis: str) -> tuple[pd.Series, tuple[str, ...]]:
    positive, risk = _support_columns(table, axis)
    pieces = []
    used: list[str] = []
    for column in positive:
        pieces.append(_support_like(table[column], high_is_support=True))
        used.append(column)
    for column in risk:
        pieces.append(_support_like(table[column], high_is_support=False))
        used.append(column)
    if not pieces:
        return pd.Series(np.nan, index=table.index, dtype=float), ()
    return pd.concat(pieces, axis=1).mean(axis=1, skipna=True).clip(0.0, 1.0), tuple(used)


def _load_joint_payload(dataset_root: str | Path, dataset: str) -> dict[str, object] | None:
    path = Path(dataset_root) / dataset / "cluster" / "pime" / "joint_source_support.pkl"
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except (OSError, pickle.UnpicklingError, EOFError):
        return None
    required = {"drug_neighbours", "target_neighbours", "source_pairs"}
    return payload if isinstance(payload, dict) and required.issubset(payload) else None


def _joint_from_payload(table: pd.DataFrame, payload: dict[str, object]) -> pd.DataFrame:
    drug_neighbours = payload.get("drug_neighbours", {})
    target_neighbours = payload.get("target_neighbours", {})
    source_pairs = set(payload.get("source_pairs", set()))
    exclude_exact = bool(payload.get("exact_query_pair_excluded", True))
    rows: list[dict[str, float]] = []

    for _, row in table.iterrows():
        drug_id = _canonical_id(row.get("dr_id"))
        target_id = _canonical_id(row.get("pr_id"))
        d_neigh = list(drug_neighbours.get(drug_id, ())) if isinstance(drug_neighbours, dict) else []
        t_neigh = list(target_neighbours.get(target_id, ())) if isinstance(target_neighbours, dict) else []
        combos: list[tuple[float, bool]] = []
        for d_id, d_sim in d_neigh:
            for t_id, t_sim in t_neigh:
                cd = _canonical_id(d_id)
                ct = _canonical_id(t_id)
                if exclude_exact and cd == drug_id and ct == target_id:
                    continue
                score = float(_hmean(float(d_sim), float(t_sim)))
                combos.append((score, (cd, ct) in source_pairs))
        combos.sort(key=lambda item: item[0], reverse=True)
        supported = [score for score, exists in combos if exists]
        top10 = combos[:10]
        rows.append(
            {
                "e2_joint_nn_sim": max(supported, default=0.0),
                "e2_joint_top5_mean_sim": float(np.mean((supported + [0.0] * 5)[:5])),
                "e2_joint_top10_density": (
                    float(sum(1 for _, exists in top10 if exists) / len(top10))
                    if top10
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows, index=table.index)


def _e2_features(
    table: pd.DataFrame,
    *,
    dataset: str,
    dataset_root: str | Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    drug_support, drug_columns = _marginal_support(table, "drug")
    target_support, target_columns = _marginal_support(table, "target")
    fallback_joint = pd.Series(
        _hmean(drug_support.fillna(0.0), target_support.fillna(0.0)),
        index=table.index,
        dtype=float,
    ).clip(0.0, 1.0)

    payload = _load_joint_payload(dataset_root, dataset)
    real_joint_available = payload is not None and {"dr_id", "pr_id"}.issubset(table.columns)
    if real_joint_available:
        joint = _joint_from_payload(table, payload)
        joint_mode = "real_source_pair_joint_support"
        top5_fallback = False
        top10_fallback = False
    else:
        joint = pd.DataFrame(index=table.index)
        joint["e2_joint_nn_sim"] = fallback_joint
        joint["e2_joint_top5_mean_sim"] = fallback_joint
        joint["e2_joint_top10_density"] = fallback_joint
        joint_mode = "harmonic_mean_marginal_fallback"
        top5_fallback = True
        top10_fallback = True

    marginal_hmean = fallback_joint
    features = pd.DataFrame(index=table.index)
    features["e2_drug_marginal_support"] = drug_support
    features["e2_target_marginal_support"] = target_support
    features["e2_joint_nn_sim"] = joint["e2_joint_nn_sim"].clip(0.0, 1.0)
    features["e2_joint_top5_mean_sim"] = joint["e2_joint_top5_mean_sim"].clip(0.0, 1.0)
    features["e2_joint_top10_density"] = joint["e2_joint_top10_density"].clip(0.0, 1.0)
    features["e2_joint_support_gap"] = (marginal_hmean - features["e2_joint_nn_sim"]).clip(-1.0, 1.0)
    features["e2_drug_target_support_imbalance"] = (drug_support - target_support).abs().clip(0.0, 1.0)

    missing: list[str] = []
    if not drug_columns:
        missing.append("drug_support_columns")
    if not target_columns:
        missing.append("target_support_columns")
    if payload is None:
        missing.append("joint_source_support_asset")

    mode = {
        "e2_real_joint_support_available": bool(real_joint_available),
        "e2_fallback_used": not bool(real_joint_available),
        "e2_top5_fallback_used": top5_fallback,
        "e2_top10_fallback_used": top10_fallback,
        "e2_joint_support_mode": joint_mode,
        "e2_drug_support_columns": "|".join(drug_columns),
        "e2_target_support_columns": "|".join(target_columns),
        "e2_missing_columns_or_assets": "|".join(missing),
    }
    return features.loc[:, E2_FEATURES], mode


def _boundary_features(
    risk: pd.Series,
    candidate_mask: pd.Series,
    deferred_mask: pd.Series,
) -> tuple[pd.Series, pd.Series, str]:
    margin = pd.Series(np.nan, index=risk.index, dtype=float)
    proximity = pd.Series(np.nan, index=risk.index, dtype=float)
    warning = ""

    candidate_risk = risk.loc[candidate_mask].dropna()
    deferred_risk = risk.loc[deferred_mask].dropna()
    if candidate_risk.empty:
        return margin, proximity, "no_candidates"
    if deferred_risk.empty:
        cutoff = float(candidate_risk.max())
        warning = "no_deferred_rows_cutoff_set_to_candidate_max_risk"
    else:
        # apply_uncertainty_policy defers the highest-risk candidate rows.
        cutoff = float(deferred_risk.min())

    margin.loc[candidate_mask] = cutoff - risk.loc[candidate_mask]
    scoped = margin.loc[candidate_mask].dropna()
    n = int(len(scoped))
    if n == 1:
        proximity.loc[scoped.index] = 1.0
    elif n > 1:
        ranks = scoped.rank(method="average", ascending=True)
        proximity.loc[ranks.index] = 1.0 - ((ranks - 1.0) / float(n - 1))
    return margin, proximity.clip(0.0, 1.0), warning


def _u2_features(table: pd.DataFrame, policy: dict[str, object]) -> tuple[pd.DataFrame, list[str]]:
    """Build the requested five U2 features without using residual labels."""

    required = ("base_confidence",)
    missing = [column for column in required if column not in table.columns]
    candidate = pd.Series(policy["candidate_mask"], index=table.index).astype(bool)
    deferred = pd.Series(policy["deferred_mask"], index=table.index).astype(bool)
    risk = pd.to_numeric(policy["calibrated_error_risk"], errors="coerce")
    confidence = pd.to_numeric(policy["candidate_confidence"], errors="coerce")

    margin, proximity, boundary_warning = _boundary_features(risk, candidate, deferred)
    if boundary_warning:
        missing.append(boundary_warning)

    features = pd.DataFrame(index=table.index)
    features["u2_calibrated_error_risk"] = risk
    features["u2_confidence_percentile"] = _rank_percentile(confidence, candidate)
    features["u2_risk_percentile"] = _rank_percentile(risk, candidate)
    features["u2_defer_boundary_margin"] = margin
    features["u2_defer_boundary_proximity"] = proximity
    return features.loc[:, U2_FEATURES], missing


def _uncertainty_result(
    table: pd.DataFrame,
    residual_mask: pd.Series,
    calibrated_probability: pd.Series,
    *,
    folds: int,
    seed: int,
    group_axis: str,
    strict_group: bool,
) -> ProfileResult:
    scores = pd.Series(np.nan, index=table.index, dtype=float)
    scores.loc[residual_mask] = calibrated_error_risk(
        table.loc[residual_mask],
        calibrated_probability.loc[residual_mask],
    )
    fold_id = pd.Series(-1, index=table.index, dtype=int)

    labels_all = _numeric(table, "base_wrong").to_numpy(dtype=float)
    positions = np.flatnonzero(residual_mask.to_numpy(dtype=bool))
    positions = positions[np.isfinite(labels_all[positions])]
    if len(positions) < 2:
        return ProfileResult(scores, fold_id, (), "insufficient_residual_n", ("calibrated_error_risk",))
    labels = labels_all[positions].astype(int)
    if np.unique(labels).size < 2:
        return ProfileResult(scores, fold_id, (), "single_residual_class", ("calibrated_error_risk",))

    split_list = make_folds(
        table.iloc[positions],
        labels,
        folds=folds,
        seed=seed,
        group_axis=group_axis,
        strict_group=strict_group,
    )
    fold_metrics: list[float] = []
    for fold, (_, valid_pos) in enumerate(split_list):
        valid_positions = positions[valid_pos]
        fold_id.iloc[valid_positions] = fold
        fold_labels = labels_all[valid_positions].astype(int)
        fold_scores = scores.iloc[valid_positions].to_numpy(dtype=float)
        fold_metrics.append(
            float(average_precision_score(fold_labels, fold_scores))
            if np.unique(fold_labels).size == 2
            else np.nan
        )
    status = "complete" if split_list else "group_cv_unavailable"
    return ProfileResult(scores, fold_id, tuple(fold_metrics), status, ("calibrated_error_risk",))


def _u2_cross_fitted_result(
    table: pd.DataFrame,
    residual_mask: pd.Series,
    features: pd.DataFrame,
    *,
    requested_features: tuple[str, ...] = U2_FEATURES,
    profile_name: str = "U2",
    folds: int,
    seed: int,
    group_axis: str,
    strict_group: bool,
    min_fit_n: int,
    min_feature_n: int,
    ranking_weight: float,
    max_pairs: int,
) -> ProfileResult:
    scores = pd.Series(np.nan, index=table.index, dtype=float)
    fold_id = pd.Series(-1, index=table.index, dtype=int)
    labels_all = _numeric(table, "base_wrong").to_numpy(dtype=float)
    positions = np.flatnonzero(residual_mask.to_numpy(dtype=bool))
    positions = positions[np.isfinite(labels_all[positions])]
    if len(positions) < int(min_fit_n):
        return ProfileResult(scores, fold_id, (), "insufficient_residual_n", ())
    labels = labels_all[positions].astype(int)
    if np.unique(labels).size < 2:
        return ProfileResult(scores, fold_id, (), "single_residual_class", ())

    selected = _usable_features(
        features.iloc[positions],
        requested_features,
        min_feature_n=min_feature_n,
    )
    if not selected:
        return ProfileResult(scores, fold_id, (), f"no_usable_{profile_name.lower()}_features", ())

    split_list = make_folds(
        table.iloc[positions],
        labels,
        folds=folds,
        seed=seed,
        group_axis=group_axis,
        strict_group=strict_group,
    )
    if not split_list:
        return ProfileResult(scores, fold_id, (), "group_cv_unavailable", selected)

    fold_metrics: list[float] = []
    for fold, (train_pos, valid_pos) in enumerate(split_list):
        train_positions = positions[train_pos]
        valid_positions = positions[valid_pos]
        train_x = features.iloc[train_positions].loc[:, selected]
        valid_x = features.iloc[valid_positions].loc[:, selected]
        fitted = _fit_hybrid_ranker(
            train_x,
            labels_all[train_positions].astype(int),
            ranking_weight=ranking_weight,
            max_pairs=max_pairs,
            seed=int(seed) + fold,
        )
        predictions = _predict_ranker(fitted, valid_x)
        scores.iloc[valid_positions] = predictions
        fold_id.iloc[valid_positions] = fold
        fold_labels = labels_all[valid_positions].astype(int)
        fold_metrics.append(
            float(average_precision_score(fold_labels, predictions))
            if np.unique(fold_labels).size == 2
            else np.nan
        )

    finite_n = int(scores.iloc[positions].notna().sum())
    status = "complete" if finite_n == len(positions) else "partial" if finite_n else "unavailable"
    return ProfileResult(scores, fold_id, tuple(fold_metrics), status, selected)


def _u2_test_result(
    validation: pd.DataFrame,
    validation_residual_mask: pd.Series,
    validation_features: pd.DataFrame,
    test: pd.DataFrame,
    test_residual_mask: pd.Series,
    test_features: pd.DataFrame,
    *,
    requested_features: tuple[str, ...] = U2_FEATURES,
    profile_name: str = "U2",
    seed: int,
    min_fit_n: int,
    min_feature_n: int,
    ranking_weight: float,
    max_pairs: int,
) -> ProfileResult:
    scores = pd.Series(np.nan, index=test.index, dtype=float)
    fold_id = pd.Series(-1, index=test.index, dtype=int)
    labels_all = _numeric(validation, "base_wrong").to_numpy(dtype=float)
    train_positions = np.flatnonzero(validation_residual_mask.to_numpy(dtype=bool))
    train_positions = train_positions[np.isfinite(labels_all[train_positions])]
    test_positions = np.flatnonzero(test_residual_mask.to_numpy(dtype=bool))

    if len(train_positions) < int(min_fit_n):
        return ProfileResult(scores, fold_id, (), "insufficient_residual_n", ())
    labels = labels_all[train_positions].astype(int)
    if np.unique(labels).size < 2:
        return ProfileResult(scores, fold_id, (), "single_residual_class", ())
    if len(test_positions) == 0:
        return ProfileResult(scores, fold_id, (), "empty_test_residual", ())

    selected = _usable_features(
        validation_features.iloc[train_positions],
        requested_features,
        min_feature_n=min_feature_n,
    )
    if not selected:
        return ProfileResult(scores, fold_id, (), f"no_usable_{profile_name.lower()}_features", ())

    fitted = _fit_hybrid_ranker(
        validation_features.iloc[train_positions].loc[:, selected],
        labels,
        ranking_weight=ranking_weight,
        max_pairs=max_pairs,
        seed=seed,
    )
    scores.iloc[test_positions] = _predict_ranker(
        fitted,
        test_features.iloc[test_positions].loc[:, selected],
    )
    return ProfileResult(scores, fold_id, (), "complete", selected)


def _candidate_row(
    metadata: dict[str, object],
    *,
    split: str,
    profile: str,
    complexity: int,
    result: ProfileResult,
    metrics: dict[str, object],
    policy: dict[str, object],
) -> dict[str, object]:
    return {
        **metadata,
        "split": split,
        "profile": profile,
        "primary_candidate": True,
        "complexity": complexity,
        "model_status": result.status,
        "used_features": "|".join(result.used_features),
        "validation_fold_mean_auprc": result.mean_fold_auprc,
        "validation_fold_se_auprc": result.se_fold_auprc,
        "validation_finite_folds": int(np.isfinite(np.asarray(result.fold_auprcs, dtype=float)).sum()),
        "candidate_n": int(policy["candidate_mask"].sum()),
        "deferred_n": int(policy["deferred_mask"].sum()),
        "residual_n": int(policy["residual_mask"].sum()),
        **metrics,
    }


def _feature_availability_row(
    metadata: dict[str, object],
    *,
    split: str,
    features: pd.DataFrame,
    residual_mask: pd.Series,
    missing_inputs: list[str],
    e2_mode: dict[str, object],
) -> dict[str, object]:
    residual_u2 = features.loc[residual_mask, list(U2_FEATURES)]
    residual_e2 = features.loc[residual_mask, list(E2_FEATURES)]
    residual_all = features.loc[residual_mask, list(U2_FEATURES + E2_FEATURES)]
    u2_null_counts = residual_u2.isna().sum()
    e2_null_counts = residual_e2.isna().sum()
    usable_u2 = _usable_features(residual_u2, U2_FEATURES, min_feature_n=2)
    usable_e2 = _usable_features(residual_e2, E2_FEATURES, min_feature_n=2)
    return {
        **metadata,
        "split": split,
        "u2_features_available": bool(len(usable_u2) == len(U2_FEATURES)),
        "e2_features_available": bool(len(usable_e2) == len(E2_FEATURES)),
        "usable_u2_features": "|".join(usable_u2),
        "usable_e2_features": "|".join(usable_e2),
        "e2_real_joint_support_available": e2_mode.get("e2_real_joint_support_available", False),
        "e2_fallback_used": e2_mode.get("e2_fallback_used", True),
        "e2_top5_fallback_used": e2_mode.get("e2_top5_fallback_used", True),
        "e2_top10_fallback_used": e2_mode.get("e2_top10_fallback_used", True),
        "e2_joint_support_mode": e2_mode.get("e2_joint_support_mode", "unavailable"),
        "e2_drug_support_columns": e2_mode.get("e2_drug_support_columns", ""),
        "e2_target_support_columns": e2_mode.get("e2_target_support_columns", ""),
        "missing_columns_or_warnings": "|".join(
            sorted(
                set(missing_inputs)
                | set(str(e2_mode.get("e2_missing_columns_or_assets", "")).split("|"))
                - {""}
            )
        ),
        "fallback_used": "none" if not e2_mode.get("e2_fallback_used", True) else "e2_harmonic_mean",
        "residual_rows_with_any_u2_null": int(residual_u2.isna().any(axis=1).sum()),
        "residual_rows_with_any_e2_null": int(residual_e2.isna().any(axis=1).sum()),
        "residual_rows_with_any_feature_null": int(residual_all.isna().any(axis=1).sum()),
        "residual_rows_with_all_features_present": int(residual_all.notna().all(axis=1).sum()),
        **{f"{column}_null_n": int(u2_null_counts[column]) for column in U2_FEATURES},
        **{f"{column}_null_n": int(e2_null_counts[column]) for column in E2_FEATURES},
    }


def _selected_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    groups: list[tuple[str, list[str]]] = [
        ("overall", []),
        ("by_model", ["model"]),
        ("by_dataset", ["dataset"]),
        ("by_group_axis", ["group_axis"]),
        ("by_model_dataset", ["model", "dataset"]),
    ]
    for scope, keys in groups:
        grouped = [((), frame)] if not keys else frame.groupby(keys, dropna=False)
        for key_values, part in grouped:
            if keys and not isinstance(key_values, tuple):
                key_values = (key_values,)
            row: dict[str, object] = {
                "scope": scope,
                "profile": "validation_selected_v2",
                "n_runs": int(len(part)),
            }
            for key, value in zip(keys, key_values if keys else ()):
                row[key] = value
            for metric in METRICS:
                if metric in part:
                    values = pd.to_numeric(part[metric], errors="coerce")
                    row[f"{metric}_mean"] = float(values.mean()) if values.notna().any() else np.nan
                    row[f"{metric}_sd"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "profile": "U",
                "profile_type": "formal_baseline",
                "features": "calibrated_error_risk",
                "notes": "Uncertainty baseline under the same candidate/defer/residual policy.",
            },
            {
                "profile": "U+E",
                "profile_type": "formal_baseline",
                "features": "formal uncertainty_support",
                "notes": "Read-only formal PAUSE comparator if present.",
            },
            {
                "profile": "U+P+E",
                "profile_type": "formal_baseline",
                "features": "formal uncertainty_prior_support",
                "notes": "Read-only formal PAUSE comparator if present.",
            },
            {
                "profile": "formal_selected",
                "profile_type": "formal_baseline",
                "features": "validation-selected formal PAUSE profile",
                "notes": "Read-only formal selected comparator if present.",
            },
            {
                "profile": "U2",
                "profile_type": "v2_candidate",
                "features": "|".join(U2_FEATURES),
                "notes": "Phase 1 U2 features only.",
            },
            {
                "profile": "U2+E2",
                "profile_type": "v2_candidate",
                "features": "|".join(U2_FEATURES + E2_FEATURES),
                "notes": "Phase 1 U2 plus E2-clean pair-level support proxy. No P2 or E2-label.",
            },
        ]
    )


def _filter_by_keys(frame: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or keys.empty:
        return pd.DataFrame(columns=frame.columns)
    shared = [column for column in POLICY_KEYS if column in frame.columns and column in keys.columns]
    if not shared:
        return pd.DataFrame(columns=frame.columns)
    return frame.merge(keys.loc[:, shared].drop_duplicates(), on=shared, how="inner")


def _formal_baseline_rows(
    *,
    split: str,
    policy_keys: pd.DataFrame,
    formal_dir: Path,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    candidate_name = "candidate_validation_runs.csv" if split == "validation" else "candidate_test_runs.csv"
    candidate_path = formal_dir / candidate_name
    if candidate_path.exists():
        candidates = _filter_by_keys(pd.read_csv(candidate_path), policy_keys)
        for source, label in FORMAL_PROFILE_MAP.items():
            part = candidates.loc[candidates["profile"].astype(str).eq(source)].copy()
            if not part.empty:
                part["profile"] = label
                rows.append(part)

    selected_name = "selection_choices.csv" if split == "validation" else "selected_test_runs.csv"
    selected_path = formal_dir / selected_name
    if selected_path.exists():
        selected = _filter_by_keys(pd.read_csv(selected_path), policy_keys)
        if not selected.empty:
            selected = selected.copy()
            selected["profile"] = "formal_selected"
            rows.append(selected)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def _metric_delta_summary(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    comparison: str,
    status_if_empty: str = "unavailable",
) -> pd.DataFrame:
    if left.empty or right.empty:
        return pd.DataFrame(
            [
                {
                    "comparison": comparison,
                    "status": status_if_empty,
                    "metric": "",
                    "n_runs": 0,
                    "left_mean": np.nan,
                    "right_mean": np.nan,
                    "delta_mean": np.nan,
                }
            ]
        )
    shared = [column for column in POLICY_KEYS if column in left.columns and column in right.columns]
    merged = left.merge(right, on=shared, suffixes=("_left", "_right"))
    if merged.empty:
        return pd.DataFrame(
            [
                {
                    "comparison": comparison,
                    "status": "no_matching_policy_keys",
                    "metric": "",
                    "n_runs": 0,
                    "left_mean": np.nan,
                    "right_mean": np.nan,
                    "delta_mean": np.nan,
                }
            ]
        )
    rows = []
    for metric in METRICS:
        lcol = f"{metric}_left"
        rcol = f"{metric}_right"
        if lcol not in merged or rcol not in merged:
            continue
        left_values = pd.to_numeric(merged[lcol], errors="coerce")
        right_values = pd.to_numeric(merged[rcol], errors="coerce")
        delta = left_values - right_values
        rows.append(
            {
                "comparison": comparison,
                "status": "ok",
                "metric": metric,
                "n_runs": int(delta.notna().sum()),
                "left_mean": float(left_values.mean()) if left_values.notna().any() else np.nan,
                "right_mean": float(right_values.mean()) if right_values.notna().any() else np.nan,
                "delta_mean": float(delta.mean()) if delta.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _component_increment_summary(test_candidates: pd.DataFrame) -> pd.DataFrame:
    left = test_candidates.loc[test_candidates["profile"].astype(str).eq("U2+E2")]
    right = test_candidates.loc[test_candidates["profile"].astype(str).eq("U2")]
    summary = _metric_delta_summary(left, right, comparison="U2+E2 - U2")
    summary.insert(0, "scope", "overall")
    return summary


def _v2_vs_formal_summary(selected_test: pd.DataFrame, formal_test: pd.DataFrame) -> pd.DataFrame:
    if formal_test.empty or "profile" not in formal_test.columns:
        return _metric_delta_summary(
            selected_test,
            pd.DataFrame(),
            comparison="v2_selected - formal_selected",
        )
    formal_selected = formal_test.loc[formal_test["profile"].astype(str).eq("formal_selected")]
    return _metric_delta_summary(
        selected_test,
        formal_selected,
        comparison="v2_selected - formal_selected",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["PACE", "TAPB", "DrugBAN"])
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--seeds", nargs="+", default=["1", "2", "3", "4", "5"])
    parser.add_argument("--group-axes", nargs="+", default=["target", "drug"])
    parser.add_argument("--test-roots", nargs="+", default=list(DEFAULT_TEST_ROOTS))
    parser.add_argument("--calibration-root", default="audit_framework/cache/validation_audits")
    parser.add_argument("--dataset-root", default="datasets")
    parser.add_argument(
        "--out-dir",
        default="audit_framework/cache/diagnostic_results/pause_v2_phase1_debug",
    )
    parser.add_argument("--confidence-source", default="base", choices=["base", "calibrated"])
    parser.add_argument("--confidence-thresholds", nargs="+", type=float, default=[0.8])
    parser.add_argument("--uncertainty-reject-fractions", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--review-fraction", type=float, default=0.20)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-seed", type=int, default=13)
    parser.add_argument("--strict-group-cv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-fit-n", type=int, default=30)
    parser.add_argument("--min-feature-n", type=int, default=20)
    parser.add_argument("--ranking-weight", type=float, default=0.5)
    parser.add_argument("--max-ranking-pairs", type=int, default=20000)
    parser.add_argument("--min-universe-n", type=int, default=10)
    parser.add_argument("--min-validation-gain", type=float, default=0.005)
    parser.add_argument("--precision", type=int, default=6)
    return parser


def _roots_from_models(models: Iterable[str]) -> list[str]:
    roots: list[str] = []
    for model in models:
        key = str(model).lower()
        if key not in MODEL_ROOTS:
            raise ValueError(f"Unknown model shard: {model}")
        roots.append(MODEL_ROOTS[key][1])
    return roots


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args.datasets = [str(dataset).lower() for dataset in args.datasets]
    args.seeds = [_normalise_seed(seed) for seed in args.seeds]
    args.test_roots = _roots_from_models(args.models)

    pairs, missing = _discover_pairs(args)
    write_csv(missing, out_dir / "missing_inputs_v2_phase1.csv", args.precision)

    coverage_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []

    for pair in pairs:
        validation = prepare_table(
            Path(pair["validation_path"]),
            str(pair["model"]),
            str(pair["dataset"]),
            str(pair["test_run"]),
            dataset_root=args.dataset_root,
        )
        test = prepare_table(
            Path(pair["test_path"]),
            str(pair["model"]),
            str(pair["dataset"]),
            str(pair["test_run"]),
            dataset_root=args.dataset_root,
        )

        for group_axis in args.group_axes:
            validation_calibration = cross_fitted_calibration(
                validation,
                folds=args.cv_folds,
                seed=args.cv_seed,
                group_axis=group_axis,
                strict_group=args.strict_group_cv,
            )
            test_calibrated, test_calibration_status = fit_calibrator(validation, test)

            for confidence_threshold in args.confidence_thresholds:
                for reject_fraction in args.uncertainty_reject_fractions:
                    metadata = _policy_metadata(
                        pair,
                        group_axis,
                        args.confidence_source,
                        confidence_threshold,
                        reject_fraction,
                        args.review_fraction,
                    )
                    validation_policy = apply_uncertainty_policy(
                        validation,
                        validation_calibration.probability,
                        confidence_threshold=confidence_threshold,
                        reject_fraction=reject_fraction,
                        confidence_source=args.confidence_source,
                    )
                    test_policy = apply_uncertainty_policy(
                        test,
                        test_calibrated,
                        confidence_threshold=confidence_threshold,
                        reject_fraction=reject_fraction,
                        confidence_source=args.confidence_source,
                    )
                    coverage_rows.append(
                        _coverage(
                            metadata,
                            "validation",
                            validation,
                            validation_policy,
                            validation_calibration.status,
                        )
                    )
                    coverage_rows.append(
                        _coverage(
                            metadata,
                            "test",
                            test,
                            test_policy,
                            test_calibration_status,
                        )
                    )

                    validation_u2_features, validation_missing = _u2_features(validation, validation_policy)
                    test_u2_features, test_missing = _u2_features(test, test_policy)
                    validation_e2_features, validation_e2_mode = _e2_features(
                        validation,
                        dataset=str(pair["dataset"]),
                        dataset_root=args.dataset_root,
                    )
                    test_e2_features, test_e2_mode = _e2_features(
                        test,
                        dataset=str(pair["dataset"]),
                        dataset_root=args.dataset_root,
                    )
                    validation_features = pd.concat([validation_u2_features, validation_e2_features], axis=1)
                    test_features = pd.concat([test_u2_features, test_e2_features], axis=1)
                    feature_rows.append(
                        _feature_availability_row(
                            metadata,
                            split="validation",
                            features=validation_features,
                            residual_mask=validation_policy["residual_mask"],
                            missing_inputs=validation_missing,
                            e2_mode=validation_e2_mode,
                        )
                    )
                    feature_rows.append(
                        _feature_availability_row(
                            metadata,
                            split="test",
                            features=test_features,
                            residual_mask=test_policy["residual_mask"],
                            missing_inputs=test_missing,
                            e2_mode=test_e2_mode,
                        )
                    )

                    validation_u = _uncertainty_result(
                        validation,
                        validation_policy["residual_mask"],
                        validation_calibration.probability,
                        folds=args.cv_folds,
                        seed=args.cv_seed,
                        group_axis=group_axis,
                        strict_group=args.strict_group_cv,
                    )
                    validation_u_metrics = evaluate_ranking(
                        validation,
                        validation_policy["residual_mask"],
                        validation_u.scores,
                        review_fraction=args.review_fraction,
                        min_universe_n=args.min_universe_n,
                        deferred_mask=validation_policy["deferred_mask"],
                    )
                    validation_rows.append(
                        _candidate_row(
                            metadata,
                            split="validation",
                            profile="U",
                            complexity=0,
                            result=validation_u,
                            metrics=validation_u_metrics,
                            policy=validation_policy,
                        )
                    )

                    local_validation_rows: list[dict[str, object]] = []
                    local_test_rows: list[dict[str, object]] = []
                    validation_u2 = _u2_cross_fitted_result(
                        validation,
                        validation_policy["residual_mask"],
                        validation_features,
                        requested_features=U2_FEATURES,
                        profile_name="U2",
                        folds=args.cv_folds,
                        seed=args.cv_seed,
                        group_axis=group_axis,
                        strict_group=args.strict_group_cv,
                        min_fit_n=args.min_fit_n,
                        min_feature_n=args.min_feature_n,
                        ranking_weight=args.ranking_weight,
                        max_pairs=args.max_ranking_pairs,
                    )
                    validation_u2_metrics = evaluate_ranking(
                        validation,
                        validation_policy["residual_mask"],
                        validation_u2.scores,
                        review_fraction=args.review_fraction,
                        min_universe_n=args.min_universe_n,
                        deferred_mask=validation_policy["deferred_mask"],
                    )
                    validation_rows.append(
                        validation_u2_row := _candidate_row(
                            metadata,
                            split="validation",
                            profile="U2",
                            complexity=1,
                            result=validation_u2,
                            metrics=validation_u2_metrics,
                            policy=validation_policy,
                        )
                    )
                    local_validation_rows.append(validation_u2_row)

                    validation_u2e2 = _u2_cross_fitted_result(
                        validation,
                        validation_policy["residual_mask"],
                        validation_features,
                        requested_features=U2_FEATURES + E2_FEATURES,
                        profile_name="U2+E2",
                        folds=args.cv_folds,
                        seed=args.cv_seed,
                        group_axis=group_axis,
                        strict_group=args.strict_group_cv,
                        min_fit_n=args.min_fit_n,
                        min_feature_n=args.min_feature_n,
                        ranking_weight=args.ranking_weight,
                        max_pairs=args.max_ranking_pairs,
                    )
                    validation_u2e2_metrics = evaluate_ranking(
                        validation,
                        validation_policy["residual_mask"],
                        validation_u2e2.scores,
                        review_fraction=args.review_fraction,
                        min_universe_n=args.min_universe_n,
                        deferred_mask=validation_policy["deferred_mask"],
                    )
                    validation_rows.append(
                        validation_u2e2_row := _candidate_row(
                            metadata,
                            split="validation",
                            profile="U2+E2",
                            complexity=2,
                            result=validation_u2e2,
                            metrics=validation_u2e2_metrics,
                            policy=validation_policy,
                        )
                    )
                    local_validation_rows.append(validation_u2e2_row)

                    test_u_scores = pd.Series(np.nan, index=test.index, dtype=float)
                    test_u_scores.loc[test_policy["residual_mask"]] = calibrated_error_risk(
                        test.loc[test_policy["residual_mask"]],
                        test_calibrated.loc[test_policy["residual_mask"]],
                    )
                    test_u = ProfileResult(
                        test_u_scores,
                        pd.Series(-1, index=test.index, dtype=int),
                        (),
                        "complete",
                        ("calibrated_error_risk",),
                    )
                    test_u_metrics = evaluate_ranking(
                        test,
                        test_policy["residual_mask"],
                        test_u.scores,
                        review_fraction=args.review_fraction,
                        min_universe_n=args.min_universe_n,
                        deferred_mask=test_policy["deferred_mask"],
                    )
                    test_rows.append(
                        _candidate_row(
                            metadata,
                            split="test",
                            profile="U",
                            complexity=0,
                            result=test_u,
                            metrics=test_u_metrics,
                            policy=test_policy,
                        )
                    )

                    test_u2 = _u2_test_result(
                        validation,
                        validation_policy["residual_mask"],
                        validation_features,
                        test,
                        test_policy["residual_mask"],
                        test_features,
                        requested_features=U2_FEATURES,
                        profile_name="U2",
                        seed=args.cv_seed,
                        min_fit_n=args.min_fit_n,
                        min_feature_n=args.min_feature_n,
                        ranking_weight=args.ranking_weight,
                        max_pairs=args.max_ranking_pairs,
                    )
                    test_u2_metrics = evaluate_ranking(
                        test,
                        test_policy["residual_mask"],
                        test_u2.scores,
                        review_fraction=args.review_fraction,
                        min_universe_n=args.min_universe_n,
                        deferred_mask=test_policy["deferred_mask"],
                    )
                    test_u2_row = _candidate_row(
                        metadata,
                        split="test",
                        profile="U2",
                        complexity=1,
                        result=test_u2,
                        metrics=test_u2_metrics,
                        policy=test_policy,
                    )
                    test_rows.append(test_u2_row)
                    local_test_rows.append(test_u2_row)

                    test_u2e2 = _u2_test_result(
                        validation,
                        validation_policy["residual_mask"],
                        validation_features,
                        test,
                        test_policy["residual_mask"],
                        test_features,
                        requested_features=U2_FEATURES + E2_FEATURES,
                        profile_name="U2+E2",
                        seed=args.cv_seed,
                        min_fit_n=args.min_fit_n,
                        min_feature_n=args.min_feature_n,
                        ranking_weight=args.ranking_weight,
                        max_pairs=args.max_ranking_pairs,
                    )
                    test_u2e2_metrics = evaluate_ranking(
                        test,
                        test_policy["residual_mask"],
                        test_u2e2.scores,
                        review_fraction=args.review_fraction,
                        min_universe_n=args.min_universe_n,
                        deferred_mask=test_policy["deferred_mask"],
                    )
                    test_u2e2_row = _candidate_row(
                        metadata,
                        split="test",
                        profile="U2+E2",
                        complexity=2,
                        result=test_u2e2,
                        metrics=test_u2e2_metrics,
                        policy=test_policy,
                    )
                    test_rows.append(test_u2e2_row)
                    local_test_rows.append(test_u2e2_row)

                    u2_metric = float(validation_u2_row.get("error_detection_auprc", np.nan))
                    u2e2_metric = float(validation_u2e2_row.get("error_detection_auprc", np.nan))
                    gain = u2e2_metric - u2_metric if np.isfinite(u2_metric) and np.isfinite(u2e2_metric) else np.nan
                    selected_profile = (
                        "U2+E2"
                        if np.isfinite(gain) and gain >= float(args.min_validation_gain)
                        else "U2"
                    )
                    selection_rows.append(
                        {
                            **metadata,
                            "selected_v2_profile": selected_profile,
                            "selection_metric": "validation_error_detection_auprc",
                            "u2_validation_error_detection_auprc": u2_metric,
                            "u2e2_validation_error_detection_auprc": u2e2_metric,
                            "validation_gain_u2e2_minus_u2": gain,
                            "min_validation_gain": float(args.min_validation_gain),
                            "selection_reason": (
                                "u2e2_gain_meets_threshold"
                                if selected_profile == "U2+E2"
                                else "u2e2_gain_below_threshold_or_unavailable"
                            ),
                        }
                    )
                    chosen = [
                        row for row in local_test_rows if str(row.get("profile")) == selected_profile
                    ][0]
                    selected_rows.append({**chosen, "selected_v2_profile": selected_profile})

    coverage = pd.DataFrame(coverage_rows)
    feature_availability = pd.DataFrame(feature_rows)
    validation_candidates = pd.DataFrame(validation_rows)
    test_candidates = pd.DataFrame(test_rows)
    selections = pd.DataFrame(selection_rows)
    selected_test = pd.DataFrame(selected_rows)
    policy_key_frame = (
        coverage.loc[coverage["split"].astype(str).eq("test"), POLICY_KEYS].drop_duplicates()
        if not coverage.empty
        else pd.DataFrame(columns=POLICY_KEYS)
    )
    formal_dir = PROJECT_ROOT / "audit_framework" / "results" / "audit"
    formal_validation = _formal_baseline_rows(
        split="validation",
        policy_keys=policy_key_frame,
        formal_dir=formal_dir,
    )
    formal_test = _formal_baseline_rows(
        split="test",
        policy_keys=policy_key_frame,
        formal_dir=formal_dir,
    )
    validation_candidates_all = pd.concat(
        [validation_candidates, formal_validation],
        ignore_index=True,
        sort=False,
    )
    test_candidates_all = pd.concat(
        [test_candidates, formal_test],
        ignore_index=True,
        sort=False,
    )

    write_csv(_manifest(), out_dir / "profile_manifest_v2_phase1.csv", args.precision)
    write_csv(coverage, out_dir / "policy_coverage_v2_phase1.csv", args.precision)
    write_csv(
        feature_availability,
        out_dir / "feature_availability_v2_phase1.csv",
        args.precision,
    )
    write_csv(
        validation_candidates_all,
        out_dir / "candidate_validation_runs_v2_phase1.csv",
        args.precision,
    )
    write_csv(
        test_candidates_all,
        out_dir / "candidate_test_runs_v2_phase1.csv",
        args.precision,
    )
    write_csv(selections, out_dir / "selection_choices_v2_phase1.csv", args.precision)
    write_csv(selected_test, out_dir / "selected_test_runs_v2_phase1.csv", args.precision)
    write_csv(
        _selected_summary(selected_test),
        out_dir / "selected_summary_v2_phase1.csv",
        args.precision,
    )
    write_csv(
        _component_increment_summary(test_candidates),
        out_dir / "component_increment_summary_v2_phase1.csv",
        args.precision,
    )
    write_csv(
        _v2_vs_formal_summary(selected_test, formal_test),
        out_dir / "v2_vs_formal_summary_v2_phase1.csv",
        args.precision,
    )


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
