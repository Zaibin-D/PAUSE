"""Calibration and residual-ranking models for PAUSE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - compatibility with older sklearn
    StratifiedGroupKFold = None

from .components import (
    ProfileSpec,
    calibrated_error_risk,
    engineer_features,
    profile_requirements_met,
)


@dataclass
class CalibrationResult:
    probability: pd.Series
    fold_id: pd.Series
    status: str
    group_axis: str


@dataclass
class ProfileResult:
    scores: pd.Series
    fold_id: pd.Series
    fold_auprcs: tuple[float, ...]
    status: str
    used_features: tuple[str, ...]

    @property
    def mean_fold_auprc(self) -> float:
        values = np.asarray(self.fold_auprcs, dtype=float)
        values = values[np.isfinite(values)]
        return float(values.mean()) if values.size else np.nan

    @property
    def se_fold_auprc(self) -> float:
        values = np.asarray(self.fold_auprcs, dtype=float)
        values = values[np.isfinite(values)]
        if values.size <= 1:
            return 0.0 if values.size == 1 else np.nan
        return float(values.std(ddof=1) / np.sqrt(values.size))


def _numeric(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table:
        return pd.Series(np.nan, index=table.index, dtype=float)
    return pd.to_numeric(table[column], errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )


def _raw_probability(table: pd.DataFrame) -> pd.Series:
    if "p_base" in table:
        return _numeric(table, "p_base").clip(0.0, 1.0)
    logits = _numeric(table, "s_base").clip(-40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _raw_channel_probability(table: pd.DataFrame, column: str) -> pd.Series:
    logits = _numeric(table, column).clip(-40.0, 40.0)
    return pd.Series(
        1.0 / (1.0 + np.exp(-logits)),
        index=table.index,
    )


def _group_values(table: pd.DataFrame, group_axis: str) -> np.ndarray:
    if group_axis == "target":
        column = "pr_id"
    elif group_axis == "drug":
        column = "dr_id"
    elif group_axis == "pair":
        if {"dr_id", "pr_id"}.issubset(table.columns):
            return (
                table["dr_id"].astype(str)
                + "::"
                + table["pr_id"].astype(str)
            ).to_numpy()
        return table.index.astype(str).to_numpy()
    elif group_axis == "none":
        return table.index.astype(str).to_numpy()
    else:
        raise ValueError(f"Unsupported group axis: {group_axis}")
    if column not in table:
        return table.index.astype(str).to_numpy()
    return table[column].astype(str).to_numpy()


def make_folds(
    table: pd.DataFrame,
    labels: np.ndarray,
    *,
    folds: int,
    seed: int,
    group_axis: str,
    strict_group: bool,
) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0 or np.unique(labels).size < 2:
        return []
    class_counts = np.bincount(labels)
    min_class = int(class_counts[class_counts > 0].min())
    n_splits = min(int(folds), min_class)
    if n_splits < 2:
        return []

    if group_axis != "none" and StratifiedGroupKFold is not None:
        groups = _group_values(table, group_axis)
        n_splits = min(n_splits, int(pd.Series(groups).nunique()))
        if n_splits >= 2:
            try:
                splitter = StratifiedGroupKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=int(seed),
                )
                splits = list(splitter.split(np.zeros(len(labels)), labels, groups))
                if all(
                    np.unique(labels[train]).size == 2
                    and np.unique(labels[valid]).size == 2
                    for train, valid in splits
                ):
                    return splits
            except ValueError:
                pass
        if strict_group:
            return []

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(seed),
    )
    return list(splitter.split(np.zeros(len(labels)), labels))


def _fit_platt(train_logits: np.ndarray, labels: np.ndarray) -> LogisticRegression | None:
    valid = np.isfinite(train_logits) & np.isfinite(labels)
    x = np.asarray(train_logits[valid], dtype=float).reshape(-1, 1)
    y = np.asarray(labels[valid], dtype=int)
    if len(y) < 4 or np.unique(y).size < 2:
        return None
    model = LogisticRegression(
        C=1.0e6,
        max_iter=1000,
        solver="lbfgs",
    )
    model.fit(x, y)
    return model


def _fit_evidence_calibrators(
    table: pd.DataFrame,
) -> dict[str, LogisticRegression | None]:
    labels = _numeric(table, "label")
    if labels.isna().all():
        base_pred = _numeric(table, "base_pred")
        base_wrong = _numeric(table, "base_wrong")
        labels = base_pred.where(base_wrong.lt(0.5), 1.0 - base_pred)
    label_values = labels.to_numpy(dtype=float)
    return {
        "prior": _fit_platt(
            _numeric(table, "s_prior").to_numpy(dtype=float),
            label_values,
        ),
    }


def _transform_evidence_probabilities(
    table: pd.DataFrame,
    calibrators: dict[str, LogisticRegression | None],
) -> dict[str, pd.Series]:
    output: dict[str, pd.Series] = {}
    for name, column in (("prior", "s_prior"),):
        logits = _numeric(table, column)
        raw = _raw_channel_probability(table, column)
        model = calibrators.get(name)
        valid = logits.notna()
        probability = raw.copy()
        if model is not None and valid.any():
            probability.loc[valid] = model.predict_proba(
                logits.loc[valid].to_numpy(dtype=float).reshape(-1, 1)
            )[:, 1]
        output[name] = probability.clip(0.0, 1.0)
    return output


def cross_fitted_calibration(
    table: pd.DataFrame,
    *,
    folds: int,
    seed: int,
    group_axis: str,
    strict_group: bool = True,
) -> CalibrationResult:
    labels = _numeric(table, "label").to_numpy(dtype=float)
    logits = _numeric(table, "s_base").to_numpy(dtype=float)
    raw = _raw_probability(table)
    valid = np.isfinite(labels) & np.isfinite(logits)
    positions = np.flatnonzero(valid)
    out = raw.copy()
    fold_id = pd.Series(-1, index=table.index, dtype=int)
    if len(positions) < 4 or np.unique(labels[valid]).size < 2:
        return CalibrationResult(out, fold_id, "raw_probability_fallback", group_axis)

    valid_table = table.iloc[positions]
    split_list = make_folds(
        valid_table,
        labels[valid].astype(int),
        folds=folds,
        seed=seed,
        group_axis=group_axis,
        strict_group=strict_group,
    )
    if not split_list:
        return CalibrationResult(out, fold_id, "raw_probability_group_fallback", group_axis)

    calibrated_any = False
    for fold, (train_pos, valid_pos) in enumerate(split_list):
        model = _fit_platt(logits[positions[train_pos]], labels[positions[train_pos]])
        if model is None:
            continue
        predictions = model.predict_proba(
            logits[positions[valid_pos]].reshape(-1, 1)
        )[:, 1]
        out.iloc[positions[valid_pos]] = predictions
        fold_id.iloc[positions[valid_pos]] = fold
        calibrated_any = True
    status = "cross_fitted" if calibrated_any else "raw_probability_fallback"
    return CalibrationResult(out.clip(0.0, 1.0), fold_id, status, group_axis)


def fit_calibrator(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.Series, str]:
    labels = _numeric(train, "label").to_numpy(dtype=float)
    train_logits = _numeric(train, "s_base").to_numpy(dtype=float)
    test_logits = _numeric(test, "s_base").to_numpy(dtype=float)
    raw_test = _raw_probability(test)
    if test.empty:
        return raw_test, "empty_test"
    model = _fit_platt(train_logits, labels)
    if model is None or not np.isfinite(test_logits).all():
        return raw_test, "raw_probability_fallback"
    predictions = model.predict_proba(test_logits.reshape(-1, 1))[:, 1]
    return pd.Series(predictions, index=test.index).clip(0.0, 1.0), "fitted"


def _training_features(
    table: pd.DataFrame,
    calibrated_probability: pd.Series,
    evidence_probabilities: dict[str, pd.Series],
) -> pd.DataFrame:
    features = engineer_features(
        table,
        calibrated_probability=calibrated_probability,
        support_reference=table,
        evidence_probabilities=evidence_probabilities,
    )
    raw_counts: dict[str, pd.Series] = {}
    for columns, prefix in (
        (("dr_id",), "drug"),
        (("pr_id",), "target"),
        (("dr_id", "pr_id"), "pair"),
    ):
        if not set(columns).issubset(table.columns):
            continue
        if len(columns) == 1:
            keys = table[columns[0]].astype(str)
        else:
            keys = pd.Series(
                list(
                    zip(
                        *[
                            table[column].astype(str).to_numpy()
                            for column in columns
                        ]
                    )
                ),
                index=table.index,
                dtype=object,
            )
        counts = keys.map(keys.value_counts()).astype(float) - 1.0
        counts = counts.clip(lower=0.0)
        raw_counts[prefix] = counts
        features[f"{prefix}_seen_in_fit"] = (counts > 0.0).astype(float)
        features[f"log_{prefix}_fit_count"] = np.log1p(counts)
    if {"drug", "target"}.issubset(raw_counts):
        drug_log = np.log1p(raw_counts["drug"])
        target_log = np.log1p(raw_counts["target"])
        features["fit_entity_support_min"] = pd.concat(
            [drug_log, target_log],
            axis=1,
        ).min(axis=1)
        features["fit_entity_support_imbalance"] = (
            drug_log - target_log
        ).abs()
        drug_seen = (raw_counts["drug"] > 0.0).astype(float)
        target_seen = (raw_counts["target"] > 0.0).astype(float)
        features["fit_joint_novelty"] = (
            (1.0 - drug_seen) * (1.0 - target_seen)
        )
        features["fit_one_sided_novelty"] = (drug_seen - target_seen).abs()
    return features


def _usable_features(
    frame: pd.DataFrame,
    requested: tuple[str, ...],
    *,
    min_feature_n: int,
) -> tuple[str, ...]:
    usable = []
    for feature in requested:
        if feature not in frame:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        finite = values.dropna()
        if len(finite) >= int(min_feature_n) and finite.nunique() >= 2:
            usable.append(feature)
    return tuple(usable)


def _fit_hybrid_ranker(
    train_x: pd.DataFrame,
    labels: np.ndarray,
    *,
    ranking_weight: float,
    max_pairs: int,
    seed: int,
) -> tuple[SimpleImputer, StandardScaler, LogisticRegression]:
    imputer = SimpleImputer(strategy="median", add_indicator=False)
    scaler = StandardScaler()
    x = scaler.fit_transform(imputer.fit_transform(train_x))
    y = np.asarray(labels, dtype=int)
    fit_x = [x]
    fit_y = [y]
    fit_weight = [np.ones(len(y), dtype=float)]

    if float(ranking_weight) > 0.0:
        positive = np.flatnonzero(y == 1)
        negative = np.flatnonzero(y == 0)
        if positive.size and negative.size:
            rng = np.random.default_rng(int(seed))
            total_pairs = int(positive.size * negative.size)
            pair_n = min(int(max_pairs), total_pairs)
            pos_idx = rng.choice(positive, size=pair_n, replace=pair_n > positive.size)
            neg_idx = rng.choice(negative, size=pair_n, replace=pair_n > negative.size)
            differences = x[pos_idx] - x[neg_idx]
            fit_x.extend([differences, -differences])
            fit_y.extend(
                [
                    np.ones(pair_n, dtype=int),
                    np.zeros(pair_n, dtype=int),
                ]
            )
            fit_weight.extend(
                [
                    np.full(pair_n, float(ranking_weight), dtype=float),
                    np.full(pair_n, float(ranking_weight), dtype=float),
                ]
            )

    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
    )
    model.fit(
        np.vstack(fit_x),
        np.concatenate(fit_y),
        sample_weight=np.concatenate(fit_weight),
    )
    return imputer, scaler, model


def _predict_ranker(
    fitted: tuple[SimpleImputer, StandardScaler, LogisticRegression],
    test_x: pd.DataFrame,
) -> np.ndarray:
    imputer, scaler, model = fitted
    x = scaler.transform(imputer.transform(test_x))
    return model.predict_proba(x)[:, 1]


def cross_fitted_profile(
    table: pd.DataFrame,
    residual_mask: pd.Series,
    calibrated_probability: pd.Series,
    profile: ProfileSpec,
    *,
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
    residual_positions = np.flatnonzero(residual_mask.to_numpy(dtype=bool))
    if profile.name == "uncertainty":
        scores.loc[residual_mask] = calibrated_error_risk(
            table.loc[residual_mask],
            calibrated_probability.loc[residual_mask],
        )

    labels_all = _numeric(table, "base_wrong").to_numpy(dtype=float)
    valid_positions = residual_positions[np.isfinite(labels_all[residual_positions])]
    if profile.name != "uncertainty" and len(valid_positions) < int(min_fit_n):
        return ProfileResult(scores, fold_id, (), "insufficient_residual_n", ())
    if len(valid_positions) < 2:
        return ProfileResult(scores, fold_id, (), "insufficient_residual_n", ())
    labels = labels_all[valid_positions].astype(int)
    if np.unique(labels).size < 2:
        return ProfileResult(scores, fold_id, (), "single_residual_class", ())

    residual_table = table.iloc[valid_positions]
    split_list = make_folds(
        residual_table,
        labels,
        folds=folds,
        seed=seed,
        group_axis=group_axis,
        strict_group=strict_group,
    )
    if not split_list:
        return ProfileResult(scores, fold_id, (), "group_cv_unavailable", ())

    fold_metrics: list[float] = []
    used_features: set[str] = (
        set(profile.features) if profile.name == "uncertainty" else set()
    )
    for fold, (train_pos, valid_pos) in enumerate(split_list):
        train_positions = valid_positions[train_pos]
        validation_positions = valid_positions[valid_pos]
        if profile.name == "uncertainty":
            fold_predictions = scores.iloc[validation_positions].to_numpy(dtype=float)
        else:
            train_table = table.iloc[train_positions]
            validation_table = table.iloc[validation_positions]
            evidence_calibrators = _fit_evidence_calibrators(train_table)
            train_evidence = _transform_evidence_probabilities(
                train_table,
                evidence_calibrators,
            )
            validation_evidence = _transform_evidence_probabilities(
                validation_table,
                evidence_calibrators,
            )
            train_features = _training_features(
                train_table,
                calibrated_probability.iloc[train_positions],
                train_evidence,
            )
            selected = _usable_features(
                train_features,
                profile.features,
                min_feature_n=min_feature_n,
            )
            if not profile_requirements_met(selected, profile):
                fold_metrics.append(np.nan)
                continue
            if not selected:
                fold_metrics.append(np.nan)
                continue
            validation_features = engineer_features(
                validation_table,
                calibrated_probability=calibrated_probability.iloc[validation_positions],
                support_reference=train_table,
                evidence_probabilities=validation_evidence,
            )
            fitted = _fit_hybrid_ranker(
                train_features.loc[:, selected],
                labels_all[train_positions].astype(int),
                ranking_weight=ranking_weight,
                max_pairs=max_pairs,
                seed=int(seed) + fold,
            )
            fold_predictions = _predict_ranker(
                fitted,
                validation_features.loc[:, selected],
            )
            scores.iloc[validation_positions] = fold_predictions
            used_features.update(selected)
        fold_id.iloc[validation_positions] = fold
        fold_labels = labels_all[validation_positions].astype(int)
        fold_metrics.append(
            float(average_precision_score(fold_labels, fold_predictions))
            if np.unique(fold_labels).size == 2
            else np.nan
        )

    finite_n = int(scores.iloc[valid_positions].notna().sum())
    requirements_met = profile_requirements_met(used_features, profile)
    if not requirements_met:
        status = (
            "profile_unavailable"
            if profile.primary
            else "diagnostic_unavailable"
        )
    elif finite_n == 0:
        status = "unavailable"
    elif finite_n < len(valid_positions):
        status = "partial"
    else:
        status = "complete"
    return ProfileResult(
        scores,
        fold_id,
        tuple(fold_metrics),
        status,
        tuple(sorted(used_features)),
    )


def fit_profile_for_test(
    validation: pd.DataFrame,
    validation_residual_mask: pd.Series,
    validation_calibrated_probability: pd.Series,
    test: pd.DataFrame,
    test_residual_mask: pd.Series,
    test_calibrated_probability: pd.Series,
    profile: ProfileSpec,
    *,
    seed: int,
    min_fit_n: int,
    min_feature_n: int,
    ranking_weight: float,
    max_pairs: int,
) -> ProfileResult:
    scores = pd.Series(np.nan, index=test.index, dtype=float)
    fold_id = pd.Series(-1, index=test.index, dtype=int)
    if profile.name == "uncertainty":
        scores.loc[test_residual_mask] = calibrated_error_risk(
            test.loc[test_residual_mask],
            test_calibrated_probability.loc[test_residual_mask],
        )
        return ProfileResult(scores, fold_id, (), "complete", profile.features)

    train_positions = np.flatnonzero(validation_residual_mask.to_numpy(dtype=bool))
    test_positions = np.flatnonzero(test_residual_mask.to_numpy(dtype=bool))
    labels_all = _numeric(validation, "base_wrong").to_numpy(dtype=float)
    train_positions = train_positions[np.isfinite(labels_all[train_positions])]
    if len(train_positions) < int(min_fit_n):
        return ProfileResult(scores, fold_id, (), "insufficient_residual_n", ())
    labels = labels_all[train_positions].astype(int)
    if np.unique(labels).size < 2:
        return ProfileResult(scores, fold_id, (), "single_residual_class", ())
    if len(test_positions) == 0:
        return ProfileResult(scores, fold_id, (), "empty_test_residual", ())

    train_table = validation.iloc[train_positions]
    evidence_calibrators = _fit_evidence_calibrators(train_table)
    train_evidence = _transform_evidence_probabilities(
        train_table,
        evidence_calibrators,
    )
    train_features = _training_features(
        train_table,
        validation_calibrated_probability.iloc[train_positions],
        train_evidence,
    )
    selected = _usable_features(
        train_features,
        profile.features,
        min_feature_n=min_feature_n,
    )
    if not profile_requirements_met(selected, profile):
        return ProfileResult(
            scores,
            fold_id,
            (),
            (
                "profile_unavailable"
                if profile.primary
                else "diagnostic_unavailable"
            ),
            (),
        )
    if not selected:
        return ProfileResult(scores, fold_id, (), "unavailable", ())
    test_features = engineer_features(
        test.iloc[test_positions],
        calibrated_probability=test_calibrated_probability.iloc[test_positions],
        support_reference=train_table,
        evidence_probabilities=_transform_evidence_probabilities(
            test.iloc[test_positions],
            evidence_calibrators,
        ),
    )
    fitted = _fit_hybrid_ranker(
        train_features.loc[:, selected],
        labels,
        ranking_weight=ranking_weight,
        max_pairs=max_pairs,
        seed=seed,
    )
    scores.iloc[test_positions] = _predict_ranker(
        fitted,
        test_features.loc[:, selected],
    )
    return ProfileResult(scores, fold_id, (), "complete", selected)
