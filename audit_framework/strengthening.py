"""Predeclared strengthening experiments around the frozen PAUSE core."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score

from audit_framework.components import (
    NATIVE_SUPPORT_FEATURES,
    PRIOR_FEATURES,
    PRIMARY_PROFILES,
    SUPPORT_FEATURES,
    ProfileSpec,
    calibrated_error_risk,
    engineer_features,
    profile_requirements_met,
)
from audit_framework.data import prepare_table, write_csv
from audit_framework.modeling import (
    ProfileResult,
    _fit_evidence_calibrators,
    _fit_hybrid_ranker,
    _numeric,
    _predict_ranker,
    _training_features,
    _transform_evidence_probabilities,
    _usable_features,
    cross_fitted_calibration,
    cross_fitted_profile,
    fit_calibrator,
    fit_profile_for_test,
    make_folds,
)
from audit_framework.policy import (
    apply_uncertainty_policy,
    evaluate_ranking,
    select_one_standard_error,
)
from audit_framework.scripts.run_audit import (
    METRICS,
    POLICY_KEYS,
    _discover_pairs,
    _pair_deduplicated_metrics,
    _policy_metadata,
    _stratified_bootstrap,
)
from audit_framework.sharding import MODEL_ROOTS


REVIEW_FRACTIONS = (0.05, 0.10, 0.20, 0.30)
CONFIDENCE_THRESHOLDS = (0.8, 0.9)
REJECT_FRACTIONS = (0.10, 0.20)
GROUP_AXES = ("target", "drug")
SEEDS = ("4", "5", "6", "7", "8")
PRIMARY_NAMES = tuple(profile.name for profile in PRIMARY_PROFILES)
UPE_PROFILE = next(
    profile
    for profile in PRIMARY_PROFILES
    if profile.name == "uncertainty_prior_support"
)
BASELINE_NAMES = (
    "raw_uncertainty",
    "native_max_shift",
    "native_density",
    "fixed_hgb_upe",
)
ALL_METHODS = PRIMARY_NAMES + BASELINE_NAMES
FALSIFICATION_VARIANTS = (
    "actual",
    "permute_prior",
    "permute_support",
    "permute_prior_support",
)
TRANSFER_METRICS = (
    "error_detection_auprc",
    "review_error_rate",
    "combined_recall_of_candidate_errors",
    "retained_accuracy_gain_vs_residual",
)


@dataclass(frozen=True)
class StrengtheningShard:
    model_key: str
    model_name: str
    dataset: str
    test_root: str
    out_dir: Path


def build_strengthening_shards(
    *,
    models: Iterable[str],
    datasets: Iterable[str],
    shard_root: str | Path,
) -> list[StrengtheningShard]:
    root = Path(shard_root)
    shards = []
    for model_key, dataset in product(models, datasets):
        key = str(model_key).lower()
        if key not in MODEL_ROOTS:
            raise ValueError(f"Unknown model: {model_key}")
        model_name, test_root = MODEL_ROOTS[key]
        shards.append(
            StrengtheningShard(
                model_key=key,
                model_name=model_name,
                dataset=str(dataset).lower(),
                test_root=test_root,
                out_dir=root / key / str(dataset).lower(),
            )
        )
    return shards


def _policy_columns() -> list[str]:
    return [
        "model",
        "dataset",
        "test_run",
        "group_axis",
        "confidence_source",
        "confidence_threshold",
        "uncertainty_reject_fraction",
    ]


def _normalise_seed(value: str) -> str:
    return value if str(value).startswith("seed_") else f"seed_{value}"


def _raw_uncertainty(table: pd.DataFrame) -> pd.Series:
    probability = pd.to_numeric(table.get("p_base"), errors="coerce")
    return calibrated_error_risk(table, probability)


def _scalar_score(table: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(table.get(column), errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )


def _balanced_weights(labels: np.ndarray) -> np.ndarray:
    y = np.asarray(labels, dtype=int)
    counts = np.bincount(y, minlength=2).astype(float)
    weights = np.ones(len(y), dtype=float)
    for value in (0, 1):
        if counts[value] > 0:
            weights[y == value] = len(y) / (2.0 * counts[value])
    return weights


def _fit_hgb(
    features: pd.DataFrame,
    labels: np.ndarray,
    *,
    seed: int,
) -> tuple[SimpleImputer, HistGradientBoostingClassifier]:
    imputer = SimpleImputer(strategy="median")
    x = imputer.fit_transform(features)
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=int(seed),
    )
    model.fit(
        x,
        np.asarray(labels, dtype=int),
        sample_weight=_balanced_weights(labels),
    )
    return imputer, model


def _predict_hgb(
    fitted: tuple[SimpleImputer, HistGradientBoostingClassifier],
    features: pd.DataFrame,
) -> np.ndarray:
    imputer, model = fitted
    return model.predict_proba(imputer.transform(features))[:, 1]


def cross_fitted_hgb(
    table: pd.DataFrame,
    residual_mask: pd.Series,
    calibrated_probability: pd.Series,
    *,
    folds: int,
    seed: int,
    group_axis: str,
    strict_group: bool,
    min_fit_n: int,
    min_feature_n: int,
) -> ProfileResult:
    scores = pd.Series(np.nan, index=table.index, dtype=float)
    fold_id = pd.Series(-1, index=table.index, dtype=int)
    labels_all = _numeric(table, "base_wrong").to_numpy(dtype=float)
    positions = np.flatnonzero(
        residual_mask.to_numpy(dtype=bool) & np.isfinite(labels_all)
    )
    if len(positions) < int(min_fit_n):
        return ProfileResult(scores, fold_id, (), "insufficient_residual_n", ())
    labels = labels_all[positions].astype(int)
    if np.unique(labels).size < 2:
        return ProfileResult(scores, fold_id, (), "single_residual_class", ())
    splits = make_folds(
        table.iloc[positions],
        labels,
        folds=folds,
        seed=seed,
        group_axis=group_axis,
        strict_group=strict_group,
    )
    if not splits:
        return ProfileResult(scores, fold_id, (), "group_cv_unavailable", ())

    fold_metrics = []
    used: set[str] = set()
    for fold, (train_pos, valid_pos) in enumerate(splits):
        train_positions = positions[train_pos]
        valid_positions = positions[valid_pos]
        train = table.iloc[train_positions]
        valid = table.iloc[valid_positions]
        calibrators = _fit_evidence_calibrators(train)
        train_features = _training_features(
            train,
            calibrated_probability.iloc[train_positions],
            _transform_evidence_probabilities(train, calibrators),
        )
        selected = _usable_features(
            train_features,
            UPE_PROFILE.features,
            min_feature_n=min_feature_n,
        )
        if not profile_requirements_met(selected, UPE_PROFILE):
            fold_metrics.append(np.nan)
            continue
        valid_features = engineer_features(
            valid,
            calibrated_probability=calibrated_probability.iloc[valid_positions],
            support_reference=train,
            evidence_probabilities=_transform_evidence_probabilities(
                valid,
                calibrators,
            ),
        )
        fitted = _fit_hgb(
            train_features.loc[:, selected],
            labels_all[train_positions].astype(int),
            seed=seed + fold,
        )
        predictions = _predict_hgb(
            fitted,
            valid_features.loc[:, selected],
        )
        scores.iloc[valid_positions] = predictions
        fold_id.iloc[valid_positions] = fold
        used.update(selected)
        fold_labels = labels_all[valid_positions].astype(int)
        fold_metrics.append(
            float(average_precision_score(fold_labels, predictions))
            if np.unique(fold_labels).size == 2
            else np.nan
        )
    finite_n = int(scores.iloc[positions].notna().sum())
    status = (
        "complete"
        if finite_n == len(positions)
        else "partial"
        if finite_n
        else "unavailable"
    )
    return ProfileResult(
        scores,
        fold_id,
        tuple(fold_metrics),
        status,
        tuple(sorted(used)),
    )


def fit_hgb_for_test(
    validation: pd.DataFrame,
    validation_residual_mask: pd.Series,
    validation_calibrated_probability: pd.Series,
    test: pd.DataFrame,
    test_residual_mask: pd.Series,
    test_calibrated_probability: pd.Series,
    *,
    seed: int,
    min_fit_n: int,
    min_feature_n: int,
) -> ProfileResult:
    scores = pd.Series(np.nan, index=test.index, dtype=float)
    fold_id = pd.Series(-1, index=test.index, dtype=int)
    labels_all = _numeric(validation, "base_wrong").to_numpy(dtype=float)
    train_positions = np.flatnonzero(
        validation_residual_mask.to_numpy(dtype=bool)
        & np.isfinite(labels_all)
    )
    test_positions = np.flatnonzero(test_residual_mask.to_numpy(dtype=bool))
    if len(train_positions) < int(min_fit_n):
        return ProfileResult(scores, fold_id, (), "insufficient_residual_n", ())
    labels = labels_all[train_positions].astype(int)
    if np.unique(labels).size < 2:
        return ProfileResult(scores, fold_id, (), "single_residual_class", ())
    if not len(test_positions):
        return ProfileResult(scores, fold_id, (), "empty_test_residual", ())

    train = validation.iloc[train_positions]
    calibrators = _fit_evidence_calibrators(train)
    train_features = _training_features(
        train,
        validation_calibrated_probability.iloc[train_positions],
        _transform_evidence_probabilities(train, calibrators),
    )
    selected = _usable_features(
        train_features,
        UPE_PROFILE.features,
        min_feature_n=min_feature_n,
    )
    if not profile_requirements_met(selected, UPE_PROFILE):
        return ProfileResult(scores, fold_id, (), "unavailable", ())
    test_subset = test.iloc[test_positions]
    test_features = engineer_features(
        test_subset,
        calibrated_probability=test_calibrated_probability.iloc[test_positions],
        support_reference=train,
        evidence_probabilities=_transform_evidence_probabilities(
            test_subset,
            calibrators,
        ),
    )
    fitted = _fit_hgb(
        train_features.loc[:, selected],
        labels,
        seed=seed,
    )
    scores.iloc[test_positions] = _predict_hgb(
        fitted,
        test_features.loc[:, selected],
    )
    return ProfileResult(scores, fold_id, (), "complete", selected)


def _permuted_columns(
    selected: tuple[str, ...],
    blocks: tuple[str, ...],
) -> tuple[str, ...]:
    requested: set[str] = set()
    if "prior" in blocks:
        requested.update(PRIOR_FEATURES)
    if "support" in blocks:
        requested.update(SUPPORT_FEATURES)
    return tuple(column for column in selected if column in requested)


def _permute_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    rng: np.random.Generator,
) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        values = output[column].to_numpy(copy=True)
        output[column] = values[rng.permutation(len(values))]
    return output


def cross_fitted_permuted_upe(
    table: pd.DataFrame,
    residual_mask: pd.Series,
    calibrated_probability: pd.Series,
    *,
    permute_blocks: tuple[str, ...],
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
    positions = np.flatnonzero(
        residual_mask.to_numpy(dtype=bool) & np.isfinite(labels_all)
    )
    if len(positions) < int(min_fit_n):
        return ProfileResult(scores, fold_id, (), "insufficient_residual_n", ())
    labels = labels_all[positions].astype(int)
    if np.unique(labels).size < 2:
        return ProfileResult(scores, fold_id, (), "single_residual_class", ())
    splits = make_folds(
        table.iloc[positions],
        labels,
        folds=folds,
        seed=seed,
        group_axis=group_axis,
        strict_group=strict_group,
    )
    if not splits:
        return ProfileResult(scores, fold_id, (), "group_cv_unavailable", ())

    fold_metrics = []
    used: set[str] = set()
    for fold, (train_pos, valid_pos) in enumerate(splits):
        train_positions = positions[train_pos]
        valid_positions = positions[valid_pos]
        train = table.iloc[train_positions]
        valid = table.iloc[valid_positions]
        calibrators = _fit_evidence_calibrators(train)
        train_features = _training_features(
            train,
            calibrated_probability.iloc[train_positions],
            _transform_evidence_probabilities(train, calibrators),
        )
        selected = _usable_features(
            train_features,
            UPE_PROFILE.features,
            min_feature_n=min_feature_n,
        )
        if not profile_requirements_met(selected, UPE_PROFILE):
            fold_metrics.append(np.nan)
            continue
        valid_features = engineer_features(
            valid,
            calibrated_probability=calibrated_probability.iloc[valid_positions],
            support_reference=train,
            evidence_probabilities=_transform_evidence_probabilities(
                valid,
                calibrators,
            ),
        )
        columns = _permuted_columns(selected, permute_blocks)
        rng = np.random.default_rng(seed + 1009 * (fold + 1))
        permuted_train = _permute_frame(
            train_features.loc[:, selected],
            columns,
            rng=rng,
        )
        permuted_valid = _permute_frame(
            valid_features.loc[:, selected],
            columns,
            rng=rng,
        )
        fitted = _fit_hybrid_ranker(
            permuted_train,
            labels_all[train_positions].astype(int),
            ranking_weight=ranking_weight,
            max_pairs=max_pairs,
            seed=seed + fold,
        )
        predictions = _predict_ranker(fitted, permuted_valid)
        scores.iloc[valid_positions] = predictions
        fold_id.iloc[valid_positions] = fold
        used.update(selected)
        fold_labels = labels_all[valid_positions].astype(int)
        fold_metrics.append(
            float(average_precision_score(fold_labels, predictions))
            if np.unique(fold_labels).size == 2
            else np.nan
        )
    finite_n = int(scores.iloc[positions].notna().sum())
    status = (
        "complete"
        if finite_n == len(positions)
        else "partial"
        if finite_n
        else "unavailable"
    )
    return ProfileResult(
        scores,
        fold_id,
        tuple(fold_metrics),
        status,
        tuple(sorted(used)),
    )


def _result_row(
    metadata: dict[str, object],
    *,
    split: str,
    method: str,
    method_family: str,
    result: ProfileResult,
    metrics: dict[str, object],
    candidate_n: int,
    deferred_n: int,
    deduplicated_metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    row = {
        **metadata,
        "split": split,
        "method": method,
        "method_family": method_family,
        "model_status": result.status,
        "used_features": "|".join(result.used_features),
        "validation_fold_mean_auprc": result.mean_fold_auprc,
        "validation_fold_se_auprc": result.se_fold_auprc,
        "candidate_n": int(candidate_n),
        "deferred_n": int(deferred_n),
        **metrics,
    }
    if deduplicated_metrics:
        row.update(
            {
                f"deduplicated_{key}": value
                for key, value in deduplicated_metrics.items()
            }
        )
    row["total_action_n"] = int(row["deferred_n"] + row["reviewed_n"])
    row["total_action_fraction"] = (
        float(row["total_action_n"] / row["candidate_n"])
        if row["candidate_n"]
        else np.nan
    )
    return row


def _static_result(
    scores: pd.Series,
    *,
    status: str,
    features: tuple[str, ...],
) -> ProfileResult:
    return ProfileResult(
        scores=scores,
        fold_id=pd.Series(-1, index=scores.index, dtype=int),
        fold_auprcs=(),
        status=status,
        used_features=features,
    )


def _baseline_scores(
    table: pd.DataFrame,
) -> dict[str, ProfileResult]:
    return {
        "raw_uncertainty": _static_result(
            _raw_uncertainty(table),
            status="complete",
            features=("raw_error_risk",),
        ),
        "native_max_shift": _static_result(
            _scalar_score(table, "native_domain_shift_score"),
            status=(
                "complete"
                if "native_domain_shift_score" in table
                else "unavailable"
            ),
            features=("native_domain_shift_score",),
        ),
        "native_density": _static_result(
            _scalar_score(table, "native_domain_density_distance"),
            status=(
                "complete"
                if "native_domain_density_distance" in table
                else "unavailable"
            ),
            features=("native_domain_density_distance",),
        ),
    }


def run_strengthening_shard(
    shard: StrengtheningShard,
    *,
    calibration_root: str,
    dataset_root: str,
    review_fractions: Iterable[float] = REVIEW_FRACTIONS,
    precision: int = 12,
    cv_folds: int = 3,
    cv_seed: int = 2026,
    min_fit_n: int = 20,
    min_feature_n: int = 10,
    min_universe_n: int = 20,
    min_validation_gain: float = 0.005,
    ranking_weight: float = 0.50,
    max_ranking_pairs: int = 5000,
) -> None:
    class Args:
        pass

    args = Args()
    args.calibration_root = calibration_root
    args.dataset_root = dataset_root
    args.test_roots = [shard.test_root]
    args.datasets = [shard.dataset]
    args.seeds = list(SEEDS)
    pairs, missing = _discover_pairs(args)
    if not missing.empty:
        raise RuntimeError(
            f"{shard.model_key}/{shard.dataset} missing inputs: "
            f"{missing.to_dict('records')}"
        )
    if len(pairs) != len(SEEDS):
        raise RuntimeError(
            f"{shard.model_key}/{shard.dataset} expected {len(SEEDS)} "
            f"pairs, found {len(pairs)}"
        )

    candidate_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    falsification_rows: list[dict[str, object]] = []
    for pair_index, pair in enumerate(pairs):
        print(
            f"[strengthening] {pair['model']}/{pair['dataset']}/"
            f"{pair['test_run']}",
            flush=True,
        )
        validation = prepare_table(
            Path(pair["validation_path"]),
            str(pair["model"]),
            str(pair["dataset"]),
            str(pair["test_run"]),
            dataset_root=dataset_root,
        )
        test = prepare_table(
            Path(pair["test_path"]),
            str(pair["model"]),
            str(pair["dataset"]),
            str(pair["test_run"]),
            dataset_root=dataset_root,
        )
        test_calibrated, _ = fit_calibrator(validation, test)
        for group_index, group_axis in enumerate(GROUP_AXES):
            validation_calibration = cross_fitted_calibration(
                validation,
                folds=cv_folds,
                seed=cv_seed,
                group_axis=group_axis,
                strict_group=True,
            )
            for policy_index, (threshold, reject_fraction) in enumerate(
                product(CONFIDENCE_THRESHOLDS, REJECT_FRACTIONS)
            ):
                selection_metadata = _policy_metadata(
                    pair,
                    group_axis,
                    "base",
                    threshold,
                    reject_fraction,
                    0.20,
                )
                validation_policy = apply_uncertainty_policy(
                    validation,
                    validation_calibration.probability,
                    confidence_threshold=threshold,
                    reject_fraction=reject_fraction,
                    confidence_source="base",
                )
                test_policy = apply_uncertainty_policy(
                    test,
                    test_calibrated,
                    confidence_threshold=threshold,
                    reject_fraction=reject_fraction,
                    confidence_source="base",
                )

                validation_results = {}
                test_results = {}
                for profile in PRIMARY_PROFILES:
                    validation_results[profile.name] = cross_fitted_profile(
                        validation,
                        validation_policy["residual_mask"],
                        validation_calibration.probability,
                        profile,
                        folds=cv_folds,
                        seed=cv_seed,
                        group_axis=group_axis,
                        strict_group=True,
                        min_fit_n=min_fit_n,
                        min_feature_n=min_feature_n,
                        ranking_weight=ranking_weight,
                        max_pairs=max_ranking_pairs,
                    )
                    test_results[profile.name] = fit_profile_for_test(
                        validation,
                        validation_policy["residual_mask"],
                        validation_calibration.probability,
                        test,
                        test_policy["residual_mask"],
                        test_calibrated,
                        profile,
                        seed=cv_seed,
                        min_fit_n=min_fit_n,
                        min_feature_n=min_feature_n,
                        ranking_weight=ranking_weight,
                        max_pairs=max_ranking_pairs,
                    )

                validation_results.update(_baseline_scores(validation))
                test_results.update(_baseline_scores(test))
                validation_results["fixed_hgb_upe"] = cross_fitted_hgb(
                    validation,
                    validation_policy["residual_mask"],
                    validation_calibration.probability,
                    folds=cv_folds,
                    seed=cv_seed,
                    group_axis=group_axis,
                    strict_group=True,
                    min_fit_n=min_fit_n,
                    min_feature_n=min_feature_n,
                )
                test_results["fixed_hgb_upe"] = fit_hgb_for_test(
                    validation,
                    validation_policy["residual_mask"],
                    validation_calibration.probability,
                    test,
                    test_policy["residual_mask"],
                    test_calibrated,
                    seed=cv_seed,
                    min_fit_n=min_fit_n,
                    min_feature_n=min_feature_n,
                )

                selection_candidates = []
                for profile in PRIMARY_PROFILES:
                    result = validation_results[profile.name]
                    selection_candidates.append(
                        {
                            **selection_metadata,
                            "profile": profile.name,
                            "primary_candidate": True,
                            "complexity": profile.complexity,
                            "validation_fold_mean_auprc": (
                                result.mean_fold_auprc
                            ),
                            "validation_fold_se_auprc": (
                                result.se_fold_auprc
                            ),
                        }
                    )
                selected = select_one_standard_error(
                    pd.DataFrame(selection_candidates),
                    min_validation_gain=min_validation_gain,
                )
                selected_name = str(selected["selected_profile"])

                permutation_seed = (
                    cv_seed
                    + pair_index * 10_000
                    + group_index * 1_000
                    + policy_index * 100
                )
                actual = validation_results[
                    "uncertainty_prior_support"
                ]
                falsification = {"actual": actual}
                for variant, blocks in (
                    ("permute_prior", ("prior",)),
                    ("permute_support", ("support",)),
                    ("permute_prior_support", ("prior", "support")),
                ):
                    falsification[variant] = cross_fitted_permuted_upe(
                        validation,
                        validation_policy["residual_mask"],
                        validation_calibration.probability,
                        permute_blocks=blocks,
                        folds=cv_folds,
                        seed=permutation_seed,
                        group_axis=group_axis,
                        strict_group=True,
                        min_fit_n=min_fit_n,
                        min_feature_n=min_feature_n,
                        ranking_weight=ranking_weight,
                        max_pairs=max_ranking_pairs,
                    )
                for variant, result in falsification.items():
                    falsification_rows.append(
                        {
                            **{
                                key: value
                                for key, value in selection_metadata.items()
                                if key != "review_fraction"
                            },
                            "variant": variant,
                            "model_status": result.status,
                            "validation_fold_mean_auprc": (
                                result.mean_fold_auprc
                            ),
                            "actual_validation_fold_mean_auprc": (
                                actual.mean_fold_auprc
                            ),
                            "delta_vs_actual": (
                                result.mean_fold_auprc
                                - actual.mean_fold_auprc
                            ),
                        }
                    )

                for review_fraction in review_fractions:
                    metadata = {
                        **selection_metadata,
                        "review_fraction": float(review_fraction),
                    }
                    local_rows = []
                    for split, table, policy, results in (
                        (
                            "validation",
                            validation,
                            validation_policy,
                            validation_results,
                        ),
                        ("test", test, test_policy, test_results),
                    ):
                        for method in ALL_METHODS:
                            result = results[method]
                            metrics = evaluate_ranking(
                                table,
                                policy["residual_mask"],
                                result.scores,
                                review_fraction=review_fraction,
                                min_universe_n=min_universe_n,
                                deferred_mask=policy["deferred_mask"],
                            )
                            deduplicated = {}
                            if split == "test" and shard.dataset == "human":
                                deduplicated, _ = _pair_deduplicated_metrics(
                                    validation,
                                    test,
                                    test_policy["residual_mask"],
                                    test_policy["deferred_mask"],
                                    result.scores,
                                    review_fraction=review_fraction,
                                    min_universe_n=min_universe_n,
                                )
                            row = _result_row(
                                metadata,
                                split=split,
                                method=method,
                                method_family=(
                                    "primary"
                                    if method in PRIMARY_NAMES
                                    else "baseline"
                                ),
                                result=result,
                                metrics=metrics,
                                candidate_n=policy["candidate_n"],
                                deferred_n=policy["deferred_n"],
                                deduplicated_metrics=deduplicated,
                            )
                            candidate_rows.append(row)
                            if split == "test":
                                local_rows.append(row)

                    local_test = pd.DataFrame(local_rows)
                    selected_row = local_test.loc[
                        local_test["method"].eq(selected_name)
                    ].iloc[0].to_dict()
                    uncertainty_row = local_test.loc[
                        local_test["method"].eq("uncertainty")
                    ].iloc[0]
                    selected_row.update(
                        {
                            "selected_profile": selected_name,
                            "selection_fallback": selected[
                                "selection_fallback"
                            ],
                            "selection_reason": selected[
                                "selection_reason"
                            ],
                        }
                    )
                    for metric in METRICS:
                        selected_row[f"uncertainty_{metric}"] = (
                            uncertainty_row[metric]
                        )
                        selected_row[f"delta_{metric}_vs_uncertainty"] = (
                            selected_row[metric] - uncertainty_row[metric]
                        )
                        dedup_metric = f"deduplicated_{metric}"
                        if dedup_metric in selected_row:
                            selected_row[
                                f"uncertainty_{dedup_metric}"
                            ] = uncertainty_row.get(dedup_metric, np.nan)
                            selected_row[
                                f"delta_{dedup_metric}_vs_uncertainty"
                            ] = (
                                selected_row[dedup_metric]
                                - uncertainty_row.get(dedup_metric, np.nan)
                            )
                    selected_rows.append(selected_row)

    shard.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        pd.DataFrame(candidate_rows),
        shard.out_dir / "candidate_budget_runs.csv",
        precision,
    )
    write_csv(
        pd.DataFrame(selected_rows),
        shard.out_dir / "selected_budget_runs.csv",
        precision,
    )
    write_csv(
        pd.DataFrame(falsification_rows),
        shard.out_dir / "falsification_validation_runs.csv",
        precision,
    )
    manifest = {
        "model_key": shard.model_key,
        "model_name": shard.model_name,
        "dataset": shard.dataset,
        "seeds": list(SEEDS),
        "group_axes": list(GROUP_AXES),
        "confidence_thresholds": list(CONFIDENCE_THRESHOLDS),
        "reject_fractions": list(REJECT_FRACTIONS),
        "review_fractions": [float(value) for value in review_fractions],
        "methods": list(ALL_METHODS),
        "falsification_variants": list(FALSIFICATION_VARIANTS),
    }
    (shard.out_dir / "shard_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    validate_strengthening_shard(shard)
    (shard.out_dir / "COMPLETE").write_text("complete\n", encoding="ascii")


def validate_strengthening_shard(
    shard: StrengtheningShard,
) -> dict[str, int]:
    required = (
        "candidate_budget_runs.csv",
        "selected_budget_runs.csv",
        "falsification_validation_runs.csv",
        "shard_manifest.json",
    )
    missing = [name for name in required if not (shard.out_dir / name).exists()]
    if missing:
        raise RuntimeError(
            f"{shard.model_key}/{shard.dataset} missing {missing}"
        )
    candidates = pd.read_csv(shard.out_dir / required[0])
    selected = pd.read_csv(shard.out_dir / required[1])
    falsification = pd.read_csv(shard.out_dir / required[2])
    policy_count = len(SEEDS) * len(GROUP_AXES) * len(
        CONFIDENCE_THRESHOLDS
    ) * len(REJECT_FRACTIONS)
    expected_candidate = (
        policy_count * len(REVIEW_FRACTIONS) * 2 * len(ALL_METHODS)
    )
    expected_selected = policy_count * len(REVIEW_FRACTIONS)
    expected_falsification = policy_count * len(FALSIFICATION_VARIANTS)
    observed = {
        "candidate_rows": len(candidates),
        "selected_rows": len(selected),
        "falsification_rows": len(falsification),
    }
    expected = {
        "candidate_rows": expected_candidate,
        "selected_rows": expected_selected,
        "falsification_rows": expected_falsification,
    }
    if observed != expected:
        raise RuntimeError(
            f"{shard.model_key}/{shard.dataset} incomplete: "
            f"expected {expected}, observed {observed}"
        )
    candidate_keys = (
        _policy_columns()
        + ["review_fraction", "split", "method"]
    )
    selected_keys = _policy_columns() + ["review_fraction"]
    falsification_keys = _policy_columns() + ["variant"]
    for frame, keys, name in (
        (candidates, candidate_keys, required[0]),
        (selected, selected_keys, required[1]),
        (falsification, falsification_keys, required[2]),
    ):
        if frame.duplicated(keys).any():
            raise RuntimeError(
                f"{shard.model_key}/{shard.dataset} duplicate keys in {name}"
            )
    if set(candidates["method"].astype(str)) != set(ALL_METHODS):
        raise RuntimeError("Candidate method set differs from protocol")
    if set(falsification["variant"].astype(str)) != set(
        FALSIFICATION_VARIANTS
    ):
        raise RuntimeError("Falsification variant set differs from protocol")
    return observed


def _summary_rows(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    value_columns: Iterable[str],
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    grouped = [((), frame)] if not group_columns else frame.groupby(
        group_columns,
        dropna=False,
    )
    for group_key, group in grouped:
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        metadata = dict(zip(group_columns, keys))
        for offset, value_column in enumerate(value_columns):
            values = pd.to_numeric(group[value_column], errors="coerce")
            finite = values.dropna()
            if finite.empty:
                continue
            work = group.copy()
            work[value_column] = values
            lo, hi = _stratified_bootstrap(
                work,
                value_column,
                resamples=resamples,
                seed=seed + offset,
            )
            rows.append(
                {
                    **metadata,
                    "measure": value_column,
                    "mean": float(finite.mean()),
                    "ci_low": lo,
                    "ci_high": hi,
                    "independent_runs": int(len(finite)),
                    "positive_fraction": float((finite > 0.0).mean()),
                }
            )
    return pd.DataFrame(rows)


def _cluster_selected(selected: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "model",
        "dataset",
        "test_run",
        "group_axis",
        "review_fraction",
    ]
    value_columns = [
        column
        for column in selected.columns
        if column.startswith("delta_")
        and column.endswith("_vs_uncertainty")
    ]
    numeric = selected.copy()
    for column in value_columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    return numeric.groupby(keys, dropna=False)[value_columns].mean().reset_index()


def _curve_auc(selected: pd.DataFrame) -> pd.DataFrame:
    keys = _policy_columns()
    rows = []
    for key, group in selected.groupby(keys, dropna=False):
        ordered = group.sort_values("total_action_fraction")
        x = pd.to_numeric(
            ordered["total_action_fraction"],
            errors="coerce",
        ).to_numpy(dtype=float)
        y = pd.to_numeric(
            ordered["combined_recall_of_candidate_errors"],
            errors="coerce",
        ).to_numpy(dtype=float)
        u = pd.to_numeric(
            ordered["uncertainty_combined_recall_of_candidate_errors"],
            errors="coerce",
        ).to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(u)
        x, y, u = x[valid], y[valid], u[valid]
        if len(x) < 2 or float(x.max() - x.min()) <= 0.0:
            continue
        span = float(x.max() - x.min())
        rows.append(
            {
                **dict(zip(keys, key)),
                "pause_normalized_action_auc": float(
                    np.trapezoid(y, x) / span
                ),
                "uncertainty_normalized_action_auc": float(
                    np.trapezoid(u, x) / span
                ),
                "delta_normalized_action_auc_vs_uncertainty": float(
                    np.trapezoid(y - u, x) / span
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    cluster_keys = ["model", "dataset", "test_run", "group_axis"]
    value_columns = [
        column
        for column in frame.columns
        if column.endswith("action_auc")
        or column.endswith("vs_uncertainty")
    ]
    return frame.groupby(
        cluster_keys,
        dropna=False,
    )[value_columns].mean().reset_index()


def merge_and_summarize_strengthening(
    shards: Iterable[StrengtheningShard],
    *,
    merged_dir: str | Path,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    precision: int = 12,
) -> None:
    destination = Path(merged_dir)
    destination.mkdir(parents=True, exist_ok=True)
    tables = {
        "candidate_budget_runs.csv": [],
        "selected_budget_runs.csv": [],
        "falsification_validation_runs.csv": [],
    }
    for shard in shards:
        validate_strengthening_shard(shard)
        for name in tables:
            tables[name].append(pd.read_csv(shard.out_dir / name))
    merged = {
        name: pd.concat(frames, ignore_index=True)
        for name, frames in tables.items()
    }
    for name, frame in merged.items():
        write_csv(frame, destination / name, precision)

    selected_clusters = _cluster_selected(merged["selected_budget_runs.csv"])
    write_csv(
        selected_clusters,
        destination / "equal_budget_cluster_runs.csv",
        precision,
    )
    equal_summary = _summary_rows(
        selected_clusters,
        group_columns=["group_axis", "review_fraction"],
        value_columns=[
            "delta_combined_recall_of_candidate_errors_vs_uncertainty",
            "delta_review_error_rate_vs_uncertainty",
            "delta_retained_accuracy_gain_vs_residual_vs_uncertainty",
            "delta_error_detection_auprc_vs_uncertainty",
        ],
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    write_csv(
        equal_summary,
        destination / "equal_budget_summary.csv",
        precision,
    )

    heterogeneity = []
    for axis in ("model", "dataset", "test_run"):
        grouped = (
            selected_clusters.groupby(
                ["group_axis", "review_fraction", axis],
                dropna=False,
            )[
                [
                    "delta_combined_recall_of_candidate_errors_vs_uncertainty",
                    "delta_review_error_rate_vs_uncertainty",
                    (
                        "delta_retained_accuracy_gain_vs_residual"
                        "_vs_uncertainty"
                    ),
                ]
            ]
            .mean()
            .reset_index()
        )
        grouped["heterogeneity_axis"] = axis
        grouped["stratum"] = grouped[axis].astype(str)
        heterogeneity.append(
            grouped.drop(columns=[axis], errors="ignore")
        )
    write_csv(
        pd.concat(heterogeneity, ignore_index=True),
        destination / "equal_budget_heterogeneity.csv",
        precision,
    )

    curve = _curve_auc(merged["selected_budget_runs.csv"])
    write_csv(curve, destination / "equal_budget_curve_auc_runs.csv", precision)
    curve_summary = _summary_rows(
        curve,
        group_columns=["group_axis"],
        value_columns=["delta_normalized_action_auc_vs_uncertainty"],
        resamples=bootstrap_resamples,
        seed=bootstrap_seed + 1_000,
    )
    write_csv(
        curve_summary,
        destination / "equal_budget_curve_auc_summary.csv",
        precision,
    )

    human = selected_clusters.loc[
        selected_clusters["dataset"].astype(str).eq("human")
    ]
    dedup_columns = [
        column
        for column in selected_clusters
        if column.startswith("delta_deduplicated_")
        and column.endswith("_vs_uncertainty")
    ]
    if dedup_columns:
        human_summary = _summary_rows(
            human,
            group_columns=["group_axis", "review_fraction"],
            value_columns=dedup_columns,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed + 2_000,
        )
        write_csv(
            human_summary,
            destination / "equal_budget_human_deduplicated_summary.csv",
            precision,
        )

    candidates = merged["candidate_budget_runs.csv"]
    test_candidates = candidates.loc[candidates["split"].eq("test")].copy()
    policy_budget = _policy_columns() + ["review_fraction"]
    selected_test = merged["selected_budget_runs.csv"]
    pause_columns = policy_budget + [
        "combined_recall_of_candidate_errors",
        "review_error_rate",
        "retained_accuracy_gain_vs_residual",
        "error_detection_auprc",
    ]
    pause = selected_test[pause_columns].rename(
        columns={
            metric: f"pause_{metric}"
            for metric in TRANSFER_METRICS
        }
    )
    baselines = test_candidates.loc[
        test_candidates["method"].isin(BASELINE_NAMES)
    ].merge(pause, on=policy_budget, how="left", validate="many_to_one")
    for metric in TRANSFER_METRICS:
        baselines[f"delta_{metric}_vs_pause"] = (
            pd.to_numeric(baselines[metric], errors="coerce")
            - pd.to_numeric(baselines[f"pause_{metric}"], errors="coerce")
        )
    baseline_clusters = (
        baselines.groupby(
            [
                "model",
                "dataset",
                "test_run",
                "group_axis",
                "review_fraction",
                "method",
            ],
            dropna=False,
        )[
            [f"delta_{metric}_vs_pause" for metric in TRANSFER_METRICS]
        ]
        .mean()
        .reset_index()
    )
    write_csv(
        baseline_clusters,
        destination / "strong_baseline_cluster_runs.csv",
        precision,
    )
    baseline_summary = _summary_rows(
        baseline_clusters,
        group_columns=["group_axis", "review_fraction", "method"],
        value_columns=[
            f"delta_{metric}_vs_pause" for metric in TRANSFER_METRICS
        ],
        resamples=bootstrap_resamples,
        seed=bootstrap_seed + 3_000,
    )
    write_csv(
        baseline_summary,
        destination / "strong_baseline_summary.csv",
        precision,
    )

    falsification = merged["falsification_validation_runs.csv"]
    actual = falsification.loc[
        falsification["variant"].eq("actual"),
        _policy_columns() + ["validation_fold_mean_auprc"],
    ].rename(
        columns={
            "validation_fold_mean_auprc": (
                "actual_validation_fold_mean_auprc"
            )
        }
    )
    controls = falsification.loc[
        ~falsification["variant"].eq("actual")
    ].merge(
        actual,
        on=_policy_columns(),
        how="left",
        validate="many_to_one",
    )
    controls["actual_minus_control_auprc"] = (
        pd.to_numeric(
            controls["actual_validation_fold_mean_auprc_y"],
            errors="coerce",
        )
        - pd.to_numeric(
            controls["validation_fold_mean_auprc"],
            errors="coerce",
        )
    )
    falsification_clusters = (
        controls.groupby(
            [
                "model",
                "dataset",
                "test_run",
                "group_axis",
                "variant",
            ],
            dropna=False,
        )["actual_minus_control_auprc"]
        .mean()
        .reset_index()
    )
    write_csv(
        falsification_clusters,
        destination / "falsification_cluster_runs.csv",
        precision,
    )
    falsification_summary = _summary_rows(
        falsification_clusters,
        group_columns=["group_axis", "variant"],
        value_columns=["actual_minus_control_auprc"],
        resamples=bootstrap_resamples,
        seed=bootstrap_seed + 4_000,
    )
    write_csv(
        falsification_summary,
        destination / "falsification_summary.csv",
        precision,
    )


def _global_one_se_choice(
    validation_clusters: pd.DataFrame,
    *,
    min_gain: float,
) -> dict[str, object]:
    stats = (
        validation_clusters.groupby(
            ["profile", "complexity"],
            dropna=False,
        )["validation_fold_mean_auprc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats["se"] = stats["std"] / np.sqrt(stats["count"].clip(lower=1))
    finite = stats.loc[stats["mean"].notna()].copy()
    if finite.empty:
        return {
            "selected_profile": "uncertainty",
            "selection_reason": "no_finite_source_validation",
        }
    best = finite.sort_values(
        ["mean", "complexity", "profile"],
        ascending=[False, True, True],
        kind="mergesort",
    ).iloc[0]
    best_se = float(best["se"]) if np.isfinite(best["se"]) else 0.0
    threshold = float(best["mean"]) - best_se
    eligible = finite.loc[finite["mean"] >= threshold].sort_values(
        ["complexity", "mean", "profile"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    selected = eligible.iloc[0]
    uncertainty = finite.loc[finite["profile"].eq("uncertainty")]
    uncertainty_mean = (
        float(uncertainty.iloc[0]["mean"])
        if not uncertainty.empty
        else np.nan
    )
    gain = float(selected["mean"]) - uncertainty_mean
    if (
        selected["profile"] != "uncertainty"
        and np.isfinite(gain)
        and gain < float(min_gain)
    ):
        selected_profile = "uncertainty"
        reason = "gain_below_minimum"
    else:
        selected_profile = str(selected["profile"])
        reason = (
            "uncertainty_selected"
            if selected_profile == "uncertainty"
            else "global_one_standard_error"
        )
    return {
        "selected_profile": selected_profile,
        "selection_reason": reason,
        "best_profile": str(best["profile"]),
        "best_mean": float(best["mean"]),
        "best_se": best_se,
        "one_se_threshold": threshold,
        "selected_source_mean": float(selected["mean"]),
        "uncertainty_source_mean": uncertainty_mean,
        "selected_gain_vs_uncertainty": gain,
        "source_cluster_n": int(
            validation_clusters[
                ["model", "dataset", "test_run"]
            ].drop_duplicates().shape[0]
        ),
    }


def run_cross_domain_transfer(
    *,
    formal_dir: str | Path,
    out_dir: str | Path,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    min_gain: float = 0.005,
    precision: int = 12,
) -> None:
    source = Path(formal_dir)
    validation = pd.read_csv(source / "candidate_validation_runs.csv")
    test = pd.read_csv(source / "candidate_test_runs.csv")
    validation = validation.loc[
        validation["profile"].astype(str).isin(PRIMARY_NAMES)
    ].copy()
    test = test.loc[test["profile"].astype(str).isin(PRIMARY_NAMES)].copy()
    complexities = {
        profile.name: profile.complexity for profile in PRIMARY_PROFILES
    }
    validation["complexity"] = validation["profile"].map(complexities)
    cluster_keys = [
        "model",
        "dataset",
        "test_run",
        "group_axis",
        "profile",
        "complexity",
    ]
    validation_clusters = (
        validation.groupby(cluster_keys, dropna=False)[
            "validation_fold_mean_auprc"
        ]
        .mean()
        .reset_index()
    )

    choices = []
    transfer_rows = []
    for transfer_type, domain_column in (
        ("leave_one_dataset_out", "dataset"),
        ("leave_one_model_out", "model"),
    ):
        for group_axis in GROUP_AXES:
            axis_validation = validation_clusters.loc[
                validation_clusters["group_axis"].eq(group_axis)
            ]
            for held_out in sorted(axis_validation[domain_column].unique()):
                source_validation = axis_validation.loc[
                    ~axis_validation[domain_column].eq(held_out)
                ]
                choice = _global_one_se_choice(
                    source_validation,
                    min_gain=min_gain,
                )
                choice_row = {
                    "transfer_type": transfer_type,
                    "held_out_domain": held_out,
                    "group_axis": group_axis,
                    **choice,
                }
                choices.append(choice_row)

                held_test = test.loc[
                    test["group_axis"].eq(group_axis)
                    & test[domain_column].eq(held_out)
                ]
                selected = held_test.loc[
                    held_test["profile"].eq(choice["selected_profile"])
                ].copy()
                uncertainty = held_test.loc[
                    held_test["profile"].eq("uncertainty")
                ].copy()
                merge_keys = POLICY_KEYS
                uncertainty_columns = merge_keys + list(TRANSFER_METRICS)
                dedup_metrics = [
                    f"deduplicated_{metric}"
                    for metric in TRANSFER_METRICS
                    if f"deduplicated_{metric}" in uncertainty
                ]
                uncertainty_columns.extend(dedup_metrics)
                uncertainty = uncertainty[uncertainty_columns].rename(
                    columns={
                        metric: f"uncertainty_{metric}"
                        for metric in list(TRANSFER_METRICS) + dedup_metrics
                    }
                )
                selected = selected.merge(
                    uncertainty,
                    on=merge_keys,
                    how="left",
                    validate="one_to_one",
                )
                selected["transfer_type"] = transfer_type
                selected["held_out_domain"] = held_out
                selected["transferred_profile"] = choice[
                    "selected_profile"
                ]
                for metric in TRANSFER_METRICS:
                    selected[f"delta_{metric}_vs_uncertainty"] = (
                        pd.to_numeric(selected[metric], errors="coerce")
                        - pd.to_numeric(
                            selected[f"uncertainty_{metric}"],
                            errors="coerce",
                        )
                    )
                    dedup_metric = f"deduplicated_{metric}"
                    if dedup_metric in selected:
                        selected[
                            f"delta_{dedup_metric}_vs_uncertainty"
                        ] = (
                            pd.to_numeric(
                                selected[dedup_metric],
                                errors="coerce",
                            )
                            - pd.to_numeric(
                                selected[
                                    f"uncertainty_{dedup_metric}"
                                ],
                                errors="coerce",
                            )
                        )
                transfer_rows.append(selected)

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    choices_frame = pd.DataFrame(choices)
    runs = pd.concat(transfer_rows, ignore_index=True)
    write_csv(choices_frame, destination / "transfer_choices.csv", precision)
    write_csv(runs, destination / "transfer_policy_runs.csv", precision)

    value_columns = [
        f"delta_{metric}_vs_uncertainty" for metric in TRANSFER_METRICS
    ]
    value_columns.extend(
        column
        for column in runs
        if column.startswith("delta_deduplicated_")
        and column.endswith("_vs_uncertainty")
    )
    cluster_keys = [
        "transfer_type",
        "held_out_domain",
        "model",
        "dataset",
        "test_run",
        "group_axis",
        "transferred_profile",
    ]
    clusters = (
        runs.groupby(cluster_keys, dropna=False)[value_columns]
        .mean()
        .reset_index()
    )
    write_csv(clusters, destination / "transfer_cluster_runs.csv", precision)
    summary = _summary_rows(
        clusters,
        group_columns=[
            "transfer_type",
            "held_out_domain",
            "group_axis",
            "transferred_profile",
        ],
        value_columns=value_columns,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    write_csv(summary, destination / "transfer_summary.csv", precision)
