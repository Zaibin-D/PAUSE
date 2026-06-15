"""Run the formal policy-aligned PAUSE residual audit.

This is the only experiment implementation. It uses:

1. cross-fitted Platt calibration for the high-confidence policy;
2. uncertainty-first deferment;
3. grouped cross-fitted models trained only on policy-matched residual rows;
4. a small predeclared primary candidate library;
5. one-standard-error selection with a minimum gain gate and uncertainty fallback;
6. label-free empirical support from source-train and fold-fit references.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from audit_framework.data import (
    AUDIT_NAME,
    DEFAULT_DATASETS,
    DEFAULT_TEST_ROOTS,
    model_name,
    prepare_table,
    source_support_provenance,
    write_csv,
)
from audit_framework import (
    EVIDENCE_DIAGNOSTIC_PROFILES,
    PRIMARY_PROFILES,
    apply_uncertainty_policy,
    audit_capabilities,
    calibrated_error_risk,
    calibration_metrics,
    cross_fitted_calibration,
    cross_fitted_profile,
    evaluate_ranking,
    fit_calibrator,
    fit_profile_for_test,
    profile_manifest,
    select_one_standard_error,
)


POLICY_KEYS = [
    "model",
    "dataset",
    "test_run",
    "group_axis",
    "confidence_source",
    "confidence_threshold",
    "uncertainty_reject_fraction",
    "review_fraction",
]

METRICS = [
    "error_detection_auroc",
    "error_detection_auprc",
    "review_error_rate",
    "review_lift",
    "review_recall_of_residual_errors",
    "combined_recall_of_candidate_errors",
    "retained_accuracy_after_review",
    "retained_accuracy_gain_vs_residual",
]

CORE_INCREMENT_PATH = (
    ("uncertainty", "uncertainty_prior", "prior"),
    ("uncertainty", "uncertainty_support", "empirical_support"),
    (
        "uncertainty_prior",
        "uncertainty_prior_support",
        "empirical_support_after_prior",
    ),
    (
        "uncertainty_support",
        "uncertainty_prior_support",
        "prior_after_empirical_support",
    ),
    (
        "fit_support",
        "legacy_support",
        "legacy_source_domain_support_after_fit",
    ),
    (
        "source_domain_support",
        "legacy_support",
        "fit_support_after_legacy_source_domain",
    ),
    (
        "fit_support",
        "uncertainty_support",
        "native_source_domain_support_after_fit",
    ),
    (
        "native_source_domain_support",
        "uncertainty_support",
        "fit_support_after_native_source_domain",
    ),
    (
        "legacy_support",
        "combined_support",
        "native_support_after_legacy",
    ),
    (
        "uncertainty_support",
        "combined_support",
        "legacy_support_after_native",
    ),
)

DIAGNOSTIC_BASELINES = {
    "fit_support": ("uncertainty", "fit_support"),
    "source_domain_support": ("uncertainty", "source_domain_support"),
    "native_source_domain_support": (
        "uncertainty",
        "native_source_domain_support",
    ),
    "target_sequence_support": (
        "uncertainty_support",
        "target_sequence_support",
    ),
    "general_target_joint_support": (
        "uncertainty_support",
        "general_target_joint_support",
    ),
    "legacy_support": ("uncertainty", "legacy_empirical_support"),
    "combined_support": ("uncertainty", "combined_empirical_support"),
}


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


def _pair_keys(table: pd.DataFrame) -> pd.Series:
    return pd.Series(
        list(
            zip(
                table["dr_id"].map(_canonical_id),
                table["pr_id"].map(_canonical_id),
            )
        ),
        index=table.index,
        dtype=object,
    )


def _pair_deduplicated_metrics(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    residual_mask: pd.Series,
    deferred_mask: pd.Series,
    scores: pd.Series,
    *,
    review_fraction: float,
    min_universe_n: int,
) -> tuple[dict[str, float | int], dict[str, int]]:
    """Evaluate unique held-out pairs after excluding validation overlap."""

    if not {"dr_id", "pr_id", "base_wrong"}.issubset(test.columns):
        return {}, {}
    validation_pairs = (
        set(_pair_keys(validation))
        if {"dr_id", "pr_id"}.issubset(validation.columns)
        else set()
    )
    work = pd.DataFrame(
        {
            "pair_key": _pair_keys(test),
            "base_wrong": pd.to_numeric(test["base_wrong"], errors="coerce"),
            "residual": residual_mask.astype(bool),
            "deferred": deferred_mask.astype(bool),
            "score": pd.to_numeric(scores, errors="coerce"),
        },
        index=test.index,
    )
    overlap = work["pair_key"].isin(validation_pairs)
    eligible = work.loc[~overlap].copy()
    pair_rows = []
    label_conflicts = 0
    mixed_policy_states = 0
    for pair_key, group in eligible.groupby("pair_key", sort=False):
        labels = group["base_wrong"].dropna()
        if labels.nunique() > 1:
            label_conflicts += 1
        is_deferred = bool(group["deferred"].any())
        has_residual = bool(group["residual"].any())
        if is_deferred and has_residual:
            mixed_policy_states += 1
        is_residual = has_residual and not is_deferred
        residual_scores = group.loc[group["residual"], "score"].dropna()
        pair_rows.append(
            {
                "pair_key": pair_key,
                "base_wrong": (
                    float(labels.mean()) if not labels.empty else np.nan
                ),
                "residual": is_residual,
                "deferred": is_deferred,
                "score": (
                    float(residual_scores.mean())
                    if not residual_scores.empty
                    else np.nan
                ),
            }
        )
    pair_table = pd.DataFrame(pair_rows)
    if pair_table.empty:
        metrics = evaluate_ranking(
            pd.DataFrame({"base_wrong": []}),
            pd.Series(dtype=bool),
            pd.Series(dtype=float),
            review_fraction=review_fraction,
            min_universe_n=min_universe_n,
            deferred_mask=pd.Series(dtype=bool),
        )
    else:
        metrics = evaluate_ranking(
            pair_table,
            pair_table["residual"],
            pair_table["score"],
            review_fraction=review_fraction,
            min_universe_n=min_universe_n,
            deferred_mask=pair_table["deferred"],
        )
    diagnostics = {
        "test_rows_before": int(len(work)),
        "validation_unique_pairs": int(len(validation_pairs)),
        "cross_split_overlap_rows_removed": int(overlap.sum()),
        "cross_split_overlap_pairs_removed": int(
            work.loc[overlap, "pair_key"].nunique()
        ),
        "eligible_test_rows": int(len(eligible)),
        "eligible_unique_pairs": int(eligible["pair_key"].nunique()),
        "within_test_duplicate_rows_collapsed": int(
            len(eligible) - eligible["pair_key"].nunique()
        ),
        "pair_label_conflicts": int(label_conflicts),
        "mixed_policy_state_pairs": int(mixed_policy_states),
    }
    return metrics, diagnostics


def _normalise_seeds(values: list[str]) -> set[str]:
    return {
        value if str(value).startswith("seed_") else f"seed_{value}"
        for value in values
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-root",
        default="audit_framework/cache/validation_audits",
    )
    parser.add_argument(
        "--dataset-root",
        default="datasets",
        help="Dataset root used to derive label-free source-training support.",
    )
    parser.add_argument("--test-roots", nargs="+", default=list(DEFAULT_TEST_ROOTS))
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--seeds", nargs="+", default=["4", "5", "6", "7", "8"])
    parser.add_argument("--confidence-thresholds", nargs="+", type=float, default=[0.8, 0.9])
    parser.add_argument("--uncertainty-reject-fractions", nargs="+", type=float, default=[0.10, 0.20])
    parser.add_argument("--review-fraction", type=float, default=0.20)
    parser.add_argument(
        "--confidence-source",
        choices=["base", "calibrated"],
        default="base",
        help=(
            "base preserves the frozen predictor's high-confidence candidate "
            "definition; calibrated requires separately tuned thresholds."
        ),
    )
    parser.add_argument(
        "--group-axes",
        nargs="+",
        choices=["target", "drug", "pair", "none"],
        default=["target"],
        help="Grouped validation axes. Use target and drug for a sensitivity run.",
    )
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--cv-seed", type=int, default=2026)
    parser.add_argument(
        "--strict-group-cv",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--min-fit-n", type=int, default=20)
    parser.add_argument("--min-feature-n", type=int, default=10)
    parser.add_argument("--min-universe-n", type=int, default=20)
    parser.add_argument("--min-validation-gain", type=float, default=0.005)
    parser.add_argument("--ranking-weight", type=float, default=0.50)
    parser.add_argument("--max-ranking-pairs", type=int, default=5000)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument(
        "--summary-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rebuild aggregate summaries from existing run-level CSVs.",
    )
    parser.add_argument(
        "--out-dir",
        default="audit_framework/results/audit",
    )
    parser.add_argument("--precision", type=int, default=6)
    return parser


def _discover_pairs(args: argparse.Namespace) -> tuple[list[dict[str, object]], pd.DataFrame]:
    seeds = _normalise_seeds(args.seeds)
    calibration_root = Path(args.calibration_root)
    pairs: list[dict[str, object]] = []
    missing: list[dict[str, str]] = []
    for root_arg in args.test_roots:
        root = Path(root_arg)
        if not root.exists():
            missing.append(
                {
                    "test_root": str(root),
                    "dataset": "",
                    "test_run": "",
                    "reason": "missing_test_root",
                    "expected_validation": "",
                }
            )
            continue
        for dataset in args.datasets:
            for run_dir in sorted((root / dataset).glob("seed_*")):
                if run_dir.name not in seeds:
                    continue
                test_path = run_dir / AUDIT_NAME
                validation_path = (
                    calibration_root
                    / root.name
                    / dataset
                    / run_dir.name
                    / AUDIT_NAME
                )
                if not test_path.exists() or not validation_path.exists():
                    missing.append(
                        {
                            "test_root": str(root),
                            "dataset": dataset,
                            "test_run": run_dir.name,
                            "reason": (
                                "missing_test_audit"
                                if not test_path.exists()
                                else "missing_validation_audit"
                            ),
                            "expected_validation": str(validation_path),
                        }
                    )
                    continue
                pairs.append(
                    {
                        "model": model_name(root),
                        "dataset": dataset,
                        "test_run": run_dir.name,
                        "validation_path": validation_path,
                        "test_path": test_path,
                    }
                )
    return pairs, pd.DataFrame(
        missing,
        columns=[
            "test_root",
            "dataset",
            "test_run",
            "reason",
            "expected_validation",
        ],
    )


def _policy_metadata(
    pair: dict[str, object],
    group_axis: str,
    confidence_source: str,
    confidence_threshold: float,
    reject_fraction: float,
    review_fraction: float,
) -> dict[str, object]:
    return {
        "model": pair["model"],
        "dataset": pair["dataset"],
        "test_run": pair["test_run"],
        "group_axis": group_axis,
        "confidence_source": confidence_source,
        "confidence_threshold": float(confidence_threshold),
        "uncertainty_reject_fraction": float(reject_fraction),
        "review_fraction": float(review_fraction),
    }


def _coverage(
    metadata: dict[str, object],
    split: str,
    table: pd.DataFrame,
    policy: dict[str, object],
    calibration_status: str,
) -> dict[str, object]:
    labels = pd.to_numeric(table["base_wrong"], errors="coerce")
    candidate = policy["candidate_mask"]
    deferred = policy["deferred_mask"]
    residual = policy["residual_mask"]
    return {
        **metadata,
        "split": split,
        "calibration_status": calibration_status,
        "candidate_n": int(candidate.sum()),
        "candidate_error_count": float(labels.loc[candidate].sum()),
        "deferred_n": int(deferred.sum()),
        "deferred_error_count": float(labels.loc[deferred].sum()),
        "residual_n": int(residual.sum()),
        "residual_error_count": float(labels.loc[residual].sum()),
        "residual_has_two_classes": bool(labels.loc[residual].dropna().nunique() >= 2),
    }


def _capability_row(
    pair: dict[str, object],
    split: str,
    table: pd.DataFrame,
) -> dict[str, object]:
    return {
        "model": pair["model"],
        "dataset": pair["dataset"],
        "test_run": pair["test_run"],
        "split": split,
        **audit_capabilities(table),
    }


def _candidate_row(
    metadata: dict[str, object],
    profile,
    result,
    metrics: dict[str, object],
) -> dict[str, object]:
    return {
        **metadata,
        "profile": profile.name,
        "primary_candidate": profile.primary,
        "complexity": profile.complexity,
        "model_status": result.status,
        "used_features": "|".join(result.used_features),
        "validation_fold_mean_auprc": result.mean_fold_auprc,
        "validation_fold_se_auprc": result.se_fold_auprc,
        "validation_finite_folds": int(
            np.isfinite(np.asarray(result.fold_auprcs, dtype=float)).sum()
        ),
        **metrics,
    }


def _calibration_rows(
    metadata: dict[str, object],
    split: str,
    table: pd.DataFrame,
    calibrated_probability: pd.Series,
    policy: dict[str, object],
) -> list[dict[str, object]]:
    raw_probability = pd.to_numeric(table["p_base"], errors="coerce")
    score_sources = {
        "raw": calibrated_error_risk(table, raw_probability),
        "calibrated": calibrated_error_risk(table, calibrated_probability),
    }
    scopes = {
        "candidate": policy["candidate_mask"],
        "deferred": policy["deferred_mask"],
        "residual": policy["residual_mask"],
    }
    rows = []
    for scope, mask in scopes.items():
        for score_source, score in score_sources.items():
            rows.append(
                {
                    **metadata,
                    "split": split,
                    "scope": scope,
                    "score_source": score_source,
                    **calibration_metrics(table, score, mask=mask),
                }
            )
    return rows


def _domain_support_rows(
    metadata: dict[str, object],
    table: pd.DataFrame,
    residual_mask: pd.Series,
    support_reference: pd.DataFrame,
) -> list[dict[str, object]]:
    """Describe residual errors by audit-fit and source-train coverage."""

    required = {"dr_id", "pr_id", "base_wrong"}
    if not required.issubset(table.columns) or not {
        "dr_id",
        "pr_id",
    }.issubset(support_reference.columns):
        return [
            {
                **metadata,
                "split": "test",
                "support_reference": "unavailable",
                "support_state": "unavailable",
                "residual_n": 0,
                "residual_share": np.nan,
                "error_count": np.nan,
                "error_rate": np.nan,
                "error_rate_delta_vs_all": np.nan,
            }
        ]

    residual = table.loc[residual_mask, ["dr_id", "pr_id", "base_wrong"]].copy()
    if residual.empty:
        return []
    residual["base_wrong"] = pd.to_numeric(
        residual["base_wrong"],
        errors="coerce",
    )
    rows = []
    reference_drugs = set(support_reference["dr_id"].astype(str))
    reference_targets = set(support_reference["pr_id"].astype(str))
    support_sources = {
        "audit_fit": (
            residual["dr_id"].astype(str).isin(reference_drugs),
            residual["pr_id"].astype(str).isin(reference_targets),
        ),
    }
    if {"source_drug_count", "source_target_count"}.issubset(table.columns):
        source_drug_count = pd.to_numeric(
            table.loc[residual.index, "source_drug_count"],
            errors="coerce",
        )
        source_target_count = pd.to_numeric(
            table.loc[residual.index, "source_target_count"],
            errors="coerce",
        )
        support_sources["source_train"] = (
            source_drug_count.gt(0.0),
            source_target_count.gt(0.0),
        )

    for support_reference_name, (drug_seen, target_seen) in support_sources.items():
        scoped = residual.copy()
        scoped["support_state"] = np.select(
            [
                drug_seen & target_seen,
                ~drug_seen & target_seen,
                drug_seen & ~target_seen,
            ],
            [
                "both_seen",
                "drug_unseen",
                "target_unseen",
            ],
            default="both_unseen",
        )
        all_error_rate = float(scoped["base_wrong"].mean())
        scopes = [("all", scoped)]
        scopes.extend(scoped.groupby("support_state", sort=True))
        for support_state, group in scopes:
            error_count = float(group["base_wrong"].sum())
            error_rate = float(group["base_wrong"].mean())
            rows.append(
                {
                    **metadata,
                    "split": "test",
                    "support_reference": support_reference_name,
                    "support_state": support_state,
                    "residual_n": int(len(group)),
                    "residual_share": float(len(group) / len(scoped)),
                    "error_count": error_count,
                    "error_rate": error_rate,
                    "error_rate_delta_vs_all": error_rate - all_error_rate,
                }
            )
    return rows


def _audit_fit_reference_row(
    metadata: dict[str, object],
    validation: pd.DataFrame,
    validation_residual_mask: pd.Series,
    test: pd.DataFrame,
    test_residual_mask: pd.Series,
) -> dict[str, object]:
    reference = validation.loc[validation_residual_mask]
    evaluated = test.loc[test_residual_mask]

    def keys(frame: pd.DataFrame, columns: tuple[str, ...]) -> set[object]:
        if not set(columns).issubset(frame.columns):
            return set()
        if len(columns) == 1:
            return set(frame[columns[0]].astype(str))
        return set(
            zip(
                *[
                    frame[column].astype(str).to_numpy()
                    for column in columns
                ]
            )
        )

    reference_drugs = keys(reference, ("dr_id",))
    evaluated_drugs = keys(evaluated, ("dr_id",))
    reference_targets = keys(reference, ("pr_id",))
    evaluated_targets = keys(evaluated, ("pr_id",))
    reference_pairs = keys(reference, ("dr_id", "pr_id"))
    evaluated_pairs = keys(evaluated, ("dr_id", "pr_id"))
    return {
        **metadata,
        "reference_role": "fixed_validation_policy_residual",
        "evaluation_role": "held_out_test_policy_residual",
        "reference_n": int(len(reference)),
        "evaluation_n": int(len(evaluated)),
        "reference_outcomes_used_for_support": False,
        "evaluation_batch_appended_to_reference": False,
        "reference_frozen_before_test_scoring": True,
        "exact_pair_support_enabled_in_core": False,
        "overlap_drug_count": int(
            len(reference_drugs.intersection(evaluated_drugs))
        ),
        "overlap_target_count": int(
            len(reference_targets.intersection(evaluated_targets))
        ),
        "overlap_pair_count": int(
            len(reference_pairs.intersection(evaluated_pairs))
        ),
    }


def _component_increment_rows(
    candidates: pd.DataFrame,
    *,
    split: str,
) -> list[dict[str, object]]:
    if candidates.empty:
        return []
    comparisons = list(CORE_INCREMENT_PATH)
    diagnostic_profiles = candidates.loc[
        ~candidates["primary_candidate"].astype(bool),
        "profile",
    ].astype(str).unique()
    for profile in diagnostic_profiles:
        baseline, component = DIAGNOSTIC_BASELINES.get(
            profile,
            ("uncertainty_prior", profile),
        )
        comparisons.append((baseline, profile, component))
    metric_columns = list(METRICS)
    if "validation_fold_mean_auprc" in candidates:
        metric_columns.insert(0, "validation_fold_mean_auprc")
    if split == "test":
        metric_columns.extend(
            f"deduplicated_{metric}"
            for metric in METRICS
            if f"deduplicated_{metric}" in candidates
        )

    rows = []
    for baseline_name, augmented_name, component in comparisons:
        baseline = candidates.loc[
            candidates["profile"].astype(str).eq(baseline_name)
        ]
        augmented = candidates.loc[
            candidates["profile"].astype(str).eq(augmented_name)
        ]
        if baseline.empty or augmented.empty:
            continue
        baseline_row = baseline.iloc[0]
        augmented_row = augmented.iloc[0]
        row = {
            **{
                key: augmented_row[key]
                for key in POLICY_KEYS
                if key in augmented_row
            },
            "split": split,
            "added_component": component,
            "baseline_profile": baseline_name,
            "augmented_profile": augmented_name,
            "baseline_model_status": baseline_row.get("model_status", ""),
            "augmented_model_status": augmented_row.get("model_status", ""),
        }
        for metric in metric_columns:
            baseline_value = pd.to_numeric(
                pd.Series([baseline_row.get(metric, np.nan)]),
                errors="coerce",
            ).iloc[0]
            augmented_value = pd.to_numeric(
                pd.Series([augmented_row.get(metric, np.nan)]),
                errors="coerce",
            ).iloc[0]
            row[f"baseline_{metric}"] = baseline_value
            row[f"augmented_{metric}"] = augmented_value
            row[f"delta_{metric}"] = augmented_value - baseline_value
        rows.append(row)
    return rows


def _stratified_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    work = frame[["model", "dataset", value_column]].copy()
    work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
    work = work.dropna(subset=[value_column])
    if work.empty:
        return np.nan, np.nan
    strata = [
        group[value_column].to_numpy(dtype=float)
        for _, group in work.groupby(["model", "dataset"], dropna=False)
    ]
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(resamples), dtype=float)
    for index in range(int(resamples)):
        selected_strata = rng.choice(
            len(strata),
            size=len(strata),
            replace=True,
        )
        means = []
        for stratum_index in selected_strata:
            stratum = strata[int(stratum_index)]
            means.append(
                float(
                    np.mean(
                        rng.choice(
                            stratum,
                            size=len(stratum),
                            replace=True,
                        )
                    )
                )
            )
        values[index] = float(np.mean(means))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def _summarize_selected(
    selected: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame()
    value_columns = [
        "delta_error_detection_auprc_vs_uncertainty",
        "delta_review_error_rate_vs_uncertainty",
        "delta_retained_accuracy_gain_vs_residual_vs_uncertainty",
    ]
    cluster_keys = [
        "model",
        "dataset",
        "test_run",
        "group_axis",
        "confidence_source",
    ]
    clusters = (
        selected.groupby(cluster_keys, dropna=False)[value_columns]
        .mean()
        .reset_index()
    )
    rows = []
    for group_axis, group in clusters.groupby("group_axis", dropna=False):
        for offset, value_column in enumerate(value_columns):
            values = pd.to_numeric(group[value_column], errors="coerce").dropna()
            if values.empty:
                continue
            stratum_means = (
                group.groupby(["model", "dataset"], dropna=False)[value_column]
                .mean()
                .dropna()
            )
            lo, hi = _stratified_bootstrap(
                group,
                value_column,
                resamples=resamples,
                seed=seed + offset,
            )
            rows.append(
                {
                    "group_axis": group_axis,
                    "measure": value_column,
                    "num_independent_runs": int(len(values)),
                    "num_model_dataset_strata": int(len(stratum_means)),
                    "mean": float(stratum_means.mean()),
                    "run_win_rate": float((values > 0.0).mean()),
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    return clusters, pd.DataFrame(rows)


def _summarize_increments(
    increments: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if increments.empty:
        return pd.DataFrame(), pd.DataFrame()
    value_columns = [
        column
        for column in (
            "delta_validation_fold_mean_auprc",
            "delta_error_detection_auprc",
            "delta_review_lift",
            "delta_retained_accuracy_gain_vs_residual",
        )
        if column in increments
    ]
    cluster_keys = [
        "model",
        "dataset",
        "test_run",
        "group_axis",
        "confidence_source",
        "split",
        "added_component",
    ]
    clusters = (
        increments.groupby(cluster_keys, dropna=False)[value_columns]
        .mean()
        .reset_index()
    )
    rows = []
    grouped = clusters.groupby(
        ["split", "group_axis", "added_component"],
        dropna=False,
    )
    for group_index, ((split, group_axis, component), group) in enumerate(grouped):
        for metric_index, value_column in enumerate(value_columns):
            values = pd.to_numeric(group[value_column], errors="coerce").dropna()
            if values.empty:
                continue
            stratum_means = (
                group.groupby(["model", "dataset"], dropna=False)[value_column]
                .mean()
                .dropna()
            )
            lo, hi = _stratified_bootstrap(
                group,
                value_column,
                resamples=resamples,
                seed=seed + group_index * len(value_columns) + metric_index,
            )
            rows.append(
                {
                    "split": split,
                    "group_axis": group_axis,
                    "added_component": component,
                    "measure": value_column,
                    "num_independent_runs": int(len(values)),
                    "num_model_dataset_strata": int(len(stratum_means)),
                    "mean": float(stratum_means.mean()),
                    "run_win_rate": float((values > 0.0).mean()),
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    return clusters, pd.DataFrame(rows)


def _deduplication_sensitivity_inputs(
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    human = selected.loc[
        selected["dataset"].astype(str).str.lower().eq("human")
    ].copy()
    adjusted = selected.copy()
    for metric in (
        "error_detection_auprc",
        "review_error_rate",
        "retained_accuracy_gain_vs_residual",
    ):
        standard = f"delta_{metric}_vs_uncertainty"
        deduplicated = f"delta_deduplicated_{metric}_vs_uncertainty"
        if deduplicated not in selected:
            continue
        human[standard] = pd.to_numeric(
            human[deduplicated],
            errors="coerce",
        )
        use_deduplicated = adjusted["dataset"].astype(str).str.lower().eq(
            "human"
        )
        adjusted.loc[use_deduplicated, standard] = pd.to_numeric(
            adjusted.loc[use_deduplicated, deduplicated],
            errors="coerce",
        )
    return human, adjusted


def _diagnostic_strata(
    increment_clusters: pd.DataFrame,
    component: str,
) -> pd.DataFrame:
    work = increment_clusters.loc[
        increment_clusters["added_component"].astype(str).eq(
            str(component)
        )
    ].copy()
    if work.empty:
        return pd.DataFrame()
    value_columns = [
        column
        for column in (
            "delta_validation_fold_mean_auprc",
            "delta_error_detection_auprc",
            "delta_review_lift",
            "delta_retained_accuracy_gain_vs_residual",
        )
        if column in work
    ]
    rows = []
    grouped = work.groupby(
        ["split", "group_axis", "model", "dataset"],
        dropna=False,
    )
    for keys, group in grouped:
        split, group_axis, model, dataset = keys
        for measure in value_columns:
            values = pd.to_numeric(group[measure], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "split": split,
                    "group_axis": group_axis,
                    "model": model,
                    "dataset": dataset,
                    "measure": measure,
                    "num_independent_runs": int(len(values)),
                    "mean": float(values.mean()),
                    "run_win_rate": float((values > 0.0).mean()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def _target_sequence_strata(increment_clusters: pd.DataFrame) -> pd.DataFrame:
    return _diagnostic_strata(
        increment_clusters,
        "target_sequence_support",
    )


def _diagnostic_leave_one_out(
    increments: pd.DataFrame,
    *,
    component: str,
    field: str,
    values: list[str],
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    target_only = increments.loc[
        increments["added_component"].astype(str).eq(
            str(component)
        )
    ]
    if target_only.empty or field not in target_only:
        return pd.DataFrame()
    for offset, excluded in enumerate(values):
        subset = target_only.loc[
            ~target_only[field].astype(str).eq(str(excluded))
        ]
        if subset.empty:
            continue
        _, summary = _summarize_increments(
            subset,
            resamples=resamples,
            seed=seed + offset * 100,
        )
        if summary.empty or "added_component" not in summary:
            continue
        selected = summary.loc[
            summary["added_component"].astype(str).eq(
                str(component)
            )
        ].copy()
        selected.insert(0, f"excluded_{field}", excluded)
        rows.append(selected)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _target_sequence_leave_one_dataset_out(
    increments: pd.DataFrame,
    *,
    datasets: list[str],
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    return _diagnostic_leave_one_out(
        increments,
        component="target_sequence_support",
        field="dataset",
        values=datasets,
        resamples=resamples,
        seed=seed,
    )


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    if args.summary_only:
        selected_test = pd.read_csv(out_dir / "selected_test_runs.csv")
        increments = pd.read_csv(out_dir / "component_increment_runs.csv")
        clusters, summary = _summarize_selected(
            selected_test,
            resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
        )
        increment_clusters, increment_summary = _summarize_increments(
            increments,
            resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
        )
        write_csv(
            clusters,
            out_dir / "selected_cluster_runs.csv",
            args.precision,
        )
        write_csv(summary, out_dir / "selected_summary.csv", args.precision)
        write_csv(
            increment_clusters,
            out_dir / "component_increment_cluster_runs.csv",
            args.precision,
        )
        write_csv(
            increment_summary,
            out_dir / "component_increment_summary.csv",
            args.precision,
        )
        human_input, adjusted_input = _deduplication_sensitivity_inputs(
            selected_test
        )
        human_clusters, human_summary = _summarize_selected(
            human_input,
            resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed + 10_000,
        )
        adjusted_clusters, adjusted_summary = _summarize_selected(
            adjusted_input,
            resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed + 20_000,
        )
        write_csv(
            human_clusters,
            out_dir / "human_deduplicated_selected_cluster_runs.csv",
            args.precision,
        )
        write_csv(
            human_summary,
            out_dir / "human_deduplicated_selected_summary.csv",
            args.precision,
        )
        write_csv(
            adjusted_clusters,
            out_dir
            / "human_deduplication_adjusted_selected_cluster_runs.csv",
            args.precision,
        )
        write_csv(
            adjusted_summary,
            out_dir / "human_deduplication_adjusted_selected_summary.csv",
            args.precision,
        )
        write_csv(
            _target_sequence_strata(increment_clusters),
            out_dir / "target_sequence_support_strata.csv",
            args.precision,
        )
        write_csv(
            _target_sequence_leave_one_dataset_out(
                increments,
                datasets=list(args.datasets),
                resamples=args.bootstrap_resamples,
                seed=args.bootstrap_seed + 30_000,
            ),
            out_dir / "target_sequence_support_leave_one_dataset_out.csv",
            args.precision,
        )
        write_csv(
            _diagnostic_strata(
                increment_clusters,
                "general_target_joint_support",
            ),
            out_dir / "general_target_joint_support_strata.csv",
            args.precision,
        )
        write_csv(
            _diagnostic_leave_one_out(
                increments,
                component="general_target_joint_support",
                field="dataset",
                values=list(args.datasets),
                resamples=args.bootstrap_resamples,
                seed=args.bootstrap_seed + 40_000,
            ),
            out_dir
            / "general_target_joint_support_leave_one_dataset_out.csv",
            args.precision,
        )
        models = sorted(increments.get("model", pd.Series(dtype=str)).dropna().astype(str).unique())
        write_csv(
            _diagnostic_leave_one_out(
                increments,
                component="general_target_joint_support",
                field="model",
                values=models,
                resamples=args.bootstrap_resamples,
                seed=args.bootstrap_seed + 50_000,
            ),
            out_dir / "general_target_joint_support_leave_one_model_out.csv",
            args.precision,
        )
        return

    pairs, missing = _discover_pairs(args)
    write_csv(missing, out_dir / "missing_inputs.csv", args.precision)
    write_csv(
        profile_manifest(),
        out_dir / "profile_manifest.csv",
        args.precision,
    )

    profiles = list(PRIMARY_PROFILES) + list(EVIDENCE_DIAGNOSTIC_PROFILES)

    coverage_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    selected_test_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    increment_rows: list[dict[str, object]] = []
    domain_support_rows: list[dict[str, object]] = []
    audit_fit_reference_rows: list[dict[str, object]] = []
    human_pair_sensitivity_rows: list[dict[str, object]] = []
    capability_rows: list[dict[str, object]] = []
    source_provenance_rows: dict[tuple[str, str], dict[str, object]] = {}

    for pair in pairs:
        print(
            f"[PAUSE] {pair['model']}/{pair['dataset']}/{pair['test_run']}",
            flush=True,
        )
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
        capability_rows.append(_capability_row(pair, "validation", validation))
        capability_rows.append(_capability_row(pair, "test", test))
        source_key = (str(pair["dataset"]), "cluster")
        if source_key not in source_provenance_rows:
            source_provenance_rows[source_key] = source_support_provenance(
                dataset=str(pair["dataset"]),
                split="cluster",
                dataset_root=args.dataset_root,
            )
        test_calibrated, test_calibration_status = fit_calibrator(validation, test)

        for group_axis in args.group_axes:
            validation_calibration = cross_fitted_calibration(
                validation,
                folds=args.cv_folds,
                seed=args.cv_seed,
                group_axis=group_axis,
                strict_group=args.strict_group_cv,
            )
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
                    calibration_rows.extend(
                        _calibration_rows(
                            metadata,
                            "validation",
                            validation,
                            validation_calibration.probability,
                            validation_policy,
                        )
                    )
                    domain_support_rows.extend(
                        _domain_support_rows(
                            metadata,
                            test,
                            test_policy["residual_mask"],
                            validation.loc[validation_policy["residual_mask"]],
                        )
                    )
                    audit_fit_reference_rows.append(
                        _audit_fit_reference_row(
                            metadata,
                            validation,
                            validation_policy["residual_mask"],
                            test,
                            test_policy["residual_mask"],
                        )
                    )
                    calibration_rows.extend(
                        _calibration_rows(
                            metadata,
                            "test",
                            test,
                            test_calibrated,
                            test_policy,
                        )
                    )

                    local_validation_rows = []
                    local_test_rows = []
                    for profile in profiles:
                        validation_result = cross_fitted_profile(
                            validation,
                            validation_policy["residual_mask"],
                            validation_calibration.probability,
                            profile,
                            folds=args.cv_folds,
                            seed=args.cv_seed,
                            group_axis=group_axis,
                            strict_group=args.strict_group_cv,
                            min_fit_n=args.min_fit_n,
                            min_feature_n=args.min_feature_n,
                            ranking_weight=args.ranking_weight,
                            max_pairs=args.max_ranking_pairs,
                        )
                        validation_metrics = evaluate_ranking(
                            validation,
                            validation_policy["residual_mask"],
                            validation_result.scores,
                            review_fraction=args.review_fraction,
                            min_universe_n=args.min_universe_n,
                            deferred_mask=validation_policy["deferred_mask"],
                        )
                        validation_row = _candidate_row(
                            metadata,
                            profile,
                            validation_result,
                            validation_metrics,
                        )
                        local_validation_rows.append(validation_row)
                        validation_rows.append(validation_row)

                        test_result = fit_profile_for_test(
                            validation,
                            validation_policy["residual_mask"],
                            validation_calibration.probability,
                            test,
                            test_policy["residual_mask"],
                            test_calibrated,
                            profile,
                            seed=args.cv_seed,
                            min_fit_n=args.min_fit_n,
                            min_feature_n=args.min_feature_n,
                            ranking_weight=args.ranking_weight,
                            max_pairs=args.max_ranking_pairs,
                        )
                        test_metrics = evaluate_ranking(
                            test,
                            test_policy["residual_mask"],
                            test_result.scores,
                            review_fraction=args.review_fraction,
                            min_universe_n=args.min_universe_n,
                            deferred_mask=test_policy["deferred_mask"],
                        )
                        deduplicated_metrics: dict[str, float | int] = {}
                        if str(pair["dataset"]).lower() == "human":
                            deduplicated_metrics, pair_diagnostics = (
                                _pair_deduplicated_metrics(
                                    validation,
                                    test,
                                    test_policy["residual_mask"],
                                    test_policy["deferred_mask"],
                                    test_result.scores,
                                    review_fraction=args.review_fraction,
                                    min_universe_n=args.min_universe_n,
                                )
                            )
                            if profile.name == "uncertainty":
                                human_pair_sensitivity_rows.append(
                                    {
                                        **metadata,
                                        **pair_diagnostics,
                                    }
                                )
                        test_row = {
                            **metadata,
                            "profile": profile.name,
                            "primary_candidate": profile.primary,
                            "complexity": profile.complexity,
                            "model_status": test_result.status,
                            "used_features": "|".join(test_result.used_features),
                            **test_metrics,
                            **{
                                f"deduplicated_{key}": value
                                for key, value in deduplicated_metrics.items()
                            },
                        }
                        local_test_rows.append(test_row)
                        test_rows.append(test_row)

                    local_validation = pd.DataFrame(local_validation_rows)
                    local_test = pd.DataFrame(local_test_rows)
                    increment_rows.extend(
                        _component_increment_rows(
                            local_validation,
                            split="validation",
                        )
                    )
                    increment_rows.extend(
                        _component_increment_rows(
                            local_test,
                            split="test",
                        )
                    )
                    selected = select_one_standard_error(
                        local_validation,
                        min_validation_gain=args.min_validation_gain,
                    )
                    selection_row = {
                        **metadata,
                        **{
                            key: value
                            for key, value in selected.items()
                            if key not in metadata
                        },
                    }
                    selection_rows.append(selection_row)

                    selected_test = local_test.loc[
                        local_test["profile"].astype(str)
                        == str(selected["selected_profile"])
                    ]
                    uncertainty_test = local_test.loc[
                        local_test["profile"].astype(str) == "uncertainty"
                    ]
                    if selected_test.empty or uncertainty_test.empty:
                        continue
                    selected_test = selected_test.iloc[0].to_dict()
                    uncertainty_test = uncertainty_test.iloc[0]
                    selected_test["selected_profile"] = selected["selected_profile"]
                    selected_test["selection_fallback"] = selected[
                        "selection_fallback"
                    ]
                    selected_test["selection_reason"] = selected[
                        "selection_reason"
                    ]
                    selected_test["best_profile"] = selected.get(
                        "best_profile",
                        selected["selected_profile"],
                    )
                    for metric in METRICS:
                        selected_test[f"uncertainty_{metric}"] = uncertainty_test[
                            metric
                        ]
                        selected_test[f"delta_{metric}_vs_uncertainty"] = (
                            selected_test[metric] - uncertainty_test[metric]
                        )
                        deduplicated_metric = f"deduplicated_{metric}"
                        if deduplicated_metric in selected_test:
                            selected_test[
                                f"uncertainty_{deduplicated_metric}"
                            ] = uncertainty_test.get(
                                deduplicated_metric,
                                np.nan,
                            )
                            selected_test[
                                f"delta_{deduplicated_metric}_vs_uncertainty"
                            ] = (
                                selected_test[deduplicated_metric]
                                - uncertainty_test.get(
                                    deduplicated_metric,
                                    np.nan,
                                )
                            )
                    selected_test_rows.append(selected_test)

    coverage = pd.DataFrame(coverage_rows)
    validation_candidates = pd.DataFrame(validation_rows)
    test_candidates = pd.DataFrame(test_rows)
    selections = pd.DataFrame(selection_rows)
    selected_test = pd.DataFrame(selected_test_rows)
    calibration = pd.DataFrame(calibration_rows)
    increments = pd.DataFrame(increment_rows)
    domain_support = pd.DataFrame(domain_support_rows)
    audit_fit_references = pd.DataFrame(audit_fit_reference_rows)
    source_provenance = pd.DataFrame(source_provenance_rows.values())
    human_pair_sensitivity = pd.DataFrame(human_pair_sensitivity_rows)
    capabilities = pd.DataFrame(capability_rows)
    clusters, summary = _summarize_selected(
        selected_test,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    increment_clusters, increment_summary = _summarize_increments(
        increments,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    (
        human_deduplicated_summary_input,
        human_deduplication_adjusted_input,
    ) = _deduplication_sensitivity_inputs(selected_test)
    human_deduplicated_selected = selected_test.loc[
        selected_test["dataset"].astype(str).str.lower().eq("human")
    ].copy()
    (
        human_deduplicated_clusters,
        human_deduplicated_summary,
    ) = _summarize_selected(
        human_deduplicated_summary_input,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed + 10_000,
    )
    (
        human_deduplication_adjusted_clusters,
        human_deduplication_adjusted_summary,
    ) = _summarize_selected(
        human_deduplication_adjusted_input,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed + 20_000,
    )
    target_sequence_strata = _target_sequence_strata(increment_clusters)
    target_sequence_leave_one_out = _target_sequence_leave_one_dataset_out(
        increments,
        datasets=list(args.datasets),
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed + 30_000,
    )
    general_target_joint_strata = _diagnostic_strata(
        increment_clusters,
        "general_target_joint_support",
    )
    general_target_joint_leave_one_dataset_out = _diagnostic_leave_one_out(
        increments,
        component="general_target_joint_support",
        field="dataset",
        values=list(args.datasets),
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed + 40_000,
    )
    general_target_joint_leave_one_model_out = _diagnostic_leave_one_out(
        increments,
        component="general_target_joint_support",
        field="model",
        values=sorted(capabilities["model"].dropna().astype(str).unique()),
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed + 50_000,
    )

    write_csv(coverage, out_dir / "policy_coverage.csv", args.precision)
    write_csv(
        capabilities,
        out_dir / "capability_manifest.csv",
        args.precision,
    )
    write_csv(
        calibration,
        out_dir / "calibration_diagnostics.csv",
        args.precision,
    )
    write_csv(
        domain_support,
        out_dir / "domain_support_diagnostics.csv",
        args.precision,
    )
    write_csv(
        audit_fit_references,
        out_dir / "audit_fit_reference_diagnostics.csv",
        args.precision,
    )
    write_csv(
        source_provenance,
        out_dir / "source_support_provenance.csv",
        args.precision,
    )
    write_csv(
        human_pair_sensitivity,
        out_dir / "human_pair_deduplication_diagnostics.csv",
        args.precision,
    )
    write_csv(
        validation_candidates,
        out_dir / "candidate_validation_runs.csv",
        args.precision,
    )
    write_csv(test_candidates, out_dir / "candidate_test_runs.csv", args.precision)
    write_csv(selections, out_dir / "selection_choices.csv", args.precision)
    write_csv(
        increments,
        out_dir / "component_increment_runs.csv",
        args.precision,
    )
    write_csv(
        increment_clusters,
        out_dir / "component_increment_cluster_runs.csv",
        args.precision,
    )
    write_csv(
        increment_summary,
        out_dir / "component_increment_summary.csv",
        args.precision,
    )
    write_csv(selected_test, out_dir / "selected_test_runs.csv", args.precision)
    write_csv(clusters, out_dir / "selected_cluster_runs.csv", args.precision)
    write_csv(summary, out_dir / "selected_summary.csv", args.precision)
    write_csv(
        human_deduplicated_selected,
        out_dir / "human_deduplicated_selected_test_runs.csv",
        args.precision,
    )
    write_csv(
        human_deduplicated_clusters,
        out_dir / "human_deduplicated_selected_cluster_runs.csv",
        args.precision,
    )
    write_csv(
        human_deduplicated_summary,
        out_dir / "human_deduplicated_selected_summary.csv",
        args.precision,
    )
    write_csv(
        human_deduplication_adjusted_clusters,
        out_dir / "human_deduplication_adjusted_selected_cluster_runs.csv",
        args.precision,
    )
    write_csv(
        human_deduplication_adjusted_summary,
        out_dir / "human_deduplication_adjusted_selected_summary.csv",
        args.precision,
    )
    write_csv(
        target_sequence_strata,
        out_dir / "target_sequence_support_strata.csv",
        args.precision,
    )
    write_csv(
        target_sequence_leave_one_out,
        out_dir / "target_sequence_support_leave_one_dataset_out.csv",
        args.precision,
    )
    write_csv(
        general_target_joint_strata,
        out_dir / "general_target_joint_support_strata.csv",
        args.precision,
    )
    write_csv(
        general_target_joint_leave_one_dataset_out,
        out_dir / "general_target_joint_support_leave_one_dataset_out.csv",
        args.precision,
    )
    write_csv(
        general_target_joint_leave_one_model_out,
        out_dir / "general_target_joint_support_leave_one_model_out.csv",
        args.precision,
    )


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
