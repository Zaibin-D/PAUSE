"""Component definitions for the policy-aligned PAUSE residual audit.

The feature blocks are intentionally directional and role-specific:

- uncertainty defines the first-stage safety policy;
- prior contradiction measures evidence against a positive base call;
- empirical support measures how well the audited pair is represented in the
  source-training and fold-fit domains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    features: tuple[str, ...]
    complexity: int
    primary: bool
    description: str
    required_any: tuple[str, ...] = ()
    required_blocks: tuple[tuple[str, ...], ...] = ()


UNCERTAINTY_FEATURES = ("calibrated_error_risk",)
PRIOR_FEATURES = (
    "prior_contradiction_logit",
    "prior_support_logit",
    "prior_negative_evidence",
    "prior_contradiction_probability",
    "prior_support_probability",
)
FIT_ENTITY_SUPPORT_FEATURES = (
    "log_drug_fit_count",
    "log_target_fit_count",
    "fit_entity_support_min",
    "fit_entity_support_imbalance",
    "fit_joint_novelty",
    "fit_one_sided_novelty",
)
LEGACY_FIT_SUPPORT_FEATURES = FIT_ENTITY_SUPPORT_FEATURES + (
    "log_pair_fit_count",
)
SOURCE_COUNT_FEATURES = (
    "log_source_drug_count",
    "log_source_target_count",
    "log_source_pair_count",
    "log_source_drug_cluster_count",
    "log_source_target_cluster_count",
    "log_source_cluster_pair_count",
    "source_entity_support_min",
    "source_cluster_support_min",
    "source_entity_support_imbalance",
    "source_cluster_support_imbalance",
    "source_joint_novelty",
    "source_one_sided_novelty",
)
LEGACY_SOURCE_DISTANCE_FEATURES = (
    "drug_nearest_train_distance",
    "target_nearest_train_distance",
    "drug_train_knn_distance",
    "target_train_knn_distance",
    "drug_embedding_mahalanobis",
    "target_embedding_mahalanobis",
    "domain_shift_score",
    "domain_density_distance",
)
NATIVE_SOURCE_FEATURES = (
    "drug_morgan_nearest_train_distance",
    "drug_morgan_knn_distance",
    "target_esm_nearest_train_distance",
    "target_esm_knn_distance",
    "native_domain_shift_score",
    "native_domain_density_distance",
    "native_support_imbalance",
    "native_density_imbalance",
)
TARGET_SEQUENCE_SUPPORT_FEATURES = (
    "target_mmseqs_nearest_train_distance",
    "target_mmseqs_family_unseen",
    "target_mmseqs_family_sparsity",
)
GENERAL_TARGET_JOINT_SUPPORT_FEATURES = (
    "target_direct_nearest_train_distance",
    "target_direct_knn_distance",
    "joint_nearest_train_distance",
)
LEGACY_SOURCE_DOMAIN_FEATURES = (
    SOURCE_COUNT_FEATURES + LEGACY_SOURCE_DISTANCE_FEATURES
)
NATIVE_SUPPORT_FEATURES = NATIVE_SOURCE_FEATURES
LEGACY_SUPPORT_FEATURES = (
    LEGACY_FIT_SUPPORT_FEATURES + LEGACY_SOURCE_DOMAIN_FEATURES
)
NATIVE_EMPIRICAL_SUPPORT_FEATURES = (
    FIT_ENTITY_SUPPORT_FEATURES + NATIVE_SUPPORT_FEATURES
)
SOURCE_DOMAIN_FEATURES = (
    SOURCE_COUNT_FEATURES
    + LEGACY_SOURCE_DISTANCE_FEATURES
    + NATIVE_SOURCE_FEATURES
)
COMBINED_SUPPORT_FEATURES = (
    LEGACY_SUPPORT_FEATURES + NATIVE_SOURCE_FEATURES
)
SUPPORT_FEATURES = NATIVE_EMPIRICAL_SUPPORT_FEATURES


PRIMARY_PROFILES: tuple[ProfileSpec, ...] = (
    ProfileSpec(
        "uncertainty",
        UNCERTAINTY_FEATURES,
        0,
        True,
        "Calibrated risk that the frozen base prediction is wrong.",
    ),
    ProfileSpec(
        "prior",
        PRIOR_FEATURES,
        1,
        True,
        "Calibrated prior evidence against a positive base prediction.",
        required_blocks=(PRIOR_FEATURES,),
    ),
    ProfileSpec(
        "uncertainty_prior",
        UNCERTAINTY_FEATURES + PRIOR_FEATURES,
        2,
        True,
        "Residual error risk plus calibrated prior evidence.",
        required_blocks=(PRIOR_FEATURES,),
    ),
    ProfileSpec(
        "uncertainty_support",
        UNCERTAINTY_FEATURES + SUPPORT_FEATURES,
        2,
        True,
        "Residual error risk plus label-free empirical domain support.",
        SUPPORT_FEATURES,
        (SUPPORT_FEATURES,),
    ),
    ProfileSpec(
        "uncertainty_prior_support",
        UNCERTAINTY_FEATURES + PRIOR_FEATURES + SUPPORT_FEATURES,
        3,
        True,
        "Core prior audit augmented with empirical domain support.",
        SUPPORT_FEATURES,
        (PRIOR_FEATURES, SUPPORT_FEATURES),
    ),
)


EVIDENCE_DIAGNOSTIC_PROFILES: tuple[ProfileSpec, ...] = (
    ProfileSpec(
        "fit_support",
        UNCERTAINTY_FEATURES + FIT_ENTITY_SUPPORT_FEATURES,
        1,
        False,
        "Diagnostic U+E profile using fold-fit drug/target support.",
        FIT_ENTITY_SUPPORT_FEATURES,
    ),
    ProfileSpec(
        "source_domain_support",
        UNCERTAINTY_FEATURES + LEGACY_SOURCE_DOMAIN_FEATURES,
        1,
        False,
        "Legacy U+E diagnostic using source counts and prior-space distances.",
        LEGACY_SOURCE_DOMAIN_FEATURES,
    ),
    ProfileSpec(
        "native_source_domain_support",
        UNCERTAINTY_FEATURES + NATIVE_SOURCE_FEATURES,
        1,
        False,
        "Native U+E diagnostic using Morgan/Tanimoto and ESM/cosine support.",
        NATIVE_SOURCE_FEATURES,
    ),
    ProfileSpec(
        "target_sequence_support",
        UNCERTAINTY_FEATURES
        + NATIVE_EMPIRICAL_SUPPORT_FEATURES
        + TARGET_SEQUENCE_SUPPORT_FEATURES,
        3,
        False,
        "Fixed target-E diagnostic adding MMseqs2 identity/family support.",
        TARGET_SEQUENCE_SUPPORT_FEATURES,
        (
            NATIVE_EMPIRICAL_SUPPORT_FEATURES,
            TARGET_SEQUENCE_SUPPORT_FEATURES,
        ),
    ),
    ProfileSpec(
        "general_target_joint_support",
        UNCERTAINTY_FEATURES
        + NATIVE_EMPIRICAL_SUPPORT_FEATURES
        + GENERAL_TARGET_JOINT_SUPPORT_FEATURES,
        3,
        False,
        (
            "Model-independent target-E diagnostic using direct source-target "
            "neighbours and observed source-train interaction support."
        ),
        GENERAL_TARGET_JOINT_SUPPORT_FEATURES,
        (
            NATIVE_EMPIRICAL_SUPPORT_FEATURES,
            GENERAL_TARGET_JOINT_SUPPORT_FEATURES,
        ),
    ),
    ProfileSpec(
        "legacy_support",
        UNCERTAINTY_FEATURES + LEGACY_SUPPORT_FEATURES,
        2,
        False,
        "Original E definition: fold-fit plus prior-space source support.",
        LEGACY_SOURCE_DOMAIN_FEATURES,
    ),
    ProfileSpec(
        "combined_support",
        UNCERTAINTY_FEATURES + COMBINED_SUPPORT_FEATURES,
        3,
        False,
        "Diagnostic union of the legacy and native E definitions.",
        NATIVE_SOURCE_FEATURES,
    ),
)


OPTIONAL_DISTANCE_COLUMNS = (
    "drug_nearest_train_distance",
    "target_nearest_train_distance",
    "drug_train_knn_distance",
    "target_train_knn_distance",
    "drug_embedding_mahalanobis",
    "target_embedding_mahalanobis",
    "domain_shift_score",
    "domain_density_distance",
    "drug_morgan_nearest_train_distance",
    "drug_morgan_knn_distance",
    "target_esm_nearest_train_distance",
    "target_esm_knn_distance",
    "native_domain_shift_score",
    "native_domain_density_distance",
    "native_support_imbalance",
    "native_density_imbalance",
    "target_mmseqs_nearest_train_distance",
    "target_mmseqs_family_unseen",
    "target_mmseqs_family_sparsity",
    "target_direct_nearest_train_distance",
    "target_direct_knn_distance",
    "joint_nearest_train_distance",
)


def profile_manifest() -> pd.DataFrame:
    profiles = list(PRIMARY_PROFILES) + list(EVIDENCE_DIAGNOSTIC_PROFILES)
    return pd.DataFrame(
        [
            {
                "profile": profile.name,
                "features": "|".join(profile.features),
                "complexity": profile.complexity,
                "primary_candidate": profile.primary,
                "description": profile.description,
                "required_any": "|".join(profile.required_any),
                "required_blocks": ";".join(
                    "|".join(block) for block in profile.required_blocks
                ),
            }
            for profile in profiles
        ]
    )


def profile_requirements_met(
    selected_features: Iterable[str],
    profile: ProfileSpec,
) -> bool:
    """Return whether every declared evidence block is represented."""

    selected = set(selected_features)
    blocks = profile.required_blocks
    if not blocks and profile.required_any:
        blocks = (profile.required_any,)
    return all(any(feature in selected for feature in block) for block in blocks)


def audit_capabilities(table: pd.DataFrame) -> dict[str, object]:
    """Describe label-free PAUSE inputs and the supported fallback modes."""

    def has_finite(columns: Iterable[str]) -> bool:
        return any(
            column in table and _numeric(table, column).notna().any()
            for column in columns
        )

    uncertainty = has_finite(("p_base", "s_base"))
    prior = has_finite(("p_prior", "s_prior"))
    entity_reference = {"dr_id", "pr_id"}.issubset(table.columns)
    native_support = has_finite(NATIVE_SOURCE_FEATURES)
    empirical_support = bool(entity_reference or native_support)
    direct_target = has_finite(
        (
            "target_direct_nearest_train_distance",
            "target_direct_knn_distance",
        )
    )
    joint_support = has_finite(("joint_nearest_train_distance",))

    modes = []
    if uncertainty:
        modes.append("U")
        if prior:
            modes.append("U+P")
        if empirical_support:
            modes.append("U+E")
        if prior and empirical_support:
            modes.append("U+P+E")
    return {
        "uncertainty_available": uncertainty,
        "prior_available": prior,
        "empirical_support_available": empirical_support,
        "native_support_available": native_support,
        "direct_target_support_available": direct_target,
        "joint_support_available": joint_support,
        "available_modes": "|".join(modes),
        "maximum_mode": modes[-1] if modes else "unavailable",
    }


def _numeric(table: pd.DataFrame, name: str) -> pd.Series:
    if name not in table:
        return pd.Series(np.nan, index=table.index, dtype=float)
    return pd.to_numeric(table[name], errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )


def _sigmoid_series(values: pd.Series) -> pd.Series:
    clipped = values.clip(-40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _evidence_probability(
    table: pd.DataFrame,
    evidence_probabilities: dict[str, pd.Series | np.ndarray] | None,
    name: str,
    logit_column: str,
    probability_column: str,
) -> pd.Series:
    if evidence_probabilities and name in evidence_probabilities:
        return pd.Series(
            np.asarray(evidence_probabilities[name], dtype=float),
            index=table.index,
        ).clip(0.0, 1.0)
    probability = _numeric(table, probability_column)
    if probability.notna().any():
        return probability.clip(0.0, 1.0)
    return _sigmoid_series(_numeric(table, logit_column))


def calibrated_error_risk(
    table: pd.DataFrame,
    calibrated_probability: pd.Series | np.ndarray,
) -> pd.Series:
    """Return error risk for the base model's frozen predicted class."""

    probability = pd.Series(
        np.asarray(calibrated_probability, dtype=float),
        index=table.index,
    ).clip(0.0, 1.0)
    base_pred = _numeric(table, "base_pred")
    if base_pred.isna().all():
        base_pred = (_numeric(table, "p_base") >= 0.5).astype(float)
    predicted_class_probability = probability.where(
        base_pred.ge(0.5),
        1.0 - probability,
    )
    return (1.0 - predicted_class_probability).clip(0.0, 1.0)


def _support_counts(
    table: pd.DataFrame,
    reference: pd.DataFrame | None,
    columns: str | tuple[str, ...],
) -> tuple[pd.Series, pd.Series, pd.Series]:
    columns = (columns,) if isinstance(columns, str) else tuple(columns)
    if (
        reference is None
        or any(column not in table for column in columns)
        or any(column not in reference for column in columns)
    ):
        missing = pd.Series(np.nan, index=table.index, dtype=float)
        return missing, missing.copy(), missing.copy()
    if len(columns) == 1:
        ref_values = reference[columns[0]].astype(str)
        values = table[columns[0]].astype(str)
    else:
        ref_values = pd.Series(
            list(
                zip(
                    *[
                        reference[column].astype(str).to_numpy()
                        for column in columns
                    ]
                )
            ),
            index=reference.index,
            dtype=object,
        )
        values = pd.Series(
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
    counts = ref_values.value_counts(dropna=False)
    raw_counts = values.map(counts).fillna(0.0).astype(float)
    return (
        (raw_counts > 0.0).astype(float),
        np.log1p(raw_counts),
        raw_counts,
    )


def _source_count_features(out: pd.DataFrame, table: pd.DataFrame) -> None:
    count_columns = (
        ("source_drug_count", "source_drug"),
        ("source_target_count", "source_target"),
        ("source_pair_count", "source_pair"),
        ("source_drug_cluster_count", "source_drug_cluster"),
        ("source_target_cluster_count", "source_target_cluster"),
        ("source_cluster_pair_count", "source_cluster_pair"),
    )
    for source, prefix in count_columns:
        counts = _numeric(table, source).clip(lower=0.0)
        out[f"{prefix}_seen"] = (counts > 0.0).astype(float).where(
            counts.notna()
        )
        out[f"log_{prefix}_count"] = np.log1p(counts)

    source_drug = out["log_source_drug_count"]
    source_target = out["log_source_target_count"]
    source_drug_cluster = out["log_source_drug_cluster_count"]
    source_target_cluster = out["log_source_target_cluster_count"]
    out["source_entity_support_min"] = pd.concat(
        [source_drug, source_target],
        axis=1,
    ).min(axis=1, skipna=False)
    out["source_cluster_support_min"] = pd.concat(
        [source_drug_cluster, source_target_cluster],
        axis=1,
    ).min(axis=1, skipna=False)
    out["source_entity_support_imbalance"] = (
        source_drug - source_target
    ).abs()
    out["source_cluster_support_imbalance"] = (
        source_drug_cluster - source_target_cluster
    ).abs()
    source_drug_seen = out["source_drug_seen"]
    source_target_seen = out["source_target_seen"]
    out["source_joint_novelty"] = (
        (1.0 - source_drug_seen) * (1.0 - source_target_seen)
    )
    out["source_one_sided_novelty"] = (
        source_drug_seen - source_target_seen
    ).abs()


def engineer_features(
    table: pd.DataFrame,
    *,
    calibrated_probability: pd.Series | np.ndarray | None = None,
    support_reference: pd.DataFrame | None = None,
    evidence_probabilities: dict[str, pd.Series | np.ndarray] | None = None,
) -> pd.DataFrame:
    """Create PAUSE component features without using outcome labels."""

    out = pd.DataFrame(index=table.index)
    if calibrated_probability is None:
        calibrated = _numeric(table, "p_base")
    else:
        calibrated = pd.Series(
            np.asarray(calibrated_probability, dtype=float),
            index=table.index,
        )
    out["calibrated_probability"] = calibrated.clip(0.0, 1.0)
    out["calibrated_error_risk"] = calibrated_error_risk(
        table,
        out["calibrated_probability"],
    )

    prior_probability = _evidence_probability(
        table,
        evidence_probabilities,
        "prior",
        "s_prior",
        "p_prior",
    )
    prior_gap = out["calibrated_probability"] - prior_probability
    out["prior_negative_evidence"] = 1.0 - prior_probability
    out["prior_contradiction_probability"] = prior_gap.clip(lower=0.0)
    out["prior_support_probability"] = (-prior_gap).clip(lower=0.0)
    prior_logit_gap = _numeric(table, "s_base") - _numeric(table, "s_prior")
    out["prior_contradiction_logit"] = prior_logit_gap.clip(lower=0.0)
    out["prior_support_logit"] = (-prior_logit_gap).clip(lower=0.0)

    drug_seen, drug_count, _ = _support_counts(
        table,
        support_reference,
        "dr_id",
    )
    target_seen, target_count, _ = _support_counts(
        table,
        support_reference,
        "pr_id",
    )
    pair_seen, pair_count, _ = _support_counts(
        table,
        support_reference,
        ("dr_id", "pr_id"),
    )
    out["drug_seen_in_fit"] = drug_seen
    out["target_seen_in_fit"] = target_seen
    out["pair_seen_in_fit"] = pair_seen
    out["log_drug_fit_count"] = drug_count
    out["log_target_fit_count"] = target_count
    out["log_pair_fit_count"] = pair_count
    out["fit_entity_support_min"] = pd.concat(
        [drug_count, target_count],
        axis=1,
    ).min(axis=1, skipna=False)
    out["fit_entity_support_imbalance"] = (drug_count - target_count).abs()
    out["fit_joint_novelty"] = (1.0 - drug_seen) * (1.0 - target_seen)
    out["fit_one_sided_novelty"] = (drug_seen - target_seen).abs()
    _source_count_features(out, table)

    for column in OPTIONAL_DISTANCE_COLUMNS:
        if column in table:
            out[column] = _numeric(table, column)
    return out
