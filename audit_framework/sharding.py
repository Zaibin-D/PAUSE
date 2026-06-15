"""Resumable model-dataset sharding for the fixed PAUSE audit."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import numpy as np
import pandas as pd

from audit_framework import profile_manifest
from audit_framework.data import write_csv
from audit_framework.scripts.run_audit import (
    CORE_INCREMENT_PATH,
    DIAGNOSTIC_BASELINES,
    POLICY_KEYS,
    _stratified_bootstrap,
)


MODEL_ROOTS = {
    "pace": (
        "PACE",
        "audit_framework/cache/test_audits/pace",
    ),
    "tapb": (
        "TAPB",
        "audit_framework/cache/test_audits/tapb",
    ),
    "drugban": (
        "DrugBAN",
        "audit_framework/cache/test_audits/drugban",
    ),
}

RUN_LEVEL_FILES = (
    "policy_coverage.csv",
    "capability_manifest.csv",
    "calibration_diagnostics.csv",
    "domain_support_diagnostics.csv",
    "audit_fit_reference_diagnostics.csv",
    "source_support_provenance.csv",
    "human_pair_deduplication_diagnostics.csv",
    "candidate_validation_runs.csv",
    "candidate_test_runs.csv",
    "selection_choices.csv",
    "component_increment_runs.csv",
    "selected_test_runs.csv",
    "human_deduplicated_selected_test_runs.csv",
)

TABLE_KEYS = {
    "policy_coverage.csv": POLICY_KEYS + ["split"],
    "capability_manifest.csv": ["model", "dataset", "test_run", "split"],
    "calibration_diagnostics.csv": POLICY_KEYS
    + ["split", "scope", "score_source"],
    "domain_support_diagnostics.csv": POLICY_KEYS
    + ["split", "support_reference", "support_state"],
    "audit_fit_reference_diagnostics.csv": POLICY_KEYS,
    "source_support_provenance.csv": ["dataset", "split"],
    "human_pair_deduplication_diagnostics.csv": POLICY_KEYS,
    "candidate_validation_runs.csv": POLICY_KEYS + ["profile"],
    "candidate_test_runs.csv": POLICY_KEYS + ["profile"],
    "selection_choices.csv": POLICY_KEYS,
    "component_increment_runs.csv": POLICY_KEYS
    + [
        "split",
        "added_component",
        "baseline_profile",
        "augmented_profile",
    ],
    "selected_test_runs.csv": POLICY_KEYS,
    "human_deduplicated_selected_test_runs.csv": POLICY_KEYS,
}


class ShardValidationError(RuntimeError):
    """Raised when a shard cannot be proven complete and unique."""


@dataclass(frozen=True)
class ShardSpec:
    model_key: str
    model_name: str
    dataset: str
    test_root: str
    out_dir: Path


def build_specs(
    *,
    models: Iterable[str],
    datasets: Iterable[str],
    shard_root: str | Path,
) -> list[ShardSpec]:
    root = Path(shard_root)
    specs = []
    for model_key, dataset in product(models, datasets):
        key = str(model_key).lower()
        if key not in MODEL_ROOTS:
            raise ValueError(f"Unknown model shard: {model_key}")
        display_name, test_root = MODEL_ROOTS[key]
        specs.append(
            ShardSpec(
                model_key=key,
                model_name=display_name,
                dataset=str(dataset).lower(),
                test_root=test_root,
                out_dir=root / key / str(dataset).lower(),
            )
        )
    return specs


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes"}
    )


def _normalise_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.12g}"
    return str(value)


def _key_set(frame: pd.DataFrame, columns: list[str]) -> set[tuple[str, ...]]:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ShardValidationError(
            f"Missing key columns {missing} from table with columns "
            f"{list(frame.columns)}"
        )
    return {
        tuple(_normalise_value(value) for value in row)
        for row in frame[columns].itertuples(index=False, name=None)
    }


def _assert_unique(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    label: str,
) -> None:
    if frame.empty:
        return
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ShardValidationError(f"{label} is missing key columns {missing}")
    duplicated = frame.duplicated(columns, keep=False)
    if duplicated.any():
        example = frame.loc[duplicated, columns].head(5).to_dict("records")
        raise ShardValidationError(
            f"{label} has duplicate keys on {columns}: {example}"
        )


def _policy_keys(
    spec: ShardSpec,
    *,
    seeds: Iterable[str],
    group_axes: Iterable[str],
    confidence_source: str,
    confidence_thresholds: Iterable[float],
    reject_fractions: Iterable[float],
    review_fraction: float,
) -> set[tuple[str, ...]]:
    rows = []
    for seed, group_axis, threshold, reject_fraction in product(
        seeds,
        group_axes,
        confidence_thresholds,
        reject_fractions,
    ):
        seed_name = str(seed)
        if not seed_name.startswith("seed_"):
            seed_name = f"seed_{seed_name}"
        rows.append(
            (
                spec.model_name,
                spec.dataset,
                seed_name,
                str(group_axis),
                str(confidence_source),
                float(threshold),
                float(reject_fraction),
                float(review_fraction),
            )
        )
    return {
        tuple(_normalise_value(value) for value in row)
        for row in rows
    }


def _manifest_records(frame: pd.DataFrame) -> list[tuple[str, ...]]:
    columns = [
        "profile",
        "features",
        "complexity",
        "primary_candidate",
        "description",
        "required_any",
        "required_blocks",
    ]
    if list(frame.columns) != columns:
        raise ShardValidationError(
            f"profile_manifest.csv columns differ: {list(frame.columns)}"
        )
    return sorted(
        tuple("" if pd.isna(value) else str(value) for value in row)
        for row in frame[columns].itertuples(index=False, name=None)
    )


def _expected_comparisons(
    manifest: pd.DataFrame,
) -> set[tuple[str, str, str]]:
    comparisons = {
        (component, baseline, augmented)
        for baseline, augmented, component in CORE_INCREMENT_PATH
    }
    diagnostic_profiles = manifest.loc[
        ~_truthy(manifest["primary_candidate"]),
        "profile",
    ].astype(str)
    for profile in diagnostic_profiles:
        baseline, component = DIAGNOSTIC_BASELINES.get(
            profile,
            ("uncertainty_prior", profile),
        )
        comparisons.add((component, baseline, profile))
    return comparisons


def validate_shard(
    spec: ShardSpec,
    *,
    seeds: Iterable[str],
    group_axes: Iterable[str],
    confidence_source: str,
    confidence_thresholds: Iterable[float],
    reject_fractions: Iterable[float],
    review_fraction: float,
) -> dict[str, object]:
    """Validate all run-level outputs and the exact policy Cartesian product."""

    required = ["missing_inputs.csv", "profile_manifest.csv", *RUN_LEVEL_FILES]
    missing_files = [
        name for name in required if not (spec.out_dir / name).exists()
    ]
    if missing_files:
        raise ShardValidationError(
            f"{spec.model_key}/{spec.dataset} missing files: {missing_files}"
        )

    missing_inputs = _read_csv(spec.out_dir / "missing_inputs.csv")
    if not missing_inputs.empty:
        raise ShardValidationError(
            f"{spec.model_key}/{spec.dataset} has missing inputs: "
            f"{missing_inputs.to_dict('records')}"
        )

    manifest = _read_csv(spec.out_dir / "profile_manifest.csv")
    expected_manifest = profile_manifest()
    if _manifest_records(manifest) != _manifest_records(expected_manifest):
        raise ShardValidationError(
            f"{spec.model_key}/{spec.dataset} profile manifest differs "
            "from the fixed core-plus-diagnostic manifest"
        )
    primary = set(
        manifest.loc[
            _truthy(manifest["primary_candidate"]),
            "profile",
        ].astype(str)
    )
    expected_primary = {
        "uncertainty",
        "prior",
        "uncertainty_prior",
        "uncertainty_support",
        "uncertainty_prior_support",
    }
    if primary != expected_primary:
        raise ShardValidationError(
            f"Primary profile set changed: {sorted(primary)}"
        )
    diagnostic = manifest.loc[
        manifest["profile"].astype(str).eq(
            "general_target_joint_support"
        )
    ]
    if len(diagnostic) != 1 or _truthy(
        diagnostic["primary_candidate"]
    ).any():
        raise ShardValidationError(
            "general_target_joint_support must exist exactly once as non-primary"
        )

    expected_policies = _policy_keys(
        spec,
        seeds=seeds,
        group_axes=group_axes,
        confidence_source=confidence_source,
        confidence_thresholds=confidence_thresholds,
        reject_fractions=reject_fractions,
        review_fraction=review_fraction,
    )
    profile_names = set(manifest["profile"].astype(str))
    tables = {
        name: _read_csv(spec.out_dir / name) for name in RUN_LEVEL_FILES
    }
    for name, frame in tables.items():
        _assert_unique(frame, TABLE_KEYS[name], label=name)

    for name in (
        "selection_choices.csv",
        "selected_test_runs.csv",
        "audit_fit_reference_diagnostics.csv",
    ):
        observed = _key_set(tables[name], POLICY_KEYS)
        if observed != expected_policies:
            raise ShardValidationError(
                f"{name} policy coverage differs: "
                f"expected {len(expected_policies)}, observed {len(observed)}"
            )

    for name in (
        "candidate_validation_runs.csv",
        "candidate_test_runs.csv",
    ):
        frame = tables[name]
        expected = {
            policy + (profile,)
            for policy, profile in product(expected_policies, profile_names)
        }
        observed = _key_set(frame, POLICY_KEYS + ["profile"])
        if observed != expected:
            raise ShardValidationError(
                f"{name} is incomplete: expected {len(expected)}, "
                f"observed {len(observed)}"
            )
        general = frame.loc[
            frame["profile"].astype(str).eq(
                "general_target_joint_support"
            )
        ]
        evidence_unavailable = general["model_status"].astype(str).isin(
            {
                "diagnostic_unavailable",
                "profile_unavailable",
                "unavailable",
            }
        )
        if general.empty or evidence_unavailable.any():
            raise ShardValidationError(
                f"{name} does not have complete generic target/joint evidence"
            )

    coverage = tables["policy_coverage.csv"]
    expected_coverage = {
        policy + (split,)
        for policy, split in product(
            expected_policies,
            ("validation", "test"),
        )
    }
    if _key_set(coverage, POLICY_KEYS + ["split"]) != expected_coverage:
        raise ShardValidationError("policy_coverage.csv is incomplete")

    calibration = tables["calibration_diagnostics.csv"]
    expected_calibration = {
        policy + (split, scope, score_source)
        for policy, split, scope, score_source in product(
            expected_policies,
            ("validation", "test"),
            ("candidate", "deferred", "residual"),
            ("raw", "calibrated"),
        )
    }
    if (
        _key_set(
            calibration,
            POLICY_KEYS + ["split", "scope", "score_source"],
        )
        != expected_calibration
    ):
        raise ShardValidationError(
            "calibration_diagnostics.csv is incomplete"
        )

    capabilities = tables["capability_manifest.csv"]
    expected_capabilities = {
        (
            _normalise_value(spec.model_name),
            _normalise_value(spec.dataset),
            _normalise_value(
                seed if str(seed).startswith("seed_") else f"seed_{seed}"
            ),
            split,
        )
        for seed, split in product(seeds, ("validation", "test"))
    }
    if (
        _key_set(
            capabilities,
            ["model", "dataset", "test_run", "split"],
        )
        != expected_capabilities
    ):
        raise ShardValidationError("capability_manifest.csv is incomplete")
    for column in (
        "uncertainty_available",
        "prior_available",
        "empirical_support_available",
        "native_support_available",
        "direct_target_support_available",
        "joint_support_available",
    ):
        if column not in capabilities or not _truthy(capabilities[column]).all():
            raise ShardValidationError(
                f"capability_manifest.csv has unavailable {column}"
            )

    comparisons = _expected_comparisons(manifest)
    expected_increments = {
        policy + (split, component, baseline, augmented)
        for policy, split, (
            component,
            baseline,
            augmented,
        ) in product(
            expected_policies,
            ("validation", "test"),
            comparisons,
        )
    }
    increments = tables["component_increment_runs.csv"]
    observed_increments = _key_set(
        increments,
        POLICY_KEYS
        + [
            "split",
            "added_component",
            "baseline_profile",
            "augmented_profile",
        ],
    )
    if observed_increments != expected_increments:
        raise ShardValidationError(
            "component_increment_runs.csv is incomplete: "
            f"expected {len(expected_increments)}, "
            f"observed {len(observed_increments)}"
        )

    domain = tables["domain_support_diagnostics.csv"]
    test_coverage = coverage.loc[
        coverage["split"].astype(str).eq("test")
        & pd.to_numeric(
            coverage["residual_n"],
            errors="coerce",
        ).gt(0)
    ]
    expected_domain_policies = _key_set(test_coverage, POLICY_KEYS)
    if _key_set(domain, POLICY_KEYS) != expected_domain_policies:
        raise ShardValidationError(
            "domain_support_diagnostics.csv does not match policies with "
            "non-empty test residuals"
        )

    provenance = tables["source_support_provenance.csv"]
    if len(provenance) != 1:
        raise ShardValidationError(
            "source_support_provenance.csv must have exactly one row"
        )
    if (
        str(provenance.iloc[0]["dataset"]).lower() != spec.dataset
        or str(provenance.iloc[0]["joint_support_status"]).lower()
        != "available"
        or str(provenance.iloc[0]["direct_target_support_status"]).lower()
        != "available"
    ):
        raise ShardValidationError(
            "source support provenance does not confirm direct/joint assets"
        )

    human_diagnostics = tables[
        "human_pair_deduplication_diagnostics.csv"
    ]
    human_selected = tables[
        "human_deduplicated_selected_test_runs.csv"
    ]
    if spec.dataset == "human":
        for name, frame in (
            ("human_pair_deduplication_diagnostics.csv", human_diagnostics),
            ("human_deduplicated_selected_test_runs.csv", human_selected),
        ):
            if _key_set(frame, POLICY_KEYS) != expected_policies:
                raise ShardValidationError(f"{name} is incomplete")
        required_dedup = {
            "baseline_deduplicated_error_detection_auprc",
            "augmented_deduplicated_error_detection_auprc",
            "delta_deduplicated_error_detection_auprc",
        }
        if not required_dedup.issubset(increments.columns):
            raise ShardValidationError(
                "Human component increments lack pair-deduplicated metrics"
            )
    elif not human_diagnostics.empty or not human_selected.empty:
        raise ShardValidationError(
            "Non-Human shard contains Human pair-deduplication rows"
        )

    return {
        "model": spec.model_name,
        "dataset": spec.dataset,
        "out_dir": str(spec.out_dir),
        "policies": len(expected_policies),
        "profiles": len(profile_names),
        "candidate_rows": len(tables["candidate_test_runs.csv"]),
        "increment_rows": len(increments),
        "status": "complete",
    }


def _write_completion_marker(
    spec: ShardSpec,
    report: dict[str, object],
    *,
    shard_bootstrap_resamples: int,
) -> None:
    payload = {
        "schema_version": 1,
        **report,
        "shard_bootstrap_resamples": int(shard_bootstrap_resamples),
    }
    path = spec.out_dir / "shard_complete.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def run_shard(
    spec: ShardSpec,
    *,
    python_executable: str,
    calibration_root: str,
    dataset_root: str,
    seeds: list[str],
    group_axes: list[str],
    confidence_source: str,
    confidence_thresholds: list[float],
    reject_fractions: list[float],
    review_fraction: float,
    shard_bootstrap_resamples: int,
    precision: int,
) -> dict[str, object]:
    """Run one shard, or adopt an already complete shard after validation."""

    validation_kwargs = {
        "seeds": seeds,
        "group_axes": group_axes,
        "confidence_source": confidence_source,
        "confidence_thresholds": confidence_thresholds,
        "reject_fractions": reject_fractions,
        "review_fraction": review_fraction,
    }
    try:
        report = validate_shard(spec, **validation_kwargs)
    except ShardValidationError:
        report = None
    if report is not None:
        _write_completion_marker(
            spec,
            report,
            shard_bootstrap_resamples=shard_bootstrap_resamples,
        )
        print(
            f"[shard] skip complete {spec.model_key}/{spec.dataset}",
            flush=True,
        )
        return report

    marker = spec.out_dir / "shard_complete.json"
    if marker.exists():
        marker.unlink()
    command = [
        python_executable,
        "audit_framework/scripts/run_audit.py",
        "--calibration-root",
        calibration_root,
        "--dataset-root",
        dataset_root,
        "--test-roots",
        spec.test_root,
        "--datasets",
        spec.dataset,
        "--seeds",
        *[str(seed) for seed in seeds],
        "--group-axes",
        *group_axes,
        "--confidence-source",
        confidence_source,
        "--confidence-thresholds",
        *[str(value) for value in confidence_thresholds],
        "--uncertainty-reject-fractions",
        *[str(value) for value in reject_fractions],
        "--review-fraction",
        str(review_fraction),
        "--bootstrap-resamples",
        str(shard_bootstrap_resamples),
        "--precision",
        str(precision),
        "--out-dir",
        str(spec.out_dir),
    ]
    print(
        f"[shard] run {spec.model_key}/{spec.dataset}",
        flush=True,
    )
    subprocess.run(command, check=True)
    report = validate_shard(spec, **validation_kwargs)
    _write_completion_marker(
        spec,
        report,
        shard_bootstrap_resamples=shard_bootstrap_resamples,
    )
    print(
        f"[shard] verified {spec.model_key}/{spec.dataset}: "
        f"{report['candidate_rows']} candidates, "
        f"{report['increment_rows']} increments",
        flush=True,
    )
    return report


def _sort_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    usable = [column for column in columns if column in frame]
    if not usable or frame.empty:
        return frame.reset_index(drop=True)
    return frame.sort_values(usable, kind="stable").reset_index(drop=True)


def merge_shards(
    specs: list[ShardSpec],
    *,
    merged_dir: str | Path,
    precision: int,
    validation_kwargs: dict[str, object],
) -> pd.DataFrame:
    """Validate all shards, reject duplicate keys, and merge run-level rows."""

    reports = [
        validate_shard(spec, **validation_kwargs) for spec in specs
    ]
    out_dir = Path(merged_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifests = [
        _read_csv(spec.out_dir / "profile_manifest.csv") for spec in specs
    ]
    reference_manifest = _manifest_records(manifests[0])
    if any(
        _manifest_records(manifest) != reference_manifest
        for manifest in manifests[1:]
    ):
        raise ShardValidationError("Shard profile manifests are not identical")
    write_csv(manifests[0], out_dir / "profile_manifest.csv", precision)
    write_csv(
        pd.DataFrame(
            columns=[
                "test_root",
                "dataset",
                "test_run",
                "reason",
                "expected_validation",
            ]
        ),
        out_dir / "missing_inputs.csv",
        precision,
    )

    for name in RUN_LEVEL_FILES:
        frames = [_read_csv(spec.out_dir / name) for spec in specs]
        if name == "source_support_provenance.csv":
            combined = pd.concat(frames, ignore_index=True)
            comparison_columns = list(combined.columns)
            conflicts = []
            kept = []
            for _, group in combined.groupby(
                ["dataset", "split"],
                dropna=False,
                sort=True,
            ):
                canonical = group.fillna("").astype(str)
                if len(canonical.drop_duplicates(comparison_columns)) != 1:
                    conflicts.append(
                        group[["dataset", "split"]].iloc[0].to_dict()
                    )
                kept.append(group.iloc[0])
            if conflicts:
                raise ShardValidationError(
                    f"Conflicting source provenance rows: {conflicts}"
                )
            merged = pd.DataFrame(kept)
        else:
            merged = pd.concat(frames, ignore_index=True)
            _assert_unique(merged, TABLE_KEYS[name], label=f"merged {name}")
        merged = _sort_frame(merged, TABLE_KEYS[name])
        write_csv(merged, out_dir / name, precision)

    report_frame = pd.DataFrame(reports)
    report_frame = _sort_frame(report_frame, ["model", "dataset"])
    write_csv(
        report_frame,
        out_dir / "shard_merge_manifest.csv",
        precision,
    )
    return report_frame


def summarize_merged(
    *,
    python_executable: str,
    merged_dir: str | Path,
    datasets: list[str],
    bootstrap_resamples: int,
    bootstrap_seed: int,
    precision: int,
) -> None:
    command = [
        python_executable,
        "audit_framework/scripts/run_audit.py",
        "--summary-only",
        "--datasets",
        *datasets,
        "--bootstrap-resamples",
        str(bootstrap_resamples),
        "--bootstrap-seed",
        str(bootstrap_seed),
        "--precision",
        str(precision),
        "--out-dir",
        str(merged_dir),
    ]
    subprocess.run(command, check=True)


def _summary_row(
    frame: pd.DataFrame,
    *,
    value_column: str,
    resamples: int,
    seed: int,
) -> dict[str, object]:
    values = pd.to_numeric(frame[value_column], errors="coerce").dropna()
    if values.empty:
        return {
            "num_independent_runs": 0,
            "num_model_dataset_strata": 0,
            "mean": np.nan,
            "run_win_rate": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }
    strata = (
        frame.groupby(["model", "dataset"], dropna=False)[value_column]
        .mean()
        .dropna()
    )
    lo, hi = _stratified_bootstrap(
        frame,
        value_column,
        resamples=resamples,
        seed=seed,
    )
    return {
        "num_independent_runs": int(len(values)),
        "num_model_dataset_strata": int(len(strata)),
        "mean": float(strata.mean()),
        "run_win_rate": float((values > 0.0).mean()),
        "ci_low": lo,
        "ci_high": hi,
    }


def write_diagnostic_analysis(
    *,
    merged_dir: str | Path,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    precision: int,
) -> None:
    """Write predeclared diagnostic-specific direction and dedup tables."""

    out_dir = Path(merged_dir)
    increments = _read_csv(out_dir / "component_increment_runs.csv")
    work = increments.loc[
        increments["added_component"].astype(str).eq(
            "general_target_joint_support"
        )
    ].copy()
    cluster_keys = [
        "model",
        "dataset",
        "test_run",
        "group_axis",
        "split",
    ]
    value_columns = [
        column
        for column in (
            "delta_validation_fold_mean_auprc",
            "delta_error_detection_auprc",
            "delta_deduplicated_error_detection_auprc",
            "delta_review_lift",
            "delta_retained_accuracy_gain_vs_residual",
        )
        if column in work
    ]
    clusters = (
        work.groupby(cluster_keys, dropna=False)[value_columns]
        .mean()
        .reset_index()
    )
    write_csv(
        _sort_frame(clusters, cluster_keys),
        out_dir / "general_target_joint_support_seed_strata.csv",
        precision,
    )

    validation = clusters.loc[
        clusters["split"].astype(str).eq("validation")
    ][
        [
            "model",
            "dataset",
            "test_run",
            "group_axis",
            "delta_validation_fold_mean_auprc",
        ]
    ].rename(
        columns={
            "delta_validation_fold_mean_auprc": (
                "validation_delta_fold_auprc"
            )
        }
    )
    test = clusters.loc[
        clusters["split"].astype(str).eq("test")
    ][
        [
            "model",
            "dataset",
            "test_run",
            "group_axis",
            "delta_error_detection_auprc",
        ]
    ].rename(
        columns={
            "delta_error_detection_auprc": "test_delta_auprc"
        }
    )
    direction = validation.merge(
        test,
        on=["model", "dataset", "test_run", "group_axis"],
        how="outer",
        validate="one_to_one",
    )
    paired = (
        direction["validation_delta_fold_auprc"].notna()
        & direction["test_delta_auprc"].notna()
    )
    direction["same_direction"] = pd.Series(
        pd.NA,
        index=direction.index,
        dtype="boolean",
    )
    direction["both_positive"] = pd.Series(
        pd.NA,
        index=direction.index,
        dtype="boolean",
    )
    direction.loc[paired, "same_direction"] = (
        np.sign(
            direction.loc[paired, "validation_delta_fold_auprc"]
        )
        == np.sign(direction.loc[paired, "test_delta_auprc"])
    )
    direction.loc[paired, "both_positive"] = (
        direction.loc[paired, "validation_delta_fold_auprc"].gt(0.0)
        & direction.loc[paired, "test_delta_auprc"].gt(0.0)
    )
    write_csv(
        _sort_frame(
            direction,
            ["group_axis", "model", "dataset", "test_run"],
        ),
        out_dir / "general_target_joint_support_direction_runs.csv",
        precision,
    )

    overall_rows = []
    for offset, (split, group_axis) in enumerate(
        product(("validation", "test"), ("target", "drug"))
    ):
        subset = clusters.loc[
            clusters["split"].astype(str).eq(split)
            & clusters["group_axis"].astype(str).eq(group_axis)
        ].copy()
        measure = (
            "delta_validation_fold_mean_auprc"
            if split == "validation"
            else "delta_error_detection_auprc"
        )
        overall_rows.append(
            {
                "split": split,
                "group_axis": group_axis,
                "measure": measure,
                **_summary_row(
                    subset,
                    value_column=measure,
                    resamples=bootstrap_resamples,
                    seed=bootstrap_seed + offset,
                ),
            }
        )
    write_csv(
        pd.DataFrame(overall_rows),
        out_dir / "general_target_joint_support_overall.csv",
        precision,
    )

    direction_summary = (
        direction.groupby("group_axis", dropna=False)
        .agg(
            num_independent_runs=("same_direction", "count"),
            same_direction_rate=("same_direction", "mean"),
            both_positive_rate=("both_positive", "mean"),
        )
        .reset_index()
    )
    write_csv(
        direction_summary,
        out_dir / "general_target_joint_support_direction_summary.csv",
        precision,
    )

    human = clusters.loc[
        clusters["dataset"].astype(str).str.lower().eq("human")
        & clusters["split"].astype(str).eq("test")
    ].copy()
    dedup_rows = []
    if "delta_deduplicated_error_detection_auprc" in human:
        for offset, group_axis in enumerate(("target", "drug")):
            subset = human.loc[
                human["group_axis"].astype(str).eq(group_axis)
            ].copy()
            dedup_rows.append(
                {
                    "group_axis": group_axis,
                    "measure": (
                        "delta_deduplicated_error_detection_auprc"
                    ),
                    **_summary_row(
                        subset,
                        value_column=(
                            "delta_deduplicated_error_detection_auprc"
                        ),
                        resamples=bootstrap_resamples,
                        seed=bootstrap_seed + 10_000 + offset,
                    ),
                }
            )
    write_csv(
        pd.DataFrame(dedup_rows),
        out_dir
        / "general_target_joint_support_human_pair_deduplicated_summary.csv",
        precision,
    )
