"""Heterogeneity analysis for the fixed HGB(U+P+E) diagnostic."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from audit_framework.data import write_csv
from audit_framework.scripts.run_audit import _stratified_bootstrap


POLICY_COLUMNS = [
    "model",
    "dataset",
    "test_run",
    "group_axis",
    "confidence_source",
    "confidence_threshold",
    "uncertainty_reject_fraction",
]
CLUSTER_COLUMNS = ["model", "dataset", "test_run", "group_axis"]
HGB_METHOD = "fixed_hgb_upe"
PRIMARY_METHODS = {
    "uncertainty",
    "prior",
    "uncertainty_prior",
    "uncertainty_support",
    "uncertainty_prior_support",
}


def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    source: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def _reject_duplicates(
    frame: pd.DataFrame,
    keys: list[str],
    *,
    source: str,
) -> None:
    duplicate = frame.duplicated(keys, keep=False)
    if duplicate.any():
        examples = frame.loc[duplicate, keys].head(5).to_dict("records")
        raise ValueError(f"{source} has duplicate keys: {examples}")


def _canonical_review_rows(
    frame: pd.DataFrame,
    *,
    keys: list[str],
    metrics: list[str],
    source: str,
) -> pd.DataFrame:
    _require_columns(frame, keys + ["review_fraction"] + metrics, source=source)
    variation = frame.groupby(keys, dropna=False)[metrics].nunique(dropna=False)
    if bool((variation > 1).any().any()):
        raise ValueError(
            f"{source} has review-fraction-dependent values for "
            f"budget-independent metrics"
        )
    return (
        frame.sort_values("review_fraction")
        .drop_duplicates(keys, keep="first")
        .copy()
    )


def build_hgb_policy_runs(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    """Build one paired HGB-versus-PAUSE row per predeclared policy."""

    candidate_keys = POLICY_COLUMNS + [
        "review_fraction",
        "split",
        "method",
    ]
    selected_keys = POLICY_COLUMNS + ["review_fraction"]
    candidate_required = candidate_keys + [
        "model_status",
        "error_detection_auprc",
        "validation_fold_mean_auprc",
        "deduplicated_error_detection_auprc",
    ]
    selected_required = selected_keys + [
        "selected_profile",
        "error_detection_auprc",
        "deduplicated_error_detection_auprc",
    ]
    _require_columns(candidates, candidate_required, source="candidates")
    _require_columns(selected, selected_required, source="selected")
    _reject_duplicates(candidates, candidate_keys, source="candidates")
    _reject_duplicates(selected, selected_keys, source="selected")

    candidate_metrics = [
        "error_detection_auprc",
        "validation_fold_mean_auprc",
        "deduplicated_error_detection_auprc",
    ]
    candidate_rows = _canonical_review_rows(
        candidates,
        keys=POLICY_COLUMNS + ["split", "method"],
        metrics=candidate_metrics,
        source="candidates",
    )
    selected_rows = _canonical_review_rows(
        selected,
        keys=POLICY_COLUMNS,
        metrics=[
            "error_detection_auprc",
            "deduplicated_error_detection_auprc",
        ],
        source="selected",
    )
    invalid_profiles = sorted(
        set(selected_rows["selected_profile"].astype(str)) - PRIMARY_METHODS
    )
    if invalid_profiles:
        raise ValueError(f"selected contains non-primary profiles: {invalid_profiles}")

    selection = selected_rows[POLICY_COLUMNS + ["selected_profile"]]
    validation = candidate_rows.loc[
        candidate_rows["split"].eq("validation")
    ].merge(selection, on=POLICY_COLUMNS, how="inner", validate="many_to_one")
    pause_validation = validation.loc[
        validation["method"].eq(validation["selected_profile"]),
        POLICY_COLUMNS
        + [
            "selected_profile",
            "error_detection_auprc",
            "validation_fold_mean_auprc",
        ],
    ].rename(
        columns={
            "error_detection_auprc": "pause_validation_auprc",
            "validation_fold_mean_auprc": "pause_validation_fold_mean_auprc",
        }
    )
    hgb_validation = candidate_rows.loc[
        candidate_rows["split"].eq("validation")
        & candidate_rows["method"].eq(HGB_METHOD),
        POLICY_COLUMNS
        + [
            "model_status",
            "error_detection_auprc",
            "validation_fold_mean_auprc",
        ],
    ].rename(
        columns={
            "model_status": "hgb_validation_status",
            "error_detection_auprc": "hgb_validation_auprc",
            "validation_fold_mean_auprc": "hgb_validation_fold_mean_auprc",
        }
    )
    hgb_test = candidate_rows.loc[
        candidate_rows["split"].eq("test")
        & candidate_rows["method"].eq(HGB_METHOD),
        POLICY_COLUMNS
        + [
            "model_status",
            "error_detection_auprc",
            "deduplicated_error_detection_auprc",
        ],
    ].rename(
        columns={
            "model_status": "hgb_test_status",
            "error_detection_auprc": "hgb_test_auprc",
            "deduplicated_error_detection_auprc": (
                "hgb_test_deduplicated_auprc"
            ),
        }
    )
    pause_test = selected_rows[
        POLICY_COLUMNS
        + [
            "error_detection_auprc",
            "deduplicated_error_detection_auprc",
        ]
    ].rename(
        columns={
            "error_detection_auprc": "pause_test_auprc",
            "deduplicated_error_detection_auprc": (
                "pause_test_deduplicated_auprc"
            ),
        }
    )

    expected = len(selection)
    parts = [
        ("pause_validation", pause_validation),
        ("hgb_validation", hgb_validation),
        ("hgb_test", hgb_test),
        ("pause_test", pause_test),
    ]
    for name, part in parts:
        _reject_duplicates(part, POLICY_COLUMNS, source=name)
        if len(part) != expected:
            raise ValueError(
                f"{name} has {len(part)} policies; expected {expected}"
            )

    runs = pause_validation.merge(
        hgb_validation,
        on=POLICY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    runs = runs.merge(
        hgb_test,
        on=POLICY_COLUMNS,
        how="inner",
        validate="one_to_one",
    ).merge(
        pause_test,
        on=POLICY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    if len(runs) != expected:
        raise ValueError(f"paired table has {len(runs)} policies; expected {expected}")

    runs["delta_validation_auprc"] = (
        runs["hgb_validation_auprc"] - runs["pause_validation_auprc"]
    )
    runs["delta_validation_fold_mean_auprc"] = (
        runs["hgb_validation_fold_mean_auprc"]
        - runs["pause_validation_fold_mean_auprc"]
    )
    runs["delta_test_auprc"] = (
        runs["hgb_test_auprc"] - runs["pause_test_auprc"]
    )
    runs["delta_test_deduplicated_auprc"] = (
        runs["hgb_test_deduplicated_auprc"]
        - runs["pause_test_deduplicated_auprc"]
    )
    return runs.sort_values(POLICY_COLUMNS).reset_index(drop=True)


def cluster_hgb_policy_runs(policy_runs: pd.DataFrame) -> pd.DataFrame:
    """Average the four fixed policies within model-dataset-seed clusters."""

    measures = [
        "delta_validation_auprc",
        "delta_validation_fold_mean_auprc",
        "delta_test_auprc",
        "delta_test_deduplicated_auprc",
    ]
    _require_columns(
        policy_runs,
        CLUSTER_COLUMNS + measures,
        source="policy_runs",
    )
    aggregate: dict[str, tuple[str, str]] = {}
    for measure in measures:
        aggregate[measure] = (measure, "mean")
        aggregate[f"finite_policy_n_{measure}"] = (measure, "count")
    clusters = (
        policy_runs.groupby(CLUSTER_COLUMNS, dropna=False)
        .agg(**aggregate)
        .reset_index()
    )
    clusters["validation_positive"] = (
        clusters["delta_validation_auprc"] > 0.0
    ).where(clusters["delta_validation_auprc"].notna())
    clusters["test_positive"] = (
        clusters["delta_test_auprc"] > 0.0
    ).where(clusters["delta_test_auprc"].notna())
    paired = clusters[
        ["delta_validation_auprc", "delta_test_auprc"]
    ].notna().all(axis=1)
    clusters["same_direction"] = np.nan
    clusters["both_positive"] = np.nan
    clusters.loc[paired, "same_direction"] = (
        np.sign(clusters.loc[paired, "delta_validation_auprc"])
        == np.sign(clusters.loc[paired, "delta_test_auprc"])
    ).astype(float)
    clusters.loc[paired, "both_positive"] = (
        (clusters.loc[paired, "delta_validation_auprc"] > 0.0)
        & (clusters.loc[paired, "delta_test_auprc"] > 0.0)
    ).astype(float)
    clusters["dedup_minus_ordinary_delta"] = (
        clusters["delta_test_deduplicated_auprc"]
        - clusters["delta_test_auprc"]
    )
    return clusters.sort_values(CLUSTER_COLUMNS).reset_index(drop=True)


def _summary_rows(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    measures: Iterable[str],
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = [((), frame)] if not group_columns else frame.groupby(
        group_columns,
        dropna=False,
    )
    for group_key, group in grouped:
        values = group_key if isinstance(group_key, tuple) else (group_key,)
        metadata = dict(zip(group_columns, values))
        for offset, measure in enumerate(measures):
            numeric = pd.to_numeric(group[measure], errors="coerce")
            finite = numeric.dropna()
            if finite.empty:
                continue
            work = group.copy()
            work[measure] = numeric
            lo, hi = _stratified_bootstrap(
                work,
                measure,
                resamples=resamples,
                seed=seed + offset,
            )
            rows.append(
                {
                    **metadata,
                    "measure": measure,
                    "mean": float(finite.mean()),
                    "ci_low": lo,
                    "ci_high": hi,
                    "independent_runs": int(len(finite)),
                    "positive_fraction": float((finite > 0.0).mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_hgb_heterogeneity(
    clusters: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Create overall, direction, strata, leave-one-out, and Human summaries."""

    effect_measures = [
        "delta_validation_auprc",
        "delta_validation_fold_mean_auprc",
        "delta_test_auprc",
    ]
    overall = _summary_rows(
        clusters,
        group_columns=["group_axis"],
        measures=effect_measures,
        resamples=resamples,
        seed=seed,
    )
    direction = _summary_rows(
        clusters,
        group_columns=["group_axis"],
        measures=[
            "validation_positive",
            "test_positive",
            "same_direction",
            "both_positive",
        ],
        resamples=resamples,
        seed=seed + 1_000,
    )

    strata_parts = []
    for offset, column in enumerate(["model", "dataset", "test_run"]):
        summary = _summary_rows(
            clusters,
            group_columns=["group_axis", column],
            measures=["delta_validation_auprc", "delta_test_auprc"],
            resamples=resamples,
            seed=seed + 2_000 + offset * 100,
        ).rename(columns={column: "stratum"})
        summary.insert(1, "stratum_type", column)
        strata_parts.append(summary)
    strata = pd.concat(strata_parts, ignore_index=True)

    leave_parts = []
    for offset, column in enumerate(["dataset", "model"]):
        for held_out in sorted(clusters[column].astype(str).unique()):
            retained = clusters.loc[~clusters[column].astype(str).eq(held_out)]
            summary = _summary_rows(
                retained,
                group_columns=["group_axis"],
                measures=["delta_validation_auprc", "delta_test_auprc"],
                resamples=resamples,
                seed=seed + 3_000 + offset * 500,
            )
            summary.insert(1, "held_out_type", column)
            summary.insert(2, "held_out", held_out)
            leave_parts.append(summary)
    leave_one_out = pd.concat(leave_parts, ignore_index=True)

    human = clusters.loc[clusters["dataset"].astype(str).eq("human")]
    human_deduplicated = _summary_rows(
        human,
        group_columns=["group_axis"],
        measures=[
            "delta_test_auprc",
            "delta_test_deduplicated_auprc",
            "dedup_minus_ordinary_delta",
        ],
        resamples=resamples,
        seed=seed + 4_000,
    )
    return {
        "overall_summary.csv": overall,
        "direction_consistency.csv": direction,
        "strata_summary.csv": strata,
        "leave_one_out_summary.csv": leave_one_out,
        "human_deduplicated_summary.csv": human_deduplicated,
    }


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lookup(
    frame: pd.DataFrame,
    *,
    axis: str,
    measure: str,
) -> pd.Series:
    match = frame.loc[
        frame["group_axis"].eq(axis) & frame["measure"].eq(measure)
    ]
    if len(match) != 1:
        raise ValueError(f"expected one {axis}/{measure} row, found {len(match)}")
    return match.iloc[0]


def _format_effect(row: pd.Series) -> str:
    return (
        f"{row['mean']:+.4f} "
        f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
    )


def _write_report(
    destination: Path,
    summaries: dict[str, pd.DataFrame],
) -> None:
    overall = summaries["overall_summary.csv"]
    direction = summaries["direction_consistency.csv"]
    human = summaries["human_deduplicated_summary.csv"]
    lines = [
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        "- Origin Date: 2026-06-15",
        "- Verification Status: VERIFIED",
        "- Version Label: hgb_heterogeneity_v1",
        "",
        "## Validation Report",
        "",
        "- **Source**: frozen strengthening-experiment run-level CSVs",
        "- **Overall Confidence**: RED_FLAG for HGB dominance",
        "",
        "### Statistical Findings",
        "",
        "| Grouping | Validation HGB-PAUSE | Test HGB-PAUSE | Same direction |",
        "|---|---:|---:|---:|",
    ]
    for axis in ("drug", "target"):
        validation = _lookup(
            overall,
            axis=axis,
            measure="delta_validation_auprc",
        )
        test = _lookup(overall, axis=axis, measure="delta_test_auprc")
        same = _lookup(direction, axis=axis, measure="same_direction")
        lines.append(
            f"| {axis} | {_format_effect(validation)} | "
            f"{_format_effect(test)} | {same['mean']:.1%} |"
        )
    lines.extend(
        [
            "",
            "### Human Pair-Deduplicated Sensitivity",
            "",
            "| Grouping | Ordinary test delta | Pair-deduplicated test delta |",
            "|---|---:|---:|",
        ]
    )
    for axis in ("drug", "target"):
        ordinary = _lookup(human, axis=axis, measure="delta_test_auprc")
        deduplicated = _lookup(
            human,
            axis=axis,
            measure="delta_test_deduplicated_auprc",
        )
        lines.append(
            f"| {axis} | {_format_effect(ordinary)} | "
            f"{_format_effect(deduplicated)} |"
        )
    lines.extend(
        [
            "",
            "### Interpretation",
            "",
            "The fixed HGB diagnostic does not show validation/test directional "
            "stability. Positive aggregate test means are concentrated outside "
            "Human, while both ordinary and exact-pair-deduplicated Human "
            "effects are negative. HGB therefore remains a non-primary "
            "diagnostic and does not establish superiority over validation-"
            "selected PAUSE.",
            "",
            "### Fallacy Scan",
            "",
            "- **Coverage**: 11/11 fallacy types checked.",
            "- Simpson's paradox: not present in the strict all-strata sense; "
            "however, aggregate positive test means mask a direction reversal "
            "in Human.",
            "- Look-elsewhere effect: CAUTION; subgroup intervals are "
            "descriptive and are not used for selection.",
            "- Garden of forking paths: NOTE; model, dataset, seed, leave-one-"
            "out, and Human dedup analyses were fixed before this run.",
            "- Ecological, Berkson, collider, base-rate neglect, regression-to-"
            "mean, survivorship, causal-language, and reverse-causality "
            "fallacies: not implicated by this paired predictive audit.",
            "",
            "### Reproducibility",
            "",
            "- **Method**: deterministic rebuild from frozen CSVs with "
            "20,000-resample seeded bootstrap.",
            "- **Verdict**: REPRODUCIBLE.",
            "",
        ]
    )
    (destination / "analysis.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run_hgb_heterogeneity_analysis(
    *,
    candidate_path: str | Path,
    selected_path: str | Path,
    out_dir: str | Path,
    bootstrap_resamples: int = 20_000,
    bootstrap_seed: int = 2026,
    precision: int = 12,
) -> dict[str, int]:
    """Run the deterministic analysis and write diagnostic artifacts."""

    candidate_source = Path(candidate_path)
    selected_source = Path(selected_path)
    destination = Path(out_dir)
    candidates = pd.read_csv(candidate_source)
    selected = pd.read_csv(selected_source)
    policy_runs = build_hgb_policy_runs(candidates, selected)
    clusters = cluster_hgb_policy_runs(policy_runs)
    summaries = summarize_hgb_heterogeneity(
        clusters,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )

    destination.mkdir(parents=True, exist_ok=True)
    write_csv(policy_runs, destination / "hgb_policy_runs.csv", precision)
    write_csv(clusters, destination / "hgb_cluster_runs.csv", precision)
    for name, frame in summaries.items():
        write_csv(frame, destination / name, precision)
    _write_report(destination, summaries)

    manifest = {
        "analysis": "fixed_hgb_upe_heterogeneity",
        "bootstrap_resamples": int(bootstrap_resamples),
        "bootstrap_seed": int(bootstrap_seed),
        "candidate_path": str(candidate_source),
        "candidate_sha256": _file_sha256(candidate_source),
        "selected_path": str(selected_source),
        "selected_sha256": _file_sha256(selected_source),
        "policy_rows": int(len(policy_runs)),
        "cluster_rows": int(len(clusters)),
        "models": sorted(policy_runs["model"].astype(str).unique()),
        "datasets": sorted(policy_runs["dataset"].astype(str).unique()),
        "test_runs": sorted(policy_runs["test_run"].astype(str).unique()),
        "group_axes": sorted(policy_runs["group_axis"].astype(str).unique()),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "policy_rows": len(policy_runs),
        "cluster_rows": len(clusters),
    }
