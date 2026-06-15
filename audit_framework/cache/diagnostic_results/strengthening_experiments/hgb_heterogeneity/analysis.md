## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-06-15
- Verification Status: VERIFIED
- Version Label: hgb_heterogeneity_v1

## Validation Report

- **Source**: frozen strengthening-experiment run-level CSVs
- **Overall Confidence**: RED_FLAG for HGB dominance

### Statistical Findings

| Grouping | Validation HGB-PAUSE | Test HGB-PAUSE | Same direction |
|---|---:|---:|---:|
| drug | -0.0914 [-0.2909, +0.0376] | +0.0130 [-0.0651, +0.0968] | 63.9% |
| target | -0.1530 [-0.3043, -0.0614] | +0.1264 [-0.0206, +0.2529] | 38.5% |

### Human Pair-Deduplicated Sensitivity

| Grouping | Ordinary test delta | Pair-deduplicated test delta |
|---|---:|---:|
| drug | -0.1226 [-0.1858, -0.0677] | -0.1455 [-0.2141, -0.0901] |
| target | -0.1577 [-0.2184, -0.1033] | -0.1734 [-0.2412, -0.1162] |

### Interpretation

The fixed HGB diagnostic does not show validation/test directional stability. Positive aggregate test means are concentrated outside Human, while both ordinary and exact-pair-deduplicated Human effects are negative. HGB therefore remains a non-primary diagnostic and does not establish superiority over validation-selected PAUSE.

### Fallacy Scan

- **Coverage**: 11/11 fallacy types checked.
- Simpson's paradox: not present in the strict all-strata sense; however, aggregate positive test means mask a direction reversal in Human.
- Look-elsewhere effect: CAUTION; subgroup intervals are descriptive and are not used for selection.
- Garden of forking paths: NOTE; model, dataset, seed, leave-one-out, and Human dedup analyses were fixed before this run.
- Ecological, Berkson, collider, base-rate neglect, regression-to-mean, survivorship, causal-language, and reverse-causality fallacies: not implicated by this paired predictive audit.

### Reproducibility

- **Method**: deterministic rebuild from frozen CSVs with 20,000-resample seeded bootstrap.
- **Verdict**: REPRODUCIBLE.
