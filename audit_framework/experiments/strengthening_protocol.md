# PAUSE Strengthening Experiments

Date frozen: 2026-06-15

These experiments are diagnostics around the frozen PAUSE core. They do not
modify the five primary profiles, the formal audit outputs, or the manuscript.

## Shared design

- Predictors: PACE, TAPB, and DrugBAN.
- Datasets: BioSNAP, BindingDB, and Human cluster splits.
- Seeds: 4, 5, 6, 7, and 8.
- Grouped validation axes: target and drug.
- Policy grid: base confidence thresholds 0.8 and 0.9 crossed with uncertainty
  defer fractions 0.10 and 0.20.
- Primary profiles: U, P, U+P, U+E, and U+P+E, unchanged.
- Profile selection: grouped-validation one-standard-error rule with a 0.005
  minimum gain over U and conservative fallback to U.
- Confidence intervals: one unified 20,000-resample stratified bootstrap over
  model-dataset-seed cluster estimates. Shard confidence intervals are never
  averaged.
- Human sensitivity: remove every test pair seen in validation, then give each
  remaining exact pair one vote.

## Experiment 1: equal total action budget

The uncertainty-only comparator and PAUSE receive exactly the same candidate
universe, defer count, and residual review count. Therefore the total action
count is:

```text
deferred candidates + reviewed residual candidates
```

Review fractions are fixed at 0.05, 0.10, 0.20, and 0.30. Primary outcomes are
combined recall of candidate errors, review error rate, and retained accuracy
gain. Error-detection AUPRC is reported as a budget-independent ranking metric.
The area under the combined-recall-versus-action-fraction curve is secondary.

## Experiment 2: cross-domain profile transfer

Two transfer analyses are fixed:

1. Leave one dataset out: choose one primary profile using validation results
   from the other two datasets, then apply that profile to every test run of
   the held-out dataset.
2. Leave one model out: choose one primary profile using validation results
   from the other two models, then apply that profile to every test run of the
   held-out model.

Within each source domain, the four policy settings are averaged before the
global one-standard-error calculation so a policy is not treated as an
independent biological replicate. Complexity breaks eligible ties. No
held-out test result participates in profile selection.

Because the existing test set has already been inspected during framework
development, this is a cross-domain sensitivity analysis rather than a claim
of untouched external confirmation.

## Experiment 3: strong baselines and falsification

Fixed baselines, all evaluated in the identical residual universe:

- raw frozen-predictor uncertainty;
- calibrated uncertainty U;
- native source-domain maximum shift;
- native source-domain density distance;
- fixed histogram gradient boosting on the frozen U+P+E feature block.

The gradient-boosting baseline uses grouped cross-fitting, fold-local evidence
calibration and support references, and fixed hyperparameters:

```text
learning_rate=0.05
max_iter=100
max_leaf_nodes=15
min_samples_leaf=20
l2_regularization=1.0
random_state=2026
```

It cannot enter PAUSE profile selection.

Falsification is validation-only. Inside every predeclared grouped fold, P, E,
or both feature blocks are independently permuted in the fit and held-out
partitions before fitting the unchanged U+P+E ranker. The fold assignment,
labels, candidate universe, U feature, estimator, and all policy thresholds
remain fixed. The test split is not used to select a permutation or variant.

## Output boundary

All outputs are written below:

```text
audit_framework/cache/diagnostic_results/strengthening_experiments/
```

The following locations must remain untouched:

```text
audit_framework/results/audit/
manuscript/
```
