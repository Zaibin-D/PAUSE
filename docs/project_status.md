# PAUSE Project Status

Last updated: 2026-06-15

## 1. Current Objective

PAUSE is a residual audit layer for frozen DTI predictors. It does not retrain
or replace the base predictor. Its operational question is:

> Among high-confidence predicted positives that remain after uncertainty
> triage, which cases should be reviewed first?

```text
frozen prediction
  -> high-confidence positive candidates
  -> calibrated error-risk deferment
  -> policy-matched residual candidates
  -> grouped validation-selected audit ranking
  -> limited review budget
```

The current formal scope is cluster split only. Random split is not part of the
active validation plan.

## 2. Formal Core

| Component | Role | Current decision |
|---|---|---|
| U | calibrated residual error risk | baseline and conservative fallback |
| P | directional prior/shortcut contradiction | primary audit evidence |
| E | empirical and source-domain support | primary audit evidence |

The formal PAUSE core is:

```text
U + P + E
```

P and E have separate scientific meanings:

- P asks whether the prediction conflicts with the learned prior or shortcut.
- E asks whether the frozen predictor's training experience supports the
  queried drug and target.

The clean formal E contains only:

1. grouped-fold entity support computed inside each validation fit partition:
   drug count, target count, minimum support, support imbalance, joint novelty,
   and one-sided novelty;
2. frozen source-domain drug support: Morgan radius-2, 2048-bit fingerprints
   with Tanimoto nearest-neighbour and top-5 distance;
3. frozen source-domain target support: ESM CLS cosine nearest-neighbour and
   top-5 distance;
4. fixed drug-target shift, density, and imbalance combinations derived from
   those native distances.

Exact-pair fit counts, source pair counts, source cluster counts, and distances
in the P prior representation are excluded from the formal E. They remain
available only in fixed diagnostic profiles named `source_domain_support`,
`legacy_support`, and `combined_support`.

A sixth fixed diagnostic, `target_sequence_support`, augments clean `U+E` with
MMseqs2 identity/family support. It is not a primary candidate and cannot alter
the formal profile selection.

The five profiles eligible for primary selection are:

```text
U
P
U+P
U+E
U+P+E
```

The diagnostic E profiles cannot win primary selection. The former M/S/I
research prototypes have been retired from the active repository and cannot be
enabled by a runtime flag.

## 3. Core-Only Implementation

The runtime is now genuinely core-only:

- the data loader contains only base, entity-text, and prior inputs;
- the model exposes only frozen base and prior logits;
- the training CLI contains only `base`, `prior`, `all`, and `eval` stages;
- the audit runner has no extension switch and evaluates only the frozen U/P/E
  profiles plus fixed non-primary E diagnostics;
- all source-domain support is loaded from frozen precomputed assets;
- the former local-evidence code, checkpoint, builders, tests, and dataset
  assets have been removed.

Native source-support assets are stored at:

```text
datasets/<dataset>/cluster/pime/native_source_support.pkl
datasets/<dataset>/cluster/pime/native_source_support.json
```

Current asset coverage:

| Dataset | Drugs | Targets |
|---|---:|---:|
| BioSNAP | 4,162 | 2,150 |
| BindingDB | 12,167 | 1,926 |
| Human | 1,813 | 1,503 |

They were generated in the project PyTorch environment with RDKit `2026.03.3`,
NumPy `2.1.2`, and scikit-learn `1.7.2`. RDKit is required only to rebuild the
Morgan assets; the formal audit consumes the frozen files and does not require
live RDKit computation. Cross-version verification against RDKit `2024.09.6`
changed zero Morgan distances.

BioSNAP reads the standalone frozen protein CLS asset. BindingDB and Human use
the stored 1,280-dimensional frozen ESM CLS block in `target_prior_feat.pkl`
because no separate CLS file is present. This is an embedding source for E,
not a label-derived support statistic.

Fixed MMseqs2 target-support assets are stored as:

```text
datasets/<dataset>/cluster/pime/target_sequence_support.pkl
datasets/<dataset>/cluster/pime/target_sequence_support.json
```

The predeclared family definition uses identity >=30%, query coverage >=50%,
and E-value <=1e-3. Support is the best identity multiplied by the minimum
query/target coverage among Swiss-Prot family hits represented by source-train
targets. Outcome labels are not loaded.

## 4. Formal Results

The complete cluster-split run covers three predictors, three datasets, five
seeds, target- and drug-grouped validation, and four fixed policy settings.

End-to-end validation-selected audit versus uncertainty:

| Grouping | Test AUPRC gain | 95% interval | Independent runs |
|---|---:|---:|---:|
| drug | +0.147898 | +0.067689 to +0.232070 | 43 |
| target | +0.042119 | +0.009887 to +0.076014 | 43 |

Rounded headline values:

```text
drug grouping:   +0.148, 95% CI [+0.068, +0.232]
target grouping: +0.042, 95% CI [+0.010, +0.076]
```

The native E revision preserves the drug result and modestly improves the
target headline relative to the previous core-only run.

Validation increments in error-detection AUPRC:

| Added evidence | Drug grouped | Target grouped |
|---|---:|---:|
| P versus U | +0.045 [+0.002,+0.102] | +0.050 [-0.025,+0.131] |
| clean E versus U | +0.180 [+0.119,+0.252] | +0.068 [-0.023,+0.160] |
| clean E after P | +0.160 [+0.111,+0.210] | +0.061 [-0.011,+0.148] |
| fold-fit entity support versus U | +0.105 [+0.030,+0.168] | +0.025 [-0.006,+0.057] |
| native source support versus U | +0.116 [+0.062,+0.188] | +0.070 [-0.029,+0.171] |
| native source support after fold-fit | +0.075 [+0.017,+0.154] | +0.043 [-0.063,+0.141] |
| fold-fit after native source support | +0.065 [+0.032,+0.099] | -0.002 [-0.029,+0.024] |

Legacy support adds no reliable validation gain after native support:

| Diagnostic increment | Drug grouped | Target grouped |
|---|---:|---:|
| native after legacy | -0.001 [-0.041,+0.038] | +0.012 [-0.020,+0.056] |
| legacy after native | -0.004 [-0.039,+0.023] | -0.013 [-0.029,+0.003] |

This supports keeping the cleaner native E and not enlarging the selectable
profile library.

### Scientific Interpretation

- Drug-grouped U+P+E evidence is stable and suitable as the strongest current
  result.
- Target-grouped selected audit is positive, but target E validation intervals
  remain broad and cross zero.
- The target selector therefore remains conservative: across 180 policy-level
  selections it chose U 109 times, P 54 times, U+E 11 times, and U+P+E 6 times.
- The target result should be reported as a positive but weaker cold-target
  result, not as proof that target-domain support is fully solved.
- The retired local-evidence prototypes are not part of the framework or its
  optimization path.

### Human Pair-Deduplication Sensitivity

The predeclared sensitivity analysis leaves validation selection unchanged,
removes every Human test pair that appears anywhere in Human validation, and
then gives each remaining exact test pair one pair-level vote. It removes 12
cross-split pairs (16 test rows) and collapses three additional duplicate rows.
There are no pair-label conflicts.

Human-only selected audit versus uncertainty after deduplication:

| Grouping | Test AUPRC gain | 95% interval |
|---|---:|---:|
| drug | -0.000995 | -0.017718 to +0.023798 |
| target | +0.003590 | -0.055499 to +0.063354 |

Human alone therefore does not provide stable evidence after exact-pair
deduplication.

Replacing the original Human contribution with its deduplicated result while
leaving BioSNAP and BindingDB unchanged gives:

| Grouping | Test AUPRC gain | 95% interval |
|---|---:|---:|
| drug | +0.145887 | +0.063819 to +0.231430 |
| target | +0.038913 | +0.002476 to +0.074477 |

The overall drug and target conclusions therefore survive the conservative
Human sensitivity analysis, although the target lower bound becomes narrower.

### Fixed Target-Support Diagnostic

Adding the single fixed MMseqs2 diagnostic after clean `U+E` gives:

| Grouping | Validation AUPRC increment | Test diagnostic increment |
|---|---:|---:|
| drug | +0.023229 [-0.020149,+0.076887] | +0.018204 [-0.010854,+0.051939] |
| target | +0.011231 [-0.016350,+0.045510] | +0.018046 [-0.011638,+0.053369] |

All intervals cross zero. Target-grouped leave-one-dataset-out validation
increments also cross zero:

| Excluded dataset | Increment | 95% interval |
|---|---:|---:|
| BioSNAP | +0.015285 | -0.026377 to +0.065536 |
| BindingDB | +0.004637 | -0.028140 to +0.049770 |
| Human | +0.013770 | -0.007119 to +0.045098 |

The positive tendency is heterogeneous and is strongest for PACE/BindingDB.
MMseqs2 support remains a useful fixed diagnostic but is not promoted into the
formal E core.

## 5. Leakage and Deployment Audit

The source-support loader reads only entity IDs and optional cluster IDs from
`source_train_with_id.csv`; it does not load outcome columns.

The formal run writes:

```text
audit_framework/results/audit/source_support_provenance.csv
audit_framework/results/audit/audit_fit_reference_diagnostics.csv
```

Across all 360 audit-fit diagnostics:

- the validation-derived audit reference was frozen before test scoring;
- no held-out test batch was appended to the reference;
- no reference outcomes were used to construct support features;
- exact-pair support was disabled in the formal core;
- validation/test drug, target, and pair overlaps were logged explicitly.

Human cluster validation and test contain repeated exact pairs. The largest
policy-residual overlap observed by the audit was 9 pairs. Exact-pair support
is disabled in the formal E, and the completed pair-deduplicated sensitivity
analysis confirms that the overall conclusion remains positive without those
pairs.

The fold-fit support is a frozen validation-selected audit reference for the
declared evaluation protocol. A future external deployment must construct its
support reference from the frozen predictor's actual training history or a
separately declared historical reference, never from future deployment/test
batches.

## 6. Data Asset Roles

Required by the formal U+P+E audit:

```text
drug_entity.csv
target_entity.csv
drug_prior_feat.pkl
target_prior_feat.pkl
source_train_with_id.csv
pime/native_source_support.pkl
```

`prot_cls_feat.pkl` is used when available. Otherwise, the frozen ESM CLS block
already stored in `target_prior_feat.pkl` is used.

Required only by the fixed MMseqs2 target diagnostic:

```text
mmseqs/
target_sequence_support.pkl
target_sequence_support.json
```

Required only by the new model-independent target/joint diagnostic:

```text
mmseqs/all_targets.fasta
mmseqs/source_targets.fasta
mmseqs/direct_target_to_source.tsv
direct_target_support.pkl
direct_target_support.json
joint_source_support.pkl
joint_source_support.json
```

The UniProt mapping files remain because they support label-free target-family
diagnostics. No local 3D or fragment asset is required by PAUSE.

## 7. Canonical Paths

Formal runner:

```text
audit_framework/scripts/run_audit.py
```

Native support builder:

```text
audit_framework/scripts/build_native_support.py
```

Only formal result directory:

```text
audit_framework/results/audit/
```

Source audit cache:

```text
audit_framework/cache/test_audits/
audit_framework/cache/validation_audits/
```

The manuscript is isolated under `manuscript/` and is not modified by the
audit pipeline.

## 8. Verification Status

- full target- and drug-grouped formal audit completed successfully;
- 25 audit-framework tests pass in the project PyTorch environment;
- `missing_inputs.csv` contains zero data rows;
- the frozen formal result manifest contains 5 primary and 6 fixed diagnostic
  profiles;
- the current code manifest contains the same 5 primary profiles plus the new
  non-primary `general_target_joint_support` diagnostic;
- no M/S/I profile or enable switch exists in active code;
- `audit_framework/results/audit/` is the only formal results directory;
- the retired local-evidence checkpoint and dataset assets have been deleted;
- no manuscript file was modified.

## 9. Current Interpretation and Paused Handoff

The present evidence supports PAUSE as a post-hoc, limited-budget error-audit
layer for frozen DTI predictors. It does not support claiming that PAUSE
universally solves cold-start DTI prediction.

The strongest supported claim is:

> After uncertainty triage, prior contradiction and empirical support improve
> the ranking of residual high-confidence errors, with the clearest and most
> stable evidence under drug-grouped cluster validation.

The target-grouped result is positive overall and survives the conservative
Human pair-deduplication sensitivity analysis, but it is weaker and
heterogeneous. The target selector falls back to U or P for most policy-level
decisions, and the existing fixed MMseqs2 diagnostic has intervals crossing
zero. Target E is therefore not considered solved.

The likely limitation is that the present target E is mainly a marginal,
global target-similarity measure. ESM CLS cosine and indirect Swiss-Prot family
support do not necessarily represent:

1. the target space actually used by each frozen predictor;
2. binding-site or ligand-recognition similarity;
3. whether the queried drug and target are jointly represented by source-train
   interactions.

Work on an improved target diagnostic was started and then deliberately paused.
No new result has been promoted. The formal result files and headline remain
unchanged.

The model-independent implementation has now resumed locally:

1. Primary profiles now declare required evidence blocks. Missing P or E causes
   an explicit unavailable status and conservative fallback instead of silently
   fitting a nominal U+P or U+P+E profile with only U features.
2. The runner writes a capability manifest describing whether U, P, clean E,
   direct target support, and joint support are available for each input.
3. A single non-primary diagnostic named `general_target_joint_support` has
   been implemented. It adds exactly three fixed quantities after clean E:
   direct target-to-source nearest distance, fixed top-5 target density, and
   distance to the closest observed similar source-train interaction.
4. The joint measure uses Morgan/Tanimoto drug neighbours, direct MMseqs2
   target neighbours, and source-train pair occurrence. The exact queried pair
   is excluded, and no outcome label is used.
5. The implementation also produces per-model/per-dataset strata,
   leave-one-dataset-out, and leave-one-model-out diagnostic outputs.

The new builders are:

```text
audit_framework/scripts/build_direct_target_support.py
audit_framework/scripts/build_joint_support.py
```

At that stage the local implementation passed 22 tests. A
PACE/BioSNAP/seed-4 target-grouped
smoke audit reproduced all four existing selected profiles and AUPRC values
exactly; the maximum numeric difference from the frozen formal result was zero.
Without new assets, the diagnostic is explicitly reported as unavailable.
The temporary smoke output was removed, so `audit_framework/results/audit/`
remains the only result directory.

The direct and joint assets have now been generated locally with the official
MMseqs2 container `ghcr.io/soedinglab/mmseqs2:latest`, reporting version
`8cc5ce367b5638c4306c2d7cfc652dd099a4643f`. The fixed MMseqs2 settings use the
default sensitivity and E-value behaviour with `--max-seqs 5`; no threshold
search was performed.

Direct target-to-source coverage:

| Dataset | Query targets | Unique source targets | Targets with hits |
|---|---:|---:|---:|
| BioSNAP | 2,150 | 1,279 | 1,850 |
| BindingDB | 1,926 | 1,231 | 1,804 |
| Human | 1,503 | 1,233 | 1,403 |

Frozen joint-support asset coverage:

| Dataset | Drugs | Targets | Unique source pairs |
|---|---:|---:|---:|
| BioSNAP | 4,162 | 2,150 | 9,765 |
| BindingDB | 12,167 | 1,926 | 14,928 |
| Human | 1,813 | 1,503 | 3,059 |

The asset metadata records `outcome_labels_used=false` and
`exact_query_pair_excluded=true`. The joint distance is sparse, especially for
Human, but its fixed top-5 definition has not been changed after observing this.

A completed PACE/BioSNAP/seed-4 smoke audit is stored under:

```text
audit_framework/cache/diagnostic_results/general_target_joint_smoke/
```

Across its four policies, the diagnostic increment after clean E was:

| Grouping | Mean validation fold-AUPRC increment | Mean test AUPRC increment |
|---|---:|---:|
| target | -0.019152 | +0.068754 |
| drug | +0.003412 | +0.064048 |

The target validation/test directions disagree. This smoke result is therefore
a warning against test-driven promotion, not evidence that the diagnostic is
effective. The five primary profiles and their selected metrics remained
exactly equal to the frozen formal result in this smoke run.

An attempted monolithic full diagnostic run was interrupted because it exceeded
the practical execution time. Its directory:

```text
audit_framework/cache/diagnostic_results/general_target_joint_full/
```

contains only `missing_inputs.csv` and `profile_manifest.csv`. It is incomplete
and must not be analysed or reported. The residual Python process was confirmed
and stopped on 2026-06-15.

Frozen DrugBAN/TAPB checkpoints are not prerequisites for the generic
diagnostic. They are needed only for a later optional predictor-aware
representation ablation. Cached scalar scores cannot reconstruct model-native
representations.

## 10. Next Work

The fixed `general_target_joint_support` experiment is complete and is a
negative/non-promotable diagnostic result. Do not tune its thresholds, change
its three features, or rerun variants selected from the observed test results.

Next work, if pursued, must remain separate from the frozen formal core:

1. Keep `general_target_joint_support` non-primary and preserve the completed
   shards and merged run-level outputs.
2. Preserve the completed equal-budget, cross-domain transfer, strong-baseline,
   and validation-only falsification outputs as diagnostic evidence.
3. Optionally test frozen predictor-native target representations as a clearly
   labelled predictor-aware ablation. Checkpoints must not become requirements
   for the generic PAUSE path.
4. Any new target-E hypothesis must be predeclared before grouped validation
   and must not reuse the current test outcomes for feature or threshold choice.
5. Do not modify the five primary profiles, formal headline, manuscript, or
   `audit_framework/results/audit/` based on this diagnostic.

The clean U+P+E core and its completed sensitivities remain frozen.

## 11. Commands

Build or refresh native source support:

```bash
python audit_framework/scripts/build_native_support.py \
  --datasets biosnap bindingdb human \
  --split cluster
```

Build or refresh fixed MMseqs2 target support:

```bash
python audit_framework/scripts/build_target_sequence_support.py \
  --datasets biosnap bindingdb human \
  --split cluster
```

Run the formal cluster audit:

```bash
python audit_framework/scripts/run_audit.py \
  --group-axes target drug \
  --out-dir audit_framework/results/audit
```

Run tests:

```bash
python -m unittest audit_framework.tests.test_audit
```

The fixed generic target and joint-support assets have already been built.
Rebuild only when provenance verification requires it:

```bash
python audit_framework/scripts/build_direct_target_support.py \
  --datasets biosnap bindingdb human \
  --split cluster \
  --threads 32 \
  --docker-image ghcr.io/soedinglab/mmseqs2:latest

python audit_framework/scripts/build_joint_support.py \
  --datasets biosnap bindingdb human \
  --split cluster
```

Do not use the previous monolithic diagnostic command. Run separate
model-dataset shards into non-formal cache directories, then merge run-level
outputs and summarize once.

Example for one shard only:

```bash
python audit_framework/scripts/run_audit.py \
  --test-roots audit_framework/cache/test_audits/pace \
  --datasets biosnap \
  --seeds 4 5 6 7 8 \
  --group-axes target drug \
  --bootstrap-resamples 200 \
  --out-dir \
    audit_framework/cache/diagnostic_results/general_target_joint_shards/pace/biosnap
```

The resumable orchestration command is now:

```bash
python audit_framework/scripts/run_audit_shards.py \
  --shard-bootstrap-resamples 20 \
  --final-bootstrap-resamples 20000
```

It skips already complete shards after revalidation, rejects missing or
duplicate policy keys, merges run-level rows deterministically, and performs
the 20,000-resample bootstrap only on the unified data.

Run or resume the three strengthening experiments:

```bash
python audit_framework/scripts/run_strengthening_experiments.py \
  --bootstrap-resamples 20000
```

This command validates and skips complete model-dataset shards, evaluates four
equal review budgets, runs the fixed strong baselines and validation-only
feature-block falsification, merges all nine shards, and then runs the
cross-domain transfer sensitivity.

## 12. Completed General Target/Joint Diagnostic

The nine fixed model-dataset shards are complete:

```text
3 models x 3 datasets x 5 seeds x 2 groupings x 4 policies
```

Each shard contains 480 candidate rows and 1,360 component-increment rows.
Every shard was validated before the next shard started. The merger rejects
missing, duplicate, or incomplete Cartesian policy coverage and does not
average shard-level confidence intervals.

Shard outputs:

```text
audit_framework/cache/diagnostic_results/general_target_joint_shards/
```

Merged run-level outputs and unified 20,000-resample summaries:

```text
audit_framework/cache/diagnostic_results/general_target_joint_merged/
```

The primary comparison is the fixed diagnostic after clean E:

```text
uncertainty_support -> general_target_joint_support
```

Overall AUPRC increments:

| Grouping | Validation fold-AUPRC increment | Test AUPRC increment |
|---|---:|---:|
| target | +0.007414 [-0.033718,+0.046769] | +0.031946 [+0.004573,+0.066628] |
| drug | +0.021145 [-0.009311,+0.054355] | +0.037167 [+0.011891,+0.068332] |

The target validation interval crosses zero. Among the 39 model-dataset-seed
target pairs with finite validation and test estimates, only 56.4% have the
same direction and 46.2% are positive on both validation and test. Drug
direction is more consistent (75.0% same direction; 66.7% both positive), but
its validation interval also crosses zero.

Target validation is heterogeneous:

| Stratum | Mean increment |
|---|---:|
| DrugBAN | +0.015526 |
| PACE | +0.005975 |
| TAPB | -0.006421 |
| BindingDB | +0.007555 |
| BioSNAP | +0.032973 |
| Human | -0.036937 |
| seed 4 | -0.028055 |
| seed 5 | -0.026357 |
| seed 6 | +0.002510 |
| seed 7 | +0.022301 |
| seed 8 | +0.041996 |

Target leave-one-dataset-out validation increments:

| Excluded dataset | Increment | 95% interval |
|---|---:|---:|
| BioSNAP | -0.005366 | -0.059622 to +0.047933 |
| BindingDB | +0.009750 | -0.051578 to +0.066484 |
| Human | +0.017857 | -0.009820 to +0.049943 |

Target leave-one-model-out validation increments:

| Excluded model | Increment | 95% interval |
|---|---:|---:|
| DrugBAN | -0.003076 | -0.057978 to +0.044259 |
| PACE | +0.009511 | -0.049309 to +0.065128 |
| TAPB | +0.015806 | -0.019650 to +0.057741 |

Human exact-pair-deduplicated test increments:

| Grouping | Increment | 95% interval |
|---|---:|---:|
| target | -0.004976 | -0.029807 to +0.012823 |
| drug | +0.018065 | -0.007555 to +0.050433 |

The target effect therefore disappears after Human pair deduplication. The
drug-grouped aggregate is not harmed on average, but drug validation remains
uncertain and heterogeneous; for example, its validation interval crosses zero
and seed 7 has a negative mean.

### Promotion Decision

`general_target_joint_support` must remain non-primary:

- target validation is not stably positive;
- validation/test direction is inconsistent for many runs;
- the validation result depends materially on model, dataset, and seed;
- leave-one-dataset-out and leave-one-model-out validation are not robust;
- Human target pair-deduplicated sensitivity is negative on average;
- the increment beyond clean E is not reliably positive;
- drug grouping is positive in aggregate but not uniformly protected.

The diagnostic did satisfy the design constraint of adding only one fixed
profile with no model-specific or dataset-specific threshold search. That is
not enough for promotion.

The five primary profiles reproduce the frozen formal run: all 360 selection
choices are identical, and primary candidate/test metrics differ by at most
`5e-7`, attributable to CSV rounding. `audit_framework/results/audit/` and the
manuscript were not modified.

## 13. Completed Strengthening Experiments

The predeclared protocol is:

```text
audit_framework/experiments/strengthening_protocol.md
```

All diagnostic outputs are below:

```text
audit_framework/cache/diagnostic_results/strengthening_experiments/
```

The nine model-dataset shards are complete and contain 25,920 candidate budget
rows, 1,440 validation-selected budget rows, and 1,440 validation-only
falsification rows. At review fraction 0.20, all 360 selected profiles match
the frozen formal run and metrics differ by at most `5e-7`.

### Equal Total Action Budget

PAUSE and calibrated uncertainty receive identical candidate, defer, and
review counts. Combined candidate-error recall gains are:

| Grouping | Review fraction | Gain | 95% interval |
|---|---:|---:|---:|
| drug | 0.05 | +0.0498 | +0.0201 to +0.0770 |
| drug | 0.10 | +0.0822 | +0.0362 to +0.1225 |
| drug | 0.20 | +0.1092 | +0.0484 to +0.1652 |
| drug | 0.30 | +0.1186 | +0.0577 to +0.1731 |
| target | 0.05 | +0.0210 | +0.0060 to +0.0382 |
| target | 0.10 | +0.0211 | +0.0016 to +0.0422 |
| target | 0.20 | +0.0326 | +0.0051 to +0.0631 |
| target | 0.30 | +0.0345 | +0.0071 to +0.0617 |

The normalized action-curve area gain is +0.0969
[+0.0455,+0.1439] for drug grouping and +0.0283
[+0.0054,+0.0536] for target grouping. Review error rate and retained
accuracy gains are positive at all four budgets.

Human remains the main boundary. At review fraction 0.20, combined recall gains
are +0.1643 for BioSNAP, +0.1538 for BindingDB, and +0.0044 for Human under
drug grouping. Human pair-deduplicated intervals cross zero at every budget.

### Cross-Domain Profile Transfer

Source-domain validation selects `uncertainty_support` for every drug-grouped
leave-one-dataset-out and leave-one-model-out analysis. Held-out test AUPRC
gains are:

| Held-out domain | Gain | 95% interval |
|---|---:|---:|
| BindingDB | +0.2461 | +0.1387 to +0.3581 |
| BioSNAP | +0.2312 | +0.1650 to +0.2976 |
| Human | +0.0487 | -0.0581 to +0.1690 |
| DrugBAN | +0.2634 | +0.1009 to +0.3658 |
| PACE | +0.1134 | -0.0545 to +0.2592 |
| TAPB | +0.1673 | +0.1004 to +0.2227 |

Drug-domain support is therefore transferable across several held-out domains,
although Human and PACE remain uncertain.

Target transfer is weaker. The source domains select `prior` except when
BioSNAP is held out, where they select `uncertainty_prior_support`. BioSNAP is
strongly positive, but all other target transfer intervals cross zero. Human
target combined recall is negative on average and remains negative after exact
pair deduplication.

### Strong Baselines and Falsification

Raw uncertainty and scalar native shift/density scores are significantly worse
than PAUSE. The fixed HGB(U+P+E) baseline is higher on average but not
significantly:

| Grouping | HGB AUPRC minus PAUSE | 95% interval |
|---|---:|---:|
| drug | +0.0130 | -0.0647 to +0.0963 |
| target | +0.1264 | -0.0233 to +0.2533 |

HGB does not establish dominance, but its positive target mean identifies
ranker choice as a plausible future target-domain improvement.

Validation-only feature-block permutation gives actual U+P+E minus permuted
fold AUPRC:

| Grouping | Control | Difference | 95% interval |
|---|---|---:|---:|
| drug | permute P | +0.0197 | -0.0268 to +0.1513 |
| drug | permute E | +0.1694 | +0.1185 to +0.2159 |
| drug | permute P+E | +0.2279 | +0.1823 to +0.3326 |
| target | permute P | +0.0301 | -0.0025 to +0.1140 |
| target | permute E | +0.0653 | -0.0050 to +0.1964 |
| target | permute P+E | +0.1382 | +0.0594 to +0.3120 |

The drug result contains clear information beyond U and is driven mainly by E.
For target grouping, the joint P+E block matters, but neither block alone is
stable.

These experiments strengthen the paper around equal-budget error capture and
transferable drug-domain support. They do not justify claiming that cold-target
auditing is solved, changing the five primary profiles, or modifying formal E.
The manuscript and `audit_framework/results/audit/` remain unchanged.

The test suite now contains 28 passing tests: 24 core/sharding tests and four
strengthening-experiment tests.

## 14. Ablation Inventory and Manuscript Readiness

### Existing Ablation Evidence

PAUSE already has sufficient experimental material for a rigorous ablation
section. The evidence is distributed across formal component increments,
diagnostic profiles, and the strengthening experiments rather than being
presented in one manuscript-ready table.

The completed ablations are:

1. Primary profile decomposition:

   ```text
   U
   P
   U+P
   U+E
   U+P+E
   ```

   This supports direct comparisons of P versus U, E versus U, E after P, and
   P after E without adding profiles after observing test results.

2. Clean E decomposition:

   - grouped-fold drug/target entity support;
   - frozen native drug support from Morgan/Tanimoto;
   - frozen native target support from ESM/cosine;
   - fold-fit and native source-domain support together;
   - legacy source counts and prior-space distances as redundancy controls.

   The formal validation results show that drug-grouped E adds substantial
   information beyond U and after P. Target E is positive on average but has
   broad validation intervals.

3. Feature-block permutation:

   - grouped-fold permutation of P;
   - grouped-fold permutation of E;
   - grouped-fold permutation of P and E together.

   Drug-grouped performance clearly deteriorates when E is permuted and when
   P+E are both permuted. P alone is not independently stable. For target
   grouping, the joint P+E block is supported, but neither individual block has
   a confidence interval excluding zero.

4. Negative and non-promotable diagnostics:

   - MMseqs2 target sequence support;
   - direct target and drug-conditioned joint source-pair support;
   - legacy E definitions;

   These controls do not justify expanding the five selectable profiles.

5. Comparator and robustness analyses:

   - raw frozen-predictor uncertainty;
   - calibrated uncertainty U;
   - scalar native shift and density scores;
   - fixed HGB(U+P+E);
   - equal total action budgets;
   - leave-one-dataset-out and leave-one-model-out transfer;
   - Human exact-pair deduplication;
   - model, dataset, and seed heterogeneity.

Equal-budget and transfer results are robustness and operational validation,
not feature ablations. They should be reported separately from the primary
profile and permutation ablations.

### Current Scientific Claim

The strongest evidence-supported claim is:

> Under the same defer-plus-review action budget, a validation-selected PAUSE
> audit captures more high-confidence frozen-predictor errors than uncertainty
> alone, with the strongest and most transferable evidence in drug-grouped
> evaluation.

The supporting chain is:

1. uncertainty removes an important first layer of ambiguous predictions;
2. residual high-confidence errors remain after uncertainty deferment;
3. PAUSE captures more of those errors at identical total action cost across
   four review budgets;
4. drug-grouped U+E transfers across several held-out datasets and predictors;
5. E permutation substantially degrades drug-grouped validation performance;
6. PAUSE is clearly stronger than raw uncertainty and scalar OOD rankings;
7. the formal five-profile selector remains conservative and unchanged.

The current evidence does not support the stronger claims that:

- cold-target auditing is solved;
- every P and E subcomponent is independently necessary;
- PAUSE dominates every flexible supervised failure detector;
- Human alone provides stable pair-deduplicated evidence;
- the new general target/joint diagnostic should enter the core.

### BIBM Draft Mismatch

The current files under `manuscript/bibm/` are an obsolete initial draft and
must not be treated as a description of the completed framework. In particular,
the draft still centers:

- auditor-risk and fixed auditor scalars;
- M/S and structural-response features;
- an older score-family selection procedure;
- claims that prior/conflict is the dominant component;
- result tables and policy summaries from the pre-U+P+E framework.

Those statements conflict with the frozen implementation and current evidence,
which support a core U+P+E framework driven most clearly by clean E in
drug-grouped evaluation. The manuscript therefore requires a structural
rewrite of Methods, Results, Abstract, and Discussion rather than local number
replacement. The old draft may be used only as a formatting and citation
starting point.

### Start-Writing Decision

PAUSE is ready to enter manuscript writing now. Further algorithm or feature
optimization is not a prerequisite for beginning the paper.

The reason is that the central operational claim already has:

- three frozen predictor families;
- three datasets;
- five seeds;
- strict target- and drug-grouped validation;
- four fixed policies;
- conservative validation-only profile selection;
- equal-budget effects with positive 95% intervals at four review budgets;
- cross-domain transfer;
- primary and internal E ablations;
- validation-only feature permutation;
- strong comparator models;
- Human pair-deduplicated sensitivity;
- explicit negative results and promotion rules.

Continuing to tune target features before writing would create greater
selection freedom, delay the paper, and risk weakening the methodological
discipline. The weak target and Human results should be reported as boundaries
rather than hidden or optimized away.

### Required Before Submission, Not Before Drafting

The following work is mandatory for a submission-quality manuscript but can be
performed while writing:

1. Rebuild the manuscript argument around equal-budget frozen-predictor audit,
   not the obsolete M/S story.
2. Produce one manuscript-ready primary ablation matrix containing U, P, U+P,
   U+E, U+P+E, and validation-selected PAUSE for validation and test under both
   grouping axes.
3. Report P/E permutation separately from the primary profile ablation.
4. Complete the HGB heterogeneity audit using existing outputs:
   validation/test direction, model/dataset/seed strata, leave-one-domain
   sensitivity, and Human deduplication. This requires analysis only, not new
   model tuning.
5. Generate the equal-action budget curve, cross-domain transfer figure, and
   compact robustness/limitation table.
6. Build a claim-to-result matrix so every abstract and conclusion statement
   has a frozen output source and an allowed strength.
7. Run a manuscript-level integrity and reviewer audit after the rewrite.

### Optional Work

The following work is optional and should not block manuscript drafting:

- predictor-native target representation ablation;
- additional external datasets;
- prospective biological follow-up;
- alternative fixed rankers beyond the already tested HGB;
- new target-E hypotheses.

Any optional target-E experiment must be predeclared and remain separate from
the frozen formal core. It should be pursued only if the manuscript rewrite
reveals a specific reviewer-critical gap and sufficient time remains.

### Writing Priority

The recommended immediate order is:

1. freeze the current experimental claims and tables;
2. complete HGB stability analysis from existing outputs;
3. build manuscript figures and the primary ablation table;
4. rewrite Methods and Results from the frozen implementation;
5. rewrite Abstract, Introduction, and Discussion around the supported claim;
6. run mock review, statistical consistency, citation, and reproducibility
   checks.

Decision: start writing immediately. Do not wait for another round of target
feature optimization.

## 15. Repository Simplification

The repository was simplified on 2026-06-15 so that the implementation matches
the frozen scientific claim instead of carrying an inactive second system.

Removed from active code:

```text
models/pause_auditor_model.py
models/mechanism_verifier.py
models/mechanism_verifier_impl.py
data_loader/pime_struct_dataset.py
data_process/pime_struct/
tools/train_struct_pretrain.py
diagnostics/prior_aware_intervention_audit.py
diagnostics/intervention_controls.py
diagnostics/model_randomization_diagnostics.py
```

The six local-evidence construction scripts under `data_process/pime/` were
also removed. `models/pause_model.py` now exposes only the base predictor and
the explicit prior channel. `diagnostics/export_audit_inputs.py` exports only
the frozen base/prior inputs consumed by U+P+E.

Removed physical assets:

```text
checkpoints/
datasets/<dataset>/cluster/pime/drug_fragment_*
datasets/<dataset>/cluster/pime/target_pocket_*
datasets/<dataset>/cluster/pime/p2rank_*
datasets/<dataset>/cluster/pime/pocket_residues_p2rank.csv
datasets/<dataset>/cluster/pime/target_structure_meta.csv
datasets/<dataset>/cluster/pime/structures/
```

The deletion removed approximately 1.25 GiB. The three evidence manifests were
rebuilt as `pause-evidence-store-v2` and report complete core coverage.

Historical test/validation input CSV values were preserved, but their active
paths were renamed to:

```text
audit_framework/cache/test_audits/{pace,tapb,drugban}/
audit_framework/cache/validation_audits/{pace,tapb,drugban}/
```

The input filename is now `pause_audit_inputs.csv`. The 90 cached input files
were reduced from 63 historical columns to the 20 base/prior and provenance
columns consumed by U+P+E, preserving 208,410 rows. Obsolete per-input
extension summaries were removed. No formal result value was overwritten
during this cleanup.

Verification after deletion:

- 28 tests pass in the `pytorch-2.8.0-gpu` environment;
- all active Python modules compile;
- all three datasets contain zero retired local-evidence assets;
- `checkpoints/` and `data_process/pime_struct/` do not exist;
- the manifest still contains exactly five primary profiles and seven fixed
  non-primary E diagnostics;
- all 45 model-dataset-seed input pairs are discovered with zero missing;
- a fresh PACE/BioSNAP/seed-4 target-grouped smoke reproduced 40 primary
  candidate rows with maximum numeric difference `0.0` and all four selection
  choices exactly equal to the frozen formal result;
- `audit_framework/results/audit/` and `manuscript/` were not modified.
