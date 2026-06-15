# PAUSE Strengthening Experiment Analysis

Completed: 2026-06-15

All nine model-dataset shards completed and passed exact Cartesian-product and
duplicate-key validation. The merged assets contain:

- 25,920 candidate budget rows;
- 1,440 validation-selected budget rows;
- 1,440 validation-only falsification rows.

At review fraction 0.20, all 360 selected profiles exactly match the frozen
formal audit. Metric differences are at most `5e-7` from CSV precision.

## 1. Equal total action budget

PAUSE and calibrated uncertainty use identical candidate, defer, and review
counts. PAUSE improves combined candidate-error recall at every review budget:

| Grouping | Review fraction | Recall gain | 95% interval |
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
accuracy gains are also positive at all four budgets.

The result is heterogeneous by dataset. At review fraction 0.20, combined
recall gains are +0.1643 for BioSNAP, +0.1538 for BindingDB, and +0.0044 for
Human under drug grouping. Target gains are +0.0497, +0.0386, and +0.0075.
Every seed-level mean remains positive, but Human pair-deduplicated intervals
cross zero at all budgets.

## 2. Cross-domain profile transfer

Source-domain validation always selects U+E for drug grouping. Its held-out
test AUPRC gains are:

| Held-out domain | Gain | 95% interval |
|---|---:|---:|
| BindingDB | +0.2461 | +0.1387 to +0.3581 |
| BioSNAP | +0.2312 | +0.1650 to +0.2976 |
| Human | +0.0487 | -0.0581 to +0.1690 |
| DrugBAN | +0.2634 | +0.1009 to +0.3658 |
| PACE | +0.1134 | -0.0545 to +0.2592 |
| TAPB | +0.1673 | +0.1004 to +0.2227 |

This supports a transferable drug-domain support signal, although Human and
PACE remain uncertain.

Target grouping is not comparably transferable. The source domains select P
except when BioSNAP is held out, where they select U+P+E. The BioSNAP gain is
strong (+0.2163 [+0.1527,+0.2807]), but all other target transfer intervals
cross zero. Held-out Human combined recall is negative on average, and its
pair-deduplicated target result is also negative on average.

## 3. Strong baselines and falsification

At review fraction 0.20, raw uncertainty and the two scalar native OOD
baselines are significantly worse than PAUSE for both groupings. The fixed
HGB(U+P+E) baseline is slightly higher on average:

| Grouping | HGB AUPRC minus PAUSE | 95% interval |
|---|---:|---:|
| drug | +0.0130 | -0.0647 to +0.0963 |
| target | +0.1264 | -0.0233 to +0.2533 |

The intervals cross zero, so HGB does not establish dominance. Its positive
target mean is nevertheless an important warning that ranker choice remains a
plausible source of target-domain improvement. HGB was fixed in advance and
did not enter profile selection.

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

Drug performance therefore contains clear information beyond U, driven mainly
by E. For target grouping, the joint P+E block matters, but neither block alone
is stable enough to support a strong decomposition claim.

## Decision

These experiments materially strengthen the operational paper story:

1. PAUSE captures more errors than uncertainty at the same action cost across
   four budgets.
2. Drug-domain U+E transfers across held-out datasets and predictors.
3. The main drug signal survives a direct feature-block falsification.
4. PAUSE is clearly stronger than raw uncertainty and scalar OOD scores.

They do not establish a solved cold-target audit:

- Human pair-deduplicated effects remain uncertain;
- most target transfer intervals cross zero;
- fixed HGB has a positive but uncertain target advantage;
- P beyond clean E is not independently stable under permutation.

The five formal profiles and U+P+E core should remain frozen. The strongest
paper claim should center on equal-budget error capture and transferable
drug-domain support, with target auditing reported as a weaker boundary and
open problem.
