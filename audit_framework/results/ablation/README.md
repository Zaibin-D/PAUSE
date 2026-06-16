# PAUSE Primary Ablation Matrix

Cells show absolute residual error-detection AUPRC followed by delta versus U. Confidence intervals are 20,000-resample stratified bootstraps over model-dataset strata after averaging the four policies within each model-dataset-seed-axis cluster.

| Profile                   | Drug validation                         | Drug test                               | Target validation                       | Target test                             |
|:--------------------------|:----------------------------------------|:----------------------------------------|:----------------------------------------|:----------------------------------------|
| U                         | 0.310 (ref.)                            | 0.319 (ref.)                            | 0.323 (ref.)                            | 0.319 (ref.)                            |
| P                         | 0.380 (+0.069; 95% CI +0.008 to +0.148) | 0.340 (+0.028; 95% CI -0.019 to +0.082) | 0.380 (+0.071; 95% CI -0.042 to +0.184) | 0.340 (+0.028; 95% CI -0.017 to +0.080) |
| U+P                       | 0.371 (+0.061; 95% CI +0.001 to +0.137) | 0.340 (+0.028; 95% CI -0.020 to +0.082) | 0.380 (+0.071; 95% CI -0.040 to +0.182) | 0.340 (+0.028; 95% CI -0.017 to +0.080) |
| U+E                       | 0.524 (+0.214; 95% CI +0.140 to +0.299) | 0.485 (+0.173; 95% CI +0.085 to +0.260) | 0.372 (+0.063; 95% CI -0.084 to +0.192) | 0.486 (+0.174; 95% CI +0.088 to +0.258) |
| U+P+E                     | 0.523 (+0.212; 95% CI +0.134 to +0.303) | 0.478 (+0.166; 95% CI +0.079 to +0.253) | 0.417 (+0.109; 95% CI -0.070 to +0.286) | 0.489 (+0.177; 95% CI +0.093 to +0.260) |
| Validation-selected PAUSE | 0.523 (+0.212; 95% CI +0.145 to +0.315) | 0.467 (+0.148; 95% CI +0.066 to +0.240) | 0.494 (+0.171; 95% CI +0.057 to +0.325) | 0.361 (+0.042; 95% CI +0.007 to +0.082) |

## Compact Screenshot-Style Matrix

| Profile                   | Validation          | Test                | Drug-grouped test ? vs U   | Target-grouped test ? vs U   |
|:--------------------------|:--------------------|:--------------------|:---------------------------|:-----------------------------|
| U                         | 0.311 (+0.000 vs U) | 0.316 (+0.000 vs U) | +0.000 [+0.000,+0.000]     | +0.000 [+0.000,+0.000]       |
| P                         | 0.376 (+0.073 vs U) | 0.341 (+0.030 vs U) | +0.028 [-0.019,+0.082]     | +0.028 [-0.017,+0.080]       |
| U+P                       | 0.373 (+0.070 vs U) | 0.341 (+0.030 vs U) | +0.028 [-0.020,+0.082]     | +0.028 [-0.017,+0.080]       |
| U+E                       | 0.445 (+0.141 vs U) | 0.489 (+0.178 vs U) | +0.173 [+0.085,+0.260]     | +0.174 [+0.088,+0.258]       |
| U+P+E                     | 0.457 (+0.154 vs U) | 0.488 (+0.177 vs U) | +0.166 [+0.079,+0.253]     | +0.177 [+0.093,+0.260]       |
| Validation-selected PAUSE | 0.498 (+0.187 vs U) | 0.409 (+0.093 vs U) | +0.148 [+0.066,+0.240]     | +0.042 [+0.007,+0.082]       |

## Output Files

- `primary_ablation_matrix_manuscript.csv`
- `primary_ablation_matrix_numeric.csv`
- `primary_ablation_matrix_compact.csv`
- `primary_ablation_summary_long.csv`
- `primary_ablation_cluster_runs.csv`
- `primary_ablation_matrix.xlsx`
