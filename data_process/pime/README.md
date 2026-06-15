# PAUSE Evidence Store

This package builds the non-label entity and prior inputs consumed by PAUSE.
The formal audit uses the frozen U+P+E profile library and cluster split only.

## Core Build

```bash
python -m data_process.pime.build_evidence_store \
  --dataset human \
  --split cluster
```

The core builder creates:

```text
datasets/{dataset}/cluster/pime/
  drug_entity.csv
  target_entity.csv
  drug_prior_feat.pkl
  drug_prior_meta.json
  drug_prior_table.csv
  target_prior_feat.pkl
  target_prior_meta.json
  target_prior_table.csv
  pime_manifest.json
  pime_audit_report.md
  pime_audit_stats.json
```

The audit framework separately builds label-free source support:

```text
native_source_support.pkl
target_sequence_support.pkl
direct_target_support.pkl
joint_source_support.pkl
mmseqs/
```

Outcome labels are not used to construct any support asset. The exact queried
drug-target pair is excluded from the fixed joint-support diagnostic.
