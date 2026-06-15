# PAUSE

PAUSE is a policy-aligned residual audit framework for frozen drug-target
interaction predictors. Its frozen formal core is U+P+E under strict grouped
validation.

The canonical project record is
[`docs/project_status.md`](docs/project_status.md).

## Project Layout

```text
audit_framework/   formal audit library, sharding, tests, cache, and results
diagnostics/       frozen base/prior input export
models/            base adapters, PAUSE prior head, and pretrained encoders
data_loader/       base and prior data loading
data_process/      entity, prior, and label-free support construction
config_files/      local and external-predictor configurations
external_models/   TAPB and DrugBAN code, data, and predictor checkpoints
datasets/          BindingDB, BioSNAP, and Human assets
docs/              canonical project-status document
manuscript/        isolated manuscript sources
```

## Formal Audit

```bash
python audit_framework/scripts/run_audit.py \
  --group-axes target drug \
  --out-dir audit_framework/results/audit
```

The five selectable profiles are fixed: U, P, U+P, U+E, and U+P+E. Additional
target-support profiles are diagnostics only.

## Tests

```bash
python -m unittest \
  audit_framework.tests.test_audit \
  audit_framework.tests.test_strengthening
```

## Training

```bash
python main.py --help
```

The training entry point exposes only base and prior stages on the cluster
split. PAUSE has no separate local-evidence checkpoint path.
