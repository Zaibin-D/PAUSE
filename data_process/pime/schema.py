from dataclasses import dataclass
from enum import Enum


class ChannelPermission(str, Enum):
    PRIOR = "prior"


class LeakageRisk(str, Enum):
    LOW = "low"


SPLIT_FILES = {
    "cluster": {
        "train": "source_train_with_id.csv",
        "val": "target_train_with_id.csv",
        "test": "target_test_with_id.csv",
    },
}


PIME_FILES = {
    "drug_entity": "drug_entity.csv",
    "target_entity": "target_entity.csv",
    "drug_prior_feat": "drug_prior_feat.pkl",
    "drug_prior_meta": "drug_prior_meta.json",
    "target_prior_feat": "target_prior_feat.pkl",
    "target_prior_meta": "target_prior_meta.json",
    "manifest": "pime_manifest.json",
    "audit_report": "pime_audit_report.md",
    "audit_stats": "pime_audit_stats.json",
}


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    channel_permission: ChannelPermission
    leakage_risk: LeakageRisk
    source: str
    description: str


FEATURE_SPECS = [
    FeatureSpec(
        name="drug_prior_feat",
        channel_permission=ChannelPermission.PRIOR,
        leakage_risk=LeakageRisk.LOW,
        source="RDKit global descriptors and fingerprints",
        description="Non-label global chemical evidence for the P channel.",
    ),
    FeatureSpec(
        name="target_prior_feat",
        channel_permission=ChannelPermission.PRIOR,
        leakage_risk=LeakageRisk.LOW,
        source="Frozen protein embeddings and sequence composition",
        description="Non-label target-level evidence for the P channel.",
    ),
]


def feature_spec_map():
    return {spec.name: spec for spec in FEATURE_SPECS}
