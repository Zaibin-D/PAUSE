import argparse

from data_process.pime.audit_coverage import audit_coverage
from data_process.pime.build_drug_properties import build_drug_properties
from data_process.pime.build_entity_registry import build_entity_registry
from data_process.pime.build_target_prior import build_target_prior


def build_evidence_store(dataset, split):
    build_entity_registry(dataset, split)
    build_drug_properties(dataset, split)
    build_target_prior(dataset, split)
    audit_coverage(dataset, split)


def main():
    parser = argparse.ArgumentParser(
        description="Build the PAUSE entity registry and prior evidence store."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="cluster", choices=["cluster"])
    args = parser.parse_args()
    build_evidence_store(args.dataset, args.split)


if __name__ == "__main__":
    main()
