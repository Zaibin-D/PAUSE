from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import pickle
import unittest

import numpy as np
import pandas as pd
from audit_framework.data import (
    attach_source_support,
    compute_direct_target_support_payload,
    compute_target_sequence_support_payload,
    joint_nearest_train_distance,
    source_support_provenance,
)
from audit_framework.scripts.run_audit import (
    _component_increment_rows,
    _pair_deduplicated_metrics,
)
from audit_framework.sharding import (
    ShardSpec,
    ShardValidationError,
    _expected_comparisons,
    validate_shard,
)
from configs import Config
from data_loader.data_loader import PIMEDTIDataset, pime_collate_fn
from audit_framework import (
    EVIDENCE_DIAGNOSTIC_PROFILES,
    PRIMARY_PROFILES,
    apply_uncertainty_policy,
    audit_capabilities,
    calibrated_error_risk,
    calibration_metrics,
    cross_fitted_profile,
    engineer_features,
    fit_profile_for_test,
    profile_manifest,
    select_one_standard_error,
)


class AuditFrameworkTests(unittest.TestCase):
    def _write_minimal_complete_shard(self, directory: str) -> ShardSpec:
        out_dir = Path(directory) / "pace" / "biosnap"
        out_dir.mkdir(parents=True)
        spec = ShardSpec(
            model_key="pace",
            model_name="PACE",
            dataset="biosnap",
            test_root=(
                "audit_framework/cache/test_audits/"
                "pace"
            ),
            out_dir=out_dir,
        )
        policy = {
            "model": "PACE",
            "dataset": "biosnap",
            "test_run": "seed_4",
            "group_axis": "target",
            "confidence_source": "base",
            "confidence_threshold": 0.8,
            "uncertainty_reject_fraction": 0.1,
            "review_fraction": 0.2,
        }
        manifest = profile_manifest()
        manifest.to_csv(out_dir / "profile_manifest.csv", index=False)
        pd.DataFrame(
            columns=[
                "test_root",
                "dataset",
                "test_run",
                "reason",
                "expected_validation",
            ]
        ).to_csv(out_dir / "missing_inputs.csv", index=False)

        candidates = [
            {
                **policy,
                "profile": profile.profile,
                "primary_candidate": profile.primary_candidate,
                "model_status": (
                    "group_cv_unavailable"
                    if profile.profile == "general_target_joint_support"
                    else "complete"
                ),
            }
            for profile in manifest.itertuples(index=False)
        ]
        for name in (
            "candidate_validation_runs.csv",
            "candidate_test_runs.csv",
        ):
            pd.DataFrame(candidates).to_csv(out_dir / name, index=False)
        pd.DataFrame(
            [
                {**policy, "split": "validation", "residual_n": 1},
                {**policy, "split": "test", "residual_n": 1},
            ]
        ).to_csv(out_dir / "policy_coverage.csv", index=False)
        pd.DataFrame(
            [
                {
                    "model": "PACE",
                    "dataset": "biosnap",
                    "test_run": "seed_4",
                    "split": split,
                    "uncertainty_available": True,
                    "prior_available": True,
                    "empirical_support_available": True,
                    "native_support_available": True,
                    "direct_target_support_available": True,
                    "joint_support_available": True,
                }
                for split in ("validation", "test")
            ]
        ).to_csv(out_dir / "capability_manifest.csv", index=False)
        calibration = [
            {
                **policy,
                "split": split,
                "scope": scope,
                "score_source": score_source,
            }
            for split in ("validation", "test")
            for scope in ("candidate", "deferred", "residual")
            for score_source in ("raw", "calibrated")
        ]
        pd.DataFrame(calibration).to_csv(
            out_dir / "calibration_diagnostics.csv",
            index=False,
        )
        pd.DataFrame(
            [
                {
                    **policy,
                    "split": "test",
                    "support_reference": "audit_fit",
                    "support_state": "all",
                }
            ]
        ).to_csv(out_dir / "domain_support_diagnostics.csv", index=False)
        pd.DataFrame([policy]).to_csv(
            out_dir / "audit_fit_reference_diagnostics.csv",
            index=False,
        )
        pd.DataFrame(
            [
                {
                    "dataset": "biosnap",
                    "split": "cluster",
                    "direct_target_support_status": "available",
                    "joint_support_status": "available",
                }
            ]
        ).to_csv(out_dir / "source_support_provenance.csv", index=False)
        pd.DataFrame(columns=policy).to_csv(
            out_dir / "human_pair_deduplication_diagnostics.csv",
            index=False,
        )
        pd.DataFrame([policy]).to_csv(
            out_dir / "selection_choices.csv",
            index=False,
        )
        pd.DataFrame([policy]).to_csv(
            out_dir / "selected_test_runs.csv",
            index=False,
        )
        pd.DataFrame(columns=policy).to_csv(
            out_dir / "human_deduplicated_selected_test_runs.csv",
            index=False,
        )
        increments = []
        for split in ("validation", "test"):
            for component, baseline, augmented in _expected_comparisons(
                manifest
            ):
                increments.append(
                    {
                        **policy,
                        "split": split,
                        "added_component": component,
                        "baseline_profile": baseline,
                        "augmented_profile": augmented,
                    }
                )
        pd.DataFrame(increments).to_csv(
            out_dir / "component_increment_runs.csv",
            index=False,
        )
        return spec

    def test_formal_runner_exposes_only_frozen_profiles(self) -> None:
        manifest = profile_manifest()
        profiles = set(manifest["profile"].astype(str))
        self.assertIn("fit_support", profiles)
        self.assertIn("source_domain_support", profiles)
        self.assertIn("target_sequence_support", profiles)
        self.assertIn("general_target_joint_support", profiles)
        self.assertEqual(
            len(manifest),
            len(PRIMARY_PROFILES) + len(EVIDENCE_DIAGNOSTIC_PROFILES),
        )
        self.assertTrue(
            manifest.loc[
                manifest["profile"].isin(
                    [profile.name for profile in EVIDENCE_DIAGNOSTIC_PROFILES]
                ),
                "primary_candidate",
            ].eq(False).all()
        )

    def test_dataset_loads_only_base_and_prior_inputs(self) -> None:
        with TemporaryDirectory() as directory:
            split_dir = Path(directory) / "demo" / "cluster"
            split_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "dr_id": [1],
                    "pr_id": [10],
                    "Y": [1],
                    "SMILES": ["CC"],
                    "Protein": ["ACDE"],
                }
            ).to_csv(split_dir / "source_train_with_id.csv", index=False)
            payloads = {
                "drug_cls_feat.pkl": {1: np.asarray([1.0, 0.0])},
                "drug_token_feat.pkl": {
                    1: np.asarray([[1.0, 0.0]], dtype=np.float32)
                },
                "prot_cls_feat.pkl": {10: np.asarray([0.0, 1.0])},
                "prot_token_feat.pkl": {
                    10: np.asarray([[0.0, 1.0]], dtype=np.float32)
                },
            }
            for filename, payload in payloads.items():
                with (split_dir / filename).open("wb") as handle:
                    pickle.dump(payload, handle)

            pime_dir = split_dir / "pime"
            pime_dir.mkdir()
            with (pime_dir / "drug_prior_feat.pkl").open("wb") as handle:
                pickle.dump({1: np.asarray([0.0, 1.0])}, handle)
            with (pime_dir / "target_prior_feat.pkl").open("wb") as handle:
                pickle.dump({10: np.asarray([1.0, 0.0])}, handle)
            pd.DataFrame(
                {"dr_id": [1], "canonical_smiles": ["CC"]}
            ).to_csv(pime_dir / "drug_entity.csv", index=False)
            pd.DataFrame(
                {"pr_id": [10], "protein_sequence": ["ACDE"]}
            ).to_csv(pime_dir / "target_entity.csv", index=False)

            config = Config()
            config.DATA.PATHS.ROOT_DIR = directory
            self.assertFalse(hasattr(config.MODEL, "PIME"))
            dataset = PIMEDTIDataset("demo", "cluster", "train", config)
            item = dataset[0]
            self.assertIn("pime_drug_prior", item)
            self.assertIn("pime_target_prior", item)

            batch = pime_collate_fn([item])
            self.assertIn("pime_drug_prior", batch)
            self.assertIn("pime_target_prior", batch)

    def test_error_risk_tracks_the_frozen_predicted_class(self) -> None:
        table = pd.DataFrame(
            {
                "base_pred": [1, 0, 1, 0],
                "p_base": [0.9, 0.1, 0.9, 0.1],
            }
        )
        probability = pd.Series([0.2, 0.8, 0.9, 0.1])
        risk = calibrated_error_risk(table, probability)
        np.testing.assert_allclose(risk, [0.8, 0.8, 0.1, 0.1])

    def test_prior_features_preserve_direction(self) -> None:
        table = pd.DataFrame(
            {
                "base_pred": [1, 1],
                "s_base": [3.0, 1.0],
                "s_prior": [1.0, 3.0],
            }
        )
        features = engineer_features(
            table,
            calibrated_probability=pd.Series([0.8, 0.8]),
            evidence_probabilities={
                "prior": pd.Series([0.2, 0.9]),
            },
        )
        self.assertAlmostEqual(
            features.loc[0, "prior_contradiction_probability"],
            0.6,
        )
        self.assertEqual(features.loc[0, "prior_contradiction_logit"], 2.0)
        self.assertEqual(features.loc[0, "prior_support_probability"], 0.0)
        self.assertEqual(
            features.loc[1, "prior_contradiction_probability"],
            0.0,
        )
        self.assertAlmostEqual(features.loc[1, "prior_support_probability"], 0.1)
        self.assertEqual(features.loc[1, "prior_support_logit"], 2.0)

    def test_support_features_use_only_the_fit_reference(self) -> None:
        fit = pd.DataFrame(
            {
                "dr_id": ["d1", "d2"],
                "pr_id": ["p1", "p2"],
            }
        )
        held_out = pd.DataFrame(
            {
                "base_pred": [1],
                "dr_id": ["d3"],
                "pr_id": ["p1"],
            }
        )
        features = engineer_features(
            held_out,
            calibrated_probability=pd.Series([0.8]),
            support_reference=fit,
        )
        self.assertEqual(features.loc[0, "drug_seen_in_fit"], 0.0)
        self.assertEqual(features.loc[0, "target_seen_in_fit"], 1.0)
        self.assertEqual(features.loc[0, "pair_seen_in_fit"], 0.0)
        self.assertEqual(features.loc[0, "fit_one_sided_novelty"], 1.0)

    def test_source_support_tracks_entities_clusters_and_pairs(self) -> None:
        with TemporaryDirectory() as directory:
            split_dir = Path(directory) / "demo" / "cluster"
            split_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "dr_id": [1, 1, 2],
                    "pr_id": [10, 11, 10],
                    "drug_cluster": [100, 100, 200],
                    "target_cluster": [300, 400, 300],
                }
            ).to_csv(split_dir / "source_train_with_id.csv", index=False)
            pd.DataFrame(
                {
                    "dr_id": [1, 3],
                    "pr_id": [10, 12],
                    "drug_cluster": [100, 200],
                    "target_cluster": [300, 500],
                }
            ).to_csv(split_dir / "target_test_with_id.csv", index=False)
            pime_dir = split_dir / "pime"
            pime_dir.mkdir()
            with (pime_dir / "drug_prior_feat.pkl").open("wb") as handle:
                pickle.dump(
                    {
                        1: np.asarray([0.0, 0.0, 1.0]),
                        2: np.asarray([1.0, 0.0, 1.0]),
                        3: np.asarray([4.0, 0.0, 1.0]),
                    },
                    handle,
                )
            with (pime_dir / "target_prior_feat.pkl").open("wb") as handle:
                pickle.dump(
                    {
                        10: np.asarray([0.0, 0.0, 1.0]),
                        11: np.asarray([1.0, 0.0, 1.0]),
                        12: np.asarray([0.0, 4.0, 1.0]),
                    },
                    handle,
                )

            table = pd.DataFrame(
                {
                    "split": ["cluster", "cluster"],
                    "dr_id": [1, 3],
                    "pr_id": [10, 12],
                }
            )
            supported = attach_source_support(
                table,
                dataset="demo",
                dataset_root=directory,
            )
            self.assertEqual(supported.loc[0, "source_drug_count"], 2.0)
            self.assertEqual(supported.loc[0, "source_pair_count"], 1.0)
            self.assertEqual(
                supported.loc[1, "source_drug_cluster_count"],
                1.0,
            )
            self.assertEqual(
                supported.loc[1, "source_target_cluster_count"],
                0.0,
            )
            self.assertEqual(
                supported.loc[0, "drug_nearest_train_distance"],
                0.0,
            )
            self.assertGreater(
                supported.loc[1, "drug_nearest_train_distance"],
                0.0,
            )
            self.assertGreater(
                supported.loc[1, "target_nearest_train_distance"],
                0.0,
            )

    def test_native_support_uses_morgan_and_esm_source_references(self) -> None:
        with TemporaryDirectory() as directory:
            split_dir = Path(directory) / "demo" / "cluster"
            split_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "dr_id": [1, 2],
                    "pr_id": [10, 11],
                    "Y": [1, 0],
                }
            ).to_csv(split_dir / "source_train_with_id.csv", index=False)
            pd.DataFrame(
                {
                    "dr_id": [3],
                    "pr_id": [12],
                    "Y": [1],
                }
            ).to_csv(split_dir / "target_test_with_id.csv", index=False)
            with (split_dir / "prot_cls_feat.pkl").open("wb") as handle:
                pickle.dump(
                    {
                        10: np.asarray([1.0, 0.0]),
                        11: np.asarray([0.8, 0.2]),
                        12: np.asarray([0.0, 1.0]),
                    },
                    handle,
                )
            pime_dir = split_dir / "pime"
            pime_dir.mkdir()
            pd.DataFrame(
                {
                    "dr_id": [1, 2, 3],
                    "canonical_smiles": ["CC", "CCC", "c1ccccc1"],
                }
            ).to_csv(pime_dir / "drug_entity.csv", index=False)
            with (pime_dir / "native_source_support.pkl").open("wb") as handle:
                pickle.dump(
                    {
                        "drug_nearest": {
                            "1": 0.0,
                            "2": 0.0,
                            "3": 0.8,
                        },
                        "drug_density": {
                            "1": 0.1,
                            "2": 0.1,
                            "3": 0.9,
                        },
                        "target_nearest": {
                            "10": 0.0,
                            "11": 0.0,
                            "12": 0.6,
                        },
                        "target_density": {
                            "10": 0.1,
                            "11": 0.1,
                            "12": 0.7,
                        },
                    },
                    handle,
                )

            supported = attach_source_support(
                pd.DataFrame(
                    {
                        "split": ["cluster", "cluster"],
                        "dr_id": [1, 3],
                        "pr_id": [10, 12],
                    }
                ),
                dataset="demo",
                dataset_root=directory,
            )
            self.assertEqual(
                supported.loc[0, "drug_morgan_nearest_train_distance"],
                0.0,
            )
            self.assertEqual(
                supported.loc[0, "target_esm_nearest_train_distance"],
                0.0,
            )
            self.assertGreater(
                supported.loc[1, "drug_morgan_nearest_train_distance"],
                0.0,
            )
            self.assertGreater(
                supported.loc[1, "target_esm_nearest_train_distance"],
                0.0,
            )
            self.assertGreater(
                supported.loc[1, "native_support_imbalance"],
                0.0,
            )

            provenance = source_support_provenance(
                dataset="demo",
                dataset_root=directory,
            )
            self.assertFalse(provenance["outcome_columns_loaded"])
            self.assertNotIn("Y", provenance["source_columns_loaded"])
            self.assertEqual(
                provenance["drug_native_metric"],
                "morgan_radius2_2048_tanimoto",
            )
            self.assertEqual(
                provenance["target_native_metric"],
                "prot_cls_esm_cosine",
            )

    def test_target_sequence_support_uses_only_source_target_families(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            split_dir = Path(directory) / "demo" / "cluster"
            pime_dir = split_dir / "pime"
            mmseqs_dir = pime_dir / "mmseqs"
            mmseqs_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "dr_id": [1],
                    "pr_id": [10],
                    "Y": [1],
                }
            ).to_csv(split_dir / "source_train_with_id.csv", index=False)
            pd.DataFrame(
                {
                    "pr_id": [10, 11, 12],
                    "protein_sequence": ["AAAA", "AAAT", "CCCC"],
                }
            ).to_csv(pime_dir / "target_entity.csv", index=False)
            rows = [
                ["pr_id=10", "P_SOURCE", 100, 100, 1, 100, 1, 100, 0, 200, 100, 100],
                ["pr_id=11", "P_SOURCE", 50, 100, 1, 100, 1, 100, 1e-20, 100, 100, 100],
                ["pr_id=12", "P_OTHER", 100, 100, 1, 100, 1, 100, 0, 200, 100, 100],
            ]
            pd.DataFrame(rows).to_csv(
                mmseqs_dir / "demo_vs_sprot.tsv",
                sep="\t",
                header=False,
                index=False,
            )
            payload = compute_target_sequence_support_payload(
                directory,
                "demo",
                "cluster",
            )
            self.assertIsNotNone(payload)
            assert payload is not None
            self.assertEqual(payload["target_distance"]["10"], 0.0)
            self.assertAlmostEqual(payload["target_distance"]["11"], 0.5)
            self.assertEqual(payload["target_family_unseen"]["11"], 0.0)
            self.assertEqual(payload["target_distance"]["12"], 1.0)
            self.assertEqual(payload["target_family_unseen"]["12"], 1.0)

    def test_direct_target_support_uses_fixed_source_neighbours(self) -> None:
        with TemporaryDirectory() as directory:
            split_dir = Path(directory) / "demo" / "cluster"
            pime_dir = split_dir / "pime"
            mmseqs_dir = pime_dir / "mmseqs"
            mmseqs_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "dr_id": [1, 2],
                    "pr_id": [10, 11],
                }
            ).to_csv(split_dir / "source_train_with_id.csv", index=False)
            pd.DataFrame(
                {
                    "pr_id": [10, 11, 12],
                    "protein_sequence": ["AAAA", "AAAT", "AATT"],
                }
            ).to_csv(pime_dir / "target_entity.csv", index=False)
            rows = [
                ["pr_id=10", "pr_id=10", 100, 4, 1, 4, 1, 4, 0, 20, 4, 4],
                ["pr_id=12", "pr_id=10", 50, 4, 1, 4, 1, 4, 1e-5, 10, 4, 4],
                ["pr_id=12", "pr_id=11", 25, 4, 1, 4, 1, 4, 1e-2, 5, 4, 4],
            ]
            pd.DataFrame(rows).to_csv(
                mmseqs_dir / "direct_target_to_source.tsv",
                sep="\t",
                header=False,
                index=False,
            )
            payload = compute_direct_target_support_payload(
                directory,
                "demo",
                "cluster",
            )
            self.assertIsNotNone(payload)
            assert payload is not None
            self.assertEqual(payload["target_nearest"]["10"], 0.0)
            self.assertAlmostEqual(payload["target_nearest"]["12"], 0.5)
            self.assertAlmostEqual(payload["target_density"]["12"], 0.85)
            self.assertEqual(
                payload["target_neighbours"]["12"][0],
                ("10", 0.5),
            )

    def test_joint_support_is_label_free_and_excludes_exact_pair(self) -> None:
        payload = {
            "drug_neighbours": {
                "d1": (("d1", 1.0), ("d2", 0.5)),
                "d3": (("d1", 0.8), ("d2", 0.5)),
            },
            "target_neighbours": {
                "p10": (("p10", 1.0), ("p11", 0.4)),
                "p12": (("p10", 0.7), ("p11", 0.6)),
            },
            "source_pairs": {
                ("d1", "p10"),
                ("d1", "p11"),
                ("d2", "p10"),
            },
        }
        self.assertAlmostEqual(
            joint_nearest_train_distance("d3", "p12", payload),
            0.44,
        )
        self.assertAlmostEqual(
            joint_nearest_train_distance("d1", "p10", payload),
            0.5,
        )

    def test_generic_support_assets_attach_without_model_checkpoint(self) -> None:
        with TemporaryDirectory() as directory:
            split_dir = Path(directory) / "demo" / "cluster"
            pime_dir = split_dir / "pime"
            pime_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "dr_id": ["d1", "d2"],
                    "pr_id": ["p10", "p11"],
                }
            ).to_csv(split_dir / "source_train_with_id.csv", index=False)
            with (pime_dir / "direct_target_support.pkl").open("wb") as handle:
                pickle.dump(
                    {
                        "target_nearest": {"p12": 0.3},
                        "target_density": {"p12": 0.7},
                        "target_neighbours": {
                            "p12": (("p10", 0.7), ("p11", 0.6))
                        },
                    },
                    handle,
                )
            with (pime_dir / "joint_source_support.pkl").open("wb") as handle:
                pickle.dump(
                    {
                        "drug_neighbours": {
                            "d3": (("d1", 0.8), ("d2", 0.5))
                        },
                        "target_neighbours": {
                            "p12": (("p10", 0.7), ("p11", 0.6))
                        },
                        "source_pairs": {
                            ("d1", "p11"),
                            ("d2", "p10"),
                        },
                    },
                    handle,
                )
            supported = attach_source_support(
                pd.DataFrame(
                    {
                        "split": ["cluster"],
                        "dr_id": ["d3"],
                        "pr_id": ["p12"],
                    }
                ),
                dataset="demo",
                dataset_root=directory,
            )
            self.assertEqual(
                supported.loc[0, "target_direct_nearest_train_distance"],
                0.3,
            )
            self.assertEqual(
                supported.loc[0, "target_direct_knn_distance"],
                0.7,
            )
            self.assertAlmostEqual(
                supported.loc[0, "joint_nearest_train_distance"],
                0.52,
            )

    def test_capabilities_and_prior_fallback_are_explicit(self) -> None:
        table = pd.DataFrame(
            {
                "p_base": [0.8, 0.9],
                "dr_id": ["d1", "d2"],
                "pr_id": ["p1", "p2"],
                "target_direct_nearest_train_distance": [0.2, 0.4],
                "joint_nearest_train_distance": [0.3, 0.5],
            }
        )
        capabilities = audit_capabilities(table)
        self.assertTrue(capabilities["uncertainty_available"])
        self.assertFalse(capabilities["prior_available"])
        self.assertTrue(capabilities["empirical_support_available"])
        self.assertEqual(capabilities["maximum_mode"], "U+E")
        self.assertEqual(
            PRIMARY_PROFILES[2].required_blocks,
            (tuple(PRIMARY_PROFILES[1].features),),
        )

    def test_human_pair_sensitivity_excludes_overlap_and_collapses_pairs(
        self,
    ) -> None:
        validation = pd.DataFrame(
            {
                "dr_id": [1],
                "pr_id": [10],
                "base_wrong": [0],
            }
        )
        test = pd.DataFrame(
            {
                "dr_id": [1, 2, 2, 3],
                "pr_id": [10, 20, 20, 30],
                "base_wrong": [0, 1, 1, 0],
            }
        )
        metrics, diagnostics = _pair_deduplicated_metrics(
            validation,
            test,
            pd.Series([True, True, True, True]),
            pd.Series([False, False, False, False]),
            pd.Series([0.1, 0.9, 0.7, 0.2]),
            review_fraction=0.5,
            min_universe_n=2,
        )
        self.assertEqual(diagnostics["cross_split_overlap_rows_removed"], 1)
        self.assertEqual(diagnostics["cross_split_overlap_pairs_removed"], 1)
        self.assertEqual(diagnostics["within_test_duplicate_rows_collapsed"], 1)
        self.assertEqual(metrics["universe_n"], 2)
        self.assertEqual(metrics["error_detection_auprc"], 1.0)

    def test_component_increment_rows_keep_human_deduplicated_metrics(
        self,
    ) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "model": "PACE",
                    "dataset": "human",
                    "test_run": "seed_4",
                    "group_axis": "target",
                    "confidence_source": "base",
                    "confidence_threshold": 0.8,
                    "uncertainty_reject_fraction": 0.1,
                    "review_fraction": 0.2,
                    "profile": "uncertainty_support",
                    "primary_candidate": True,
                    "model_status": "complete",
                    "error_detection_auprc": 0.2,
                    "deduplicated_error_detection_auprc": 0.15,
                },
                {
                    "model": "PACE",
                    "dataset": "human",
                    "test_run": "seed_4",
                    "group_axis": "target",
                    "confidence_source": "base",
                    "confidence_threshold": 0.8,
                    "uncertainty_reject_fraction": 0.1,
                    "review_fraction": 0.2,
                    "profile": "general_target_joint_support",
                    "primary_candidate": False,
                    "model_status": "complete",
                    "error_detection_auprc": 0.3,
                    "deduplicated_error_detection_auprc": 0.22,
                },
            ]
        )
        rows = _component_increment_rows(candidates, split="test")
        diagnostic = next(
            row
            for row in rows
            if row["added_component"]
            == "general_target_joint_support"
        )
        self.assertAlmostEqual(
            diagnostic[
                "delta_deduplicated_error_detection_auprc"
            ],
            0.07,
        )

    def test_shard_validation_accepts_complete_cartesian_product(self) -> None:
        with TemporaryDirectory() as directory:
            spec = self._write_minimal_complete_shard(directory)
            report = validate_shard(
                spec,
                seeds=["4"],
                group_axes=["target"],
                confidence_source="base",
                confidence_thresholds=[0.8],
                reject_fractions=[0.1],
                review_fraction=0.2,
            )
            self.assertEqual(report["policies"], 1)
            self.assertEqual(report["candidate_rows"], 12)
            self.assertEqual(report["increment_rows"], 34)

    def test_shard_validation_rejects_duplicate_policy_key(self) -> None:
        with TemporaryDirectory() as directory:
            spec = self._write_minimal_complete_shard(directory)
            path = spec.out_dir / "selection_choices.csv"
            frame = pd.read_csv(path)
            pd.concat([frame, frame], ignore_index=True).to_csv(
                path,
                index=False,
            )
            with self.assertRaises(ShardValidationError):
                validate_shard(
                    spec,
                    seeds=["4"],
                    group_axes=["target"],
                    confidence_source="base",
                    confidence_thresholds=[0.8],
                    reject_fractions=[0.1],
                    review_fraction=0.2,
                )

    def test_empirical_support_profiles_are_explicit(self) -> None:
        profiles = {profile.name: profile for profile in PRIMARY_PROFILES}
        support = profiles["uncertainty_support"]
        combined = profiles["uncertainty_prior_support"]
        self.assertIn("fit_entity_support_min", support.features)
        self.assertIn(
            "drug_morgan_nearest_train_distance",
            support.features,
        )
        self.assertIn(
            "target_esm_nearest_train_distance",
            support.features,
        )
        self.assertIn("native_support_imbalance", support.features)
        self.assertNotIn("log_pair_fit_count", support.features)
        self.assertNotIn("source_cluster_support_min", support.features)
        self.assertNotIn("drug_nearest_train_distance", support.features)
        self.assertNotIn("prior_negative_evidence", support.features)
        self.assertIn("prior_negative_evidence", combined.features)
        self.assertTrue(support.required_any)
        self.assertTrue(combined.required_any)
        target_diagnostic = {
            profile.name: profile for profile in EVIDENCE_DIAGNOSTIC_PROFILES
        }["target_sequence_support"]
        self.assertFalse(target_diagnostic.primary)
        self.assertIn(
            "target_mmseqs_nearest_train_distance",
            target_diagnostic.required_any,
        )
        general_diagnostic = {
            profile.name: profile for profile in EVIDENCE_DIAGNOSTIC_PROFILES
        }["general_target_joint_support"]
        self.assertFalse(general_diagnostic.primary)
        self.assertEqual(
            general_diagnostic.required_any,
            (
                "target_direct_nearest_train_distance",
                "target_direct_knn_distance",
                "joint_nearest_train_distance",
            ),
        )

    def test_fixed_diagnostics_keep_distinct_support_definitions(self) -> None:
        profiles = {
            profile.name: profile
            for profile in EVIDENCE_DIAGNOSTIC_PROFILES
        }
        fit_support = profiles["fit_support"]
        source_support = profiles["source_domain_support"]
        self.assertIn("log_drug_fit_count", fit_support.required_any)
        self.assertNotIn(
            "drug_nearest_train_distance",
            fit_support.features,
        )
        self.assertIn(
            "drug_nearest_train_distance",
            source_support.required_any,
        )
        self.assertNotIn("log_drug_fit_count", source_support.features)

    def test_policy_preserves_base_candidates_and_defers_highest_risk(self) -> None:
        table = pd.DataFrame(
            {
                "base_pred": [1, 1, 1],
                "base_confidence": [0.9, 0.9, 0.9],
                "p_base": [0.95, 0.95, 0.95],
            }
        )
        calibrated = pd.Series([0.9, 0.2, 0.6])
        policy = apply_uncertainty_policy(
            table,
            calibrated,
            confidence_threshold=0.8,
            reject_fraction=0.34,
            confidence_source="base",
        )
        self.assertEqual(policy["candidate_n"], 3)
        self.assertEqual(policy["deferred_n"], 1)
        self.assertTrue(policy["deferred_mask"].iloc[1])

        calibrated_policy = apply_uncertainty_policy(
            table,
            calibrated,
            confidence_threshold=0.8,
            reject_fraction=0.0,
            confidence_source="calibrated",
        )
        self.assertEqual(calibrated_policy["candidate_n"], 1)

    def test_calibration_metrics_report_brier_and_ece(self) -> None:
        table = pd.DataFrame({"base_wrong": [0, 0, 1, 1]})
        metrics = calibration_metrics(
            table,
            pd.Series([0.0, 0.0, 1.0, 1.0]),
            bins=2,
        )
        self.assertEqual(metrics["calibration_n"], 4)
        self.assertEqual(metrics["brier_error_risk"], 0.0)
        self.assertEqual(metrics["ece_error_risk"], 0.0)

    def test_one_se_selection_prefers_simple_model_and_gain_gate(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "profile": "uncertainty",
                    "primary_candidate": True,
                    "complexity": 0,
                    "validation_fold_mean_auprc": 0.30,
                    "validation_fold_se_auprc": 0.01,
                },
                {
                    "profile": "uncertainty_prior",
                    "primary_candidate": True,
                    "complexity": 2,
                    "validation_fold_mean_auprc": 0.34,
                    "validation_fold_se_auprc": 0.05,
                },
                {
                    "profile": "prior",
                    "primary_candidate": True,
                    "complexity": 3,
                    "validation_fold_mean_auprc": 0.35,
                    "validation_fold_se_auprc": 0.02,
                },
            ]
        )
        selected = select_one_standard_error(candidates)
        self.assertEqual(selected["selected_profile"], "uncertainty_prior")

        gated = select_one_standard_error(
            candidates.assign(
                validation_fold_mean_auprc=[0.30, 0.303, 0.304]
            ),
            min_validation_gain=0.005,
        )
        self.assertEqual(gated["selected_profile"], "uncertainty")
        self.assertTrue(gated["selection_fallback"])

    def test_grouped_cross_fitted_residual_model_is_complete(self) -> None:
        n_groups = 12
        rows_per_group = 10
        n = n_groups * rows_per_group
        wrong = np.tile([0, 1], n // 2)
        table = pd.DataFrame(
            {
                "base_wrong": wrong,
                "label": 1.0 - wrong,
                "base_pred": np.ones(n),
                "p_base": np.full(n, 0.9),
                "s_base": np.full(n, 2.0),
                "s_prior": 2.0 - 1.5 * wrong,
                "dr_id": [f"d{i % 20}" for i in range(n)],
                "pr_id": np.repeat(
                    [f"p{i}" for i in range(n_groups)],
                    rows_per_group,
                ),
            }
        )
        result = cross_fitted_profile(
            table,
            pd.Series(True, index=table.index),
            pd.Series(np.full(n, 0.8), index=table.index),
            PRIMARY_PROFILES[2],
            folds=3,
            seed=2026,
            group_axis="target",
            strict_group=True,
            min_fit_n=20,
            min_feature_n=10,
            ranking_weight=0.5,
            max_pairs=500,
        )
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.scores.notna().sum(), n)
        self.assertEqual(len(result.fold_auprcs), 3)
        self.assertGreater(result.mean_fold_auprc, 0.9)

    def test_missing_diagnostic_is_not_reported_as_zero_increment(self) -> None:
        n = 40
        validation = pd.DataFrame(
            {
                "base_wrong": np.tile([0, 1], n // 2),
                "label": 1.0 - np.tile([0, 1], n // 2),
                "base_pred": np.ones(n),
                "p_base": np.full(n, 0.9),
                "s_base": np.full(n, 2.0),
                "s_prior": np.tile([2.0, 0.5], n // 2),
                "dr_id": [f"d{i % 5}" for i in range(n)],
                "pr_id": [f"p{i % 5}" for i in range(n)],
            }
        )
        test = validation.iloc[:10].copy()
        result = fit_profile_for_test(
            validation,
            pd.Series(True, index=validation.index),
            pd.Series(np.full(n, 0.8), index=validation.index),
            test,
            pd.Series(True, index=test.index),
            pd.Series(np.full(len(test), 0.8), index=test.index),
            {
                profile.name: profile
                for profile in EVIDENCE_DIAGNOSTIC_PROFILES
            }["general_target_joint_support"],
            seed=2026,
            min_fit_n=20,
            min_feature_n=10,
            ranking_weight=0.5,
            max_pairs=200,
        )
        self.assertEqual(result.status, "diagnostic_unavailable")
        self.assertTrue(result.scores.isna().all())

    def test_missing_prior_forces_explicit_primary_fallback(self) -> None:
        n = 40
        wrong = np.tile([0, 1], n // 2)
        validation = pd.DataFrame(
            {
                "base_wrong": wrong,
                "label": 1.0 - wrong,
                "base_pred": np.ones(n),
                "p_base": np.full(n, 0.9),
                "s_base": np.full(n, 2.0),
                "dr_id": [f"d{i % 5}" for i in range(n)],
                "pr_id": [f"p{i % 5}" for i in range(n)],
            }
        )
        test = validation.iloc[:10].copy()
        result = fit_profile_for_test(
            validation,
            pd.Series(True, index=validation.index),
            pd.Series(np.full(n, 0.8), index=validation.index),
            test,
            pd.Series(True, index=test.index),
            pd.Series(np.full(len(test), 0.8), index=test.index),
            PRIMARY_PROFILES[2],
            seed=2026,
            min_fit_n=20,
            min_feature_n=10,
            ranking_weight=0.5,
            max_pairs=200,
        )
        self.assertEqual(result.status, "profile_unavailable")
        self.assertTrue(result.scores.isna().all())


if __name__ == "__main__":
    unittest.main()
