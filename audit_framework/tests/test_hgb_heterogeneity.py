import unittest

import pandas as pd

from audit_framework.hgb_heterogeneity import (
    build_hgb_policy_runs,
    cluster_hgb_policy_runs,
)


def _synthetic_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = []
    selected = []
    for review_fraction in (0.05, 0.10):
        base = {
            "model": "M",
            "dataset": "D",
            "test_run": "seed_4",
            "group_axis": "target",
            "confidence_source": "base",
            "confidence_threshold": 0.8,
            "uncertainty_reject_fraction": 0.1,
            "review_fraction": review_fraction,
        }
        selected.append(
            {
                **base,
                "selected_profile": "prior",
                "error_detection_auprc": 0.40,
                "deduplicated_error_detection_auprc": 0.35,
            }
        )
        for split, method, score, fold, dedup in (
            ("validation", "prior", 0.50, 0.48, float("nan")),
            ("validation", "fixed_hgb_upe", 0.45, 0.44, float("nan")),
            ("test", "fixed_hgb_upe", 0.60, float("nan"), 0.50),
        ):
            candidates.append(
                {
                    **base,
                    "split": split,
                    "method": method,
                    "model_status": "complete",
                    "error_detection_auprc": score,
                    "validation_fold_mean_auprc": fold,
                    "deduplicated_error_detection_auprc": dedup,
                }
            )
    return pd.DataFrame(candidates), pd.DataFrame(selected)


class HgbHeterogeneityTests(unittest.TestCase):
    def test_builds_one_policy_pair_and_cluster(self) -> None:
        candidates, selected = _synthetic_rows()
        policy = build_hgb_policy_runs(candidates, selected)
        self.assertEqual(len(policy), 1)
        self.assertAlmostEqual(policy.iloc[0]["delta_validation_auprc"], -0.05)
        self.assertAlmostEqual(policy.iloc[0]["delta_test_auprc"], 0.20)
        self.assertAlmostEqual(
            policy.iloc[0]["delta_test_deduplicated_auprc"],
            0.15,
        )

        cluster = cluster_hgb_policy_runs(policy)
        self.assertEqual(len(cluster), 1)
        self.assertEqual(cluster.iloc[0]["same_direction"], 0.0)
        self.assertEqual(cluster.iloc[0]["both_positive"], 0.0)

    def test_duplicate_candidate_key_is_rejected(self) -> None:
        candidates, selected = _synthetic_rows()
        candidates = pd.concat(
            [candidates, candidates.iloc[[0]]],
            ignore_index=True,
        )
        with self.assertRaisesRegex(ValueError, "duplicate keys"):
            build_hgb_policy_runs(candidates, selected)

    def test_review_dependent_auprc_is_rejected(self) -> None:
        candidates, selected = _synthetic_rows()
        mask = (
            candidates["review_fraction"].eq(0.10)
            & candidates["split"].eq("test")
            & candidates["method"].eq("fixed_hgb_upe")
        )
        candidates.loc[mask, "error_detection_auprc"] = 0.61
        with self.assertRaisesRegex(ValueError, "review-fraction-dependent"):
            build_hgb_policy_runs(candidates, selected)


if __name__ == "__main__":
    unittest.main()
