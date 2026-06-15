from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from audit_framework.strengthening import (
    _curve_auc,
    _global_one_se_choice,
    build_strengthening_shards,
    validate_strengthening_shard,
)


class StrengtheningExperimentTests(unittest.TestCase):
    def test_global_one_se_prefers_simpler_eligible_profile(self) -> None:
        rows = []
        for index, (u, p, upe) in enumerate(
            [
                (0.20, 0.255, 0.24),
                (0.21, 0.255, 0.30),
                (0.19, 0.255, 0.24),
                (0.20, 0.255, 0.30),
            ]
        ):
            for profile, complexity, value in (
                ("uncertainty", 0, u),
                ("prior", 1, p),
                ("uncertainty_prior_support", 3, upe),
            ):
                rows.append(
                    {
                        "model": "M",
                        "dataset": "D",
                        "test_run": f"seed_{index}",
                        "profile": profile,
                        "complexity": complexity,
                        "validation_fold_mean_auprc": value,
                    }
                )
        choice = _global_one_se_choice(
            pd.DataFrame(rows),
            min_gain=0.005,
        )
        self.assertEqual(choice["selected_profile"], "prior")

    def test_global_one_se_gain_gate_falls_back_to_uncertainty(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "model": "M",
                    "dataset": "D",
                    "test_run": f"seed_{seed}",
                    "profile": profile,
                    "complexity": complexity,
                    "validation_fold_mean_auprc": value,
                }
                for seed in range(4)
                for profile, complexity, value in (
                    ("uncertainty", 0, 0.300),
                    ("prior", 1, 0.303),
                )
            ]
        )
        choice = _global_one_se_choice(frame, min_gain=0.005)
        self.assertEqual(choice["selected_profile"], "uncertainty")
        self.assertEqual(choice["selection_reason"], "gain_below_minimum")

    def test_action_curve_auc_uses_equal_policy_keys(self) -> None:
        rows = []
        for review, pause, uncertainty in (
            (0.05, 0.30, 0.20),
            (0.10, 0.50, 0.35),
            (0.20, 0.70, 0.55),
            (0.30, 0.80, 0.65),
        ):
            rows.append(
                {
                    "model": "M",
                    "dataset": "D",
                    "test_run": "seed_4",
                    "group_axis": "target",
                    "confidence_source": "base",
                    "confidence_threshold": 0.8,
                    "uncertainty_reject_fraction": 0.1,
                    "review_fraction": review,
                    "total_action_fraction": 0.1 + 0.9 * review,
                    "combined_recall_of_candidate_errors": pause,
                    (
                        "uncertainty_"
                        "combined_recall_of_candidate_errors"
                    ): uncertainty,
                }
            )
        result = _curve_auc(pd.DataFrame(rows))
        self.assertEqual(len(result), 1)
        self.assertGreater(
            result.iloc[0][
                "delta_normalized_action_auc_vs_uncertainty"
            ],
            0.0,
        )

    def test_missing_strengthening_shard_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            shard = build_strengthening_shards(
                models=["pace"],
                datasets=["biosnap"],
                shard_root=Path(directory),
            )[0]
            with self.assertRaises(RuntimeError):
                validate_strengthening_shard(shard)


if __name__ == "__main__":
    unittest.main()
