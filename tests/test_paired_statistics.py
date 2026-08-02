"""Regression tests for paired statistical analysis."""

from __future__ import annotations

import unittest

import pandas as pd

from experiments.paired_statistics import analyze


class PairedStatisticsTests(unittest.TestCase):
    def test_binary_metric_is_selected_once_and_direction_is_consistent(
        self,
    ) -> None:
        rows = []
        for seed, hds_success, pbqs_success in (
            (1, 1, 1),
            (2, 0, 1),
            (3, 1, 1),
        ):
            for scheduler, success in (
                ("HDS", hds_success),
                ("PBQS", pbqs_success),
            ):
                rows.append(
                    {
                        "family": "manual",
                        "workflow_size": 3,
                        "deadline_factor": 1.0,
                        "seed": seed,
                        "scheduler": scheduler,
                        "sharing": "Sharable",
                        "reuse_policy": (
                            "idle_only"
                            if scheduler == "HDS"
                            else "queue_aware"
                        ),
                        "global_deadline_success": success,
                    }
                )
        result = analyze(
            pd.DataFrame(rows),
            metrics=("global_deadline_success",),
        )
        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertEqual(row["primary_test"], "exact_mcnemar")
        self.assertLess(row["hds_improvement_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
