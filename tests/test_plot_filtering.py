"""Regression tests for publication-figure paired-success filtering."""

from __future__ import annotations

import unittest

import pandas as pd

from experiments.plot_four_publication_figures_paper_aligned import (
    CONFIGS,
    FAMILIES,
    common_global_success,
    macro_metric_curve,
)


class PlotFilteringTests(unittest.TestCase):
    def test_continuous_metrics_require_all_four_global_successes(self) -> None:
        rows: list[dict[str, object]] = []
        for seed in (1, 2):
            for configuration in CONFIGS:
                scheduler, sharing = configuration.split("-", 1)
                success = not (
                    seed == 1 and configuration == "PBQS-Sharable"
                )
                rows.append(
                    {
                        "family": "montage",
                        "workflow_size": 100,
                        "deadline_factor": 3.0,
                        "seed": seed,
                        "scheduler": scheduler,
                        "sharing": sharing,
                        "configuration": configuration,
                        "topology": "small",
                        "node_count": 5,
                        "candidate_count": 20,
                        "global_deadline_success": int(success),
                    }
                )
        retained = common_global_success(pd.DataFrame(rows))
        self.assertEqual(set(retained.seed), {2})
        self.assertEqual(set(retained.configuration), set(CONFIGS))
        self.assertTrue(retained.global_deadline_success.eq(1).all())

    def test_macro_curve_weights_workflows_equally(self) -> None:
        rows: list[dict[str, object]] = []
        family_means = dict(zip(FAMILIES, (10.0, 20.0, 30.0, 40.0)))
        family_counts = dict(zip(FAMILIES, (5, 6, 7, 8)))
        for family in FAMILIES:
            for seed in range(1, family_counts[family] + 1):
                for configuration in CONFIGS:
                    rows.append(
                        {
                            "family": family,
                            "workflow_size": 100,
                            "deadline_factor": 3.0,
                            "seed": seed,
                            "configuration": configuration,
                            "global_deadline_success": 1,
                            "provisioning_cost": family_means[family],
                        }
                    )
        curve = macro_metric_curve(
            pd.DataFrame(rows),
            "HDS-Sharable",
            "provisioning_cost",
        )
        self.assertEqual(len(curve), 1)
        self.assertAlmostEqual(float(curve.iloc[0]["mean"]), 25.0)


if __name__ == "__main__":
    unittest.main()
