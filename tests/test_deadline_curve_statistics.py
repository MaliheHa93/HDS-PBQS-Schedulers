"""Regression tests for curve-level deadline inference."""

from __future__ import annotations

import unittest

import pandas as pd

from experiments.analyze_deadline_curves import (
    ORIGINAL_FACTORS,
    curve_inference,
    sustained_kappa_90,
)


class DeadlineCurveStatisticsTests(unittest.TestCase):
    def test_identical_curves_have_zero_effect_and_unit_p_value(self) -> None:
        rows = []
        for seed in (1, 2):
            for scheduler in ("HDS", "PBQS"):
                for factor in ORIGINAL_FACTORS:
                    rows.append(
                        {
                            "family": "manual",
                            "sharing": "Sharable",
                            "scheduler": scheduler,
                            "configuration": f"{scheduler}-Sharable",
                            "seed": seed,
                            "deadline_factor": factor,
                            "global_deadline_success": int(factor >= 1.5),
                        }
                    )
        frame = pd.DataFrame(rows)
        result = curve_inference(frame).iloc[0]
        self.assertAlmostEqual(
            result["auc_difference_percentage_points"],
            0.0,
        )
        self.assertEqual(result["paired_t_p_value"], 1.0)
        self.assertEqual(result["holm_adjusted_p_value"], 1.0)
        threshold = sustained_kappa_90(frame)
        self.assertTrue(threshold.sustained_kappa_90.eq(1.5).all())


if __name__ == "__main__":
    unittest.main()
