"""Tests for confidence intervals and raw-result integrity checks."""

from __future__ import annotations

import unittest

import pandas as pd

from hds.metrics import confidence_interval_95
from hds.normalization import add_normalized_columns
from hds.result_validation import case_key, validate_scalability_results


def result_row(scheduler: str) -> dict[str, object]:
    sharable = True
    return {
        "topology": "small",
        "family": "montage",
        "workflow_size": 25,
        "deadline_factor": 2.0,
        "deadline_mode": "reference",
        "seed": 1,
        "scheduler": scheduler,
        "sharing": "Sharable",
        "configuration": f"{scheduler}-Sharable",
        "candidate_count": 20,
        "eta": 0.7,
        "alpha": 0.7,
        "beta": 0.2,
        "gamma": 0.1,
        "omega_u": 0.7,
        "omega_w": 0.3,
        "tastd_mode": "downstream_reserved",
        "enable_vm_reuse": 1,
        "joint_bos_optimization": 1,
        "node_count": 5,
        "local_node_count": 3,
        "global_node_count": 2,
        "unfinished_sfc_count": 0,
        "completed_sfc_count": 4,
        "total_sfc_count": 4,
        "provisioning_cost": 0.02,
        "end_to_end_delay_s": 2.0,
        "network_data_mb": 3.0,
        "network_hop_data_mb": 3.0,
        "resource_utilization": 0.5,
        "cpu_utilization": 0.4,
        "ram_utilization": 0.6,
        "vm_time_utilization": 0.5,
        "mean_link_bandwidth_utilization": 0.1,
        "max_link_bandwidth_utilization": 0.2,
        "workflow_completed": 1,
        "global_deadline_success": 1,
        "sfc_subdeadline_success": 1,
        "workflow_deadline_success": 1,
        "accepted_sfc_ratio": 1.0,
        "first_round_submitted_sfc_count": 1,
        "first_round_admitted_sfc_count": 1,
        "first_round_accepted_sfc_ratio": 1.0,
        "first_round_scheduler_runtime_s": 0.005,
        "reference_deadline_feasible": 1,
        "deadline_kappa": 2.0,
        "sfc_deadline_miss_rate": 0.0,
        "makespan_s": 2.0,
        "vm_reuse_rate": 0.5,
        "provisioned_vm_count": 2,
        "scheduler_runtime_s": 0.01,
        "scheduler_overhead_ratio": 0.005,
        "case_wall_runtime_s": 0.02,
        "milp_runtime_s": 0.005 if scheduler == "HDS" else 0.0,
        "milp_max_gap": 0.0,
        "milp_limit_reached_count": 0,
        "workflow_deadline_s": 10.0,
        "workflow_deadline_duration_s": 10.0,
        "deadline_to_minimum_makespan_ratio": 2.0,
        "requested_candidate_count": 20,
        "initial_effective_candidate_count": 5,
        "milp_min_effective_candidate_count": (
            5 if scheduler == "HDS" else 0
        ),
        "milp_max_effective_candidate_count": (
            5 if scheduler == "HDS" else 0
        ),
    }


class ExperimentToolTests(unittest.TestCase):
    def test_student_t_interval_for_three_observations(self) -> None:
        low, high = confidence_interval_95([1.0, 2.0, 3.0])
        self.assertAlmostEqual(low, -0.4841377117, places=8)
        self.assertAlmostEqual(high, 4.4841377117, places=8)

    def test_complete_matched_pair_passes(self) -> None:
        rows = [result_row("HDS"), result_row("PBQS")]
        expected = {case_key(row) for row in rows}
        report = validate_scalability_results(
            pd.DataFrame(rows),
            expected_case_keys=expected,
            topology_shapes={"small": (3, 2)},
            minimum_seeds_for_inference=1,
        )
        self.assertTrue(report.passed)
        self.assertEqual(report.error_count, 0)

    def test_missing_expected_case_fails(self) -> None:
        rows = [result_row("HDS")]
        expected_rows = rows + [result_row("PBQS")]
        report = validate_scalability_results(
            pd.DataFrame(rows),
            expected_case_keys={case_key(row) for row in expected_rows},
            topology_shapes={"small": (3, 2)},
            minimum_seeds_for_inference=1,
        )
        self.assertFalse(report.passed)
        codes = {issue.code for issue in report.issues}
        self.assertIn("missing_cases", codes)
        self.assertIn("incomplete_scheduler_pairs", codes)

    def test_invalid_metric_fails(self) -> None:
        rows = [result_row("HDS"), result_row("PBQS")]
        rows[0]["resource_utilization"] = 1.2
        report = validate_scalability_results(
            pd.DataFrame(rows),
            topology_shapes={"small": (3, 2)},
            minimum_seeds_for_inference=1,
        )
        self.assertFalse(report.passed)
        self.assertIn(
            "out_of_range_resource_utilization",
            {issue.code for issue in report.issues},
        )

    def test_unfinished_case_allows_undefined_completion_metrics(self) -> None:
        rows = [result_row("HDS"), result_row("PBQS")]
        for row in rows:
            row["workflow_completed"] = 0
            row["global_deadline_success"] = 0
            row["sfc_subdeadline_success"] = 0
            row["workflow_deadline_success"] = 0
            row["unfinished_sfc_count"] = 1
            row["completed_sfc_count"] = 3
            row["accepted_sfc_ratio"] = 0.75
            row["end_to_end_delay_s"] = float("nan")
            row["makespan_s"] = float("nan")
        report = validate_scalability_results(
            pd.DataFrame(rows),
            topology_shapes={"small": (3, 2)},
            minimum_seeds_for_inference=1,
        )
        self.assertTrue(report.passed)
        self.assertEqual(report.error_count, 0)
        self.assertIn(
            "unfinished_workflows",
            {issue.code for issue in report.issues},
        )

    def test_success_without_completion_is_rejected(self) -> None:
        rows = [result_row("HDS"), result_row("PBQS")]
        rows[0]["workflow_completed"] = 0
        rows[0]["completed_sfc_count"] = 3
        rows[0]["unfinished_sfc_count"] = 1
        rows[0]["accepted_sfc_ratio"] = 0.75
        rows[0]["end_to_end_delay_s"] = float("nan")
        rows[0]["makespan_s"] = float("nan")
        report = validate_scalability_results(
            pd.DataFrame(rows),
            topology_shapes={"small": (3, 2)},
            minimum_seeds_for_inference=1,
        )
        self.assertFalse(report.passed)
        self.assertIn(
            "success_without_completion",
            {issue.code for issue in report.issues},
        )

    def test_reference_kappa_mismatch_is_rejected(self) -> None:
        rows = [result_row("HDS"), result_row("PBQS")]
        rows[0]["deadline_kappa"] = 3.0
        report = validate_scalability_results(
            pd.DataFrame(rows),
            topology_shapes={"small": (3, 2)},
            minimum_seeds_for_inference=1,
        )
        self.assertFalse(report.passed)
        self.assertIn(
            "deadline_factor_mismatch",
            {issue.code for issue in report.issues},
        )

    def test_documented_normalizations_preserve_raw_values(self) -> None:
        frame = pd.DataFrame(
            {
                "family": ["montage", "montage"],
                "workflow_size": [25, 25],
                "seed": [1, 1],
                "workflow_deadline_duration_s": [10.0, 20.0],
                "provisioning_cost": [2.0, 4.0],
                "end_to_end_delay_s": [8.0, 4.0],
                "network_data_mb": [1.0, 2.0],
            }
        )
        normalized = add_normalized_columns(
            frame,
            group_columns=["family", "workflow_size"],
            deadline_group_columns=["family", "workflow_size", "seed"],
        )
        self.assertEqual(normalized["provisioning_cost"].tolist(), [2.0, 4.0])
        self.assertEqual(
            normalized["provisioning_cost_normalized"].tolist(),
            [0.5, 1.0],
        )
        self.assertEqual(normalized["deadline_ratio"].tolist(), [1.0, 2.0])
        self.assertEqual(
            normalized["deadline_normalized_0_1"].tolist(),
            [0.0, 1.0],
        )


if __name__ == "__main__":
    unittest.main()
