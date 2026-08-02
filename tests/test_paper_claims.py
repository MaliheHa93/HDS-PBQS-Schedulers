"""Regression tests for directionally honest manuscript claim wording."""

from __future__ import annotations

import unittest

from experiments.summarize_paper_claims import manuscript_markdown


def scheduler_effects(
    *,
    cost: float,
    transfer: float,
    delay: float,
    cpu_points: float,
    ram_points: float,
) -> dict[str, dict[str, float]]:
    return {
        "provisioning_cost": {"sharable_relative_change_pct": cost},
        "network_data_mb": {"sharable_relative_change_pct": transfer},
        "end_to_end_delay_s": {"sharable_relative_change_pct": delay},
        "cpu_utilization": {
            "sharable_absolute_change": cpu_points / 100.0
        },
        "ram_utilization": {
            "sharable_absolute_change": ram_points / 100.0
        },
    }


class PaperClaimTests(unittest.TestCase):
    def test_opposite_transfer_directions_are_not_collapsed_into_range(
        self,
    ) -> None:
        payload = {
            "sharing_effects": {
                "schedulers": {
                    "HDS": scheduler_effects(
                        cost=-2.5,
                        transfer=0.9,
                        delay=1.4,
                        cpu_points=0.19,
                        ram_points=0.46,
                    ),
                    "PBQS": scheduler_effects(
                        cost=-26.4,
                        transfer=-20.6,
                        delay=20.8,
                        cpu_points=5.12,
                        ram_points=9.32,
                    ),
                }
            },
            "audit": {
                "datasets": {
                    "deployment": {"solver_limit_rows": 0},
                    "bos": {"solver_limit_rows": 45},
                },
                "deployment_maximum": {
                    "fog_nodes": 60,
                    "workflow_vnfs": 1000,
                },
                "bos_maximum_width": 20,
            },
        }
        markdown = manuscript_markdown(payload)
        self.assertIn(
            "increases transferred-data volume by 0.9\\%",
            markdown,
        )
        self.assertIn(
            "reduces transferred-data volume by 20.6\\%",
            markdown,
        )
        self.assertNotIn("by -", markdown)


if __name__ == "__main__":
    unittest.main()
