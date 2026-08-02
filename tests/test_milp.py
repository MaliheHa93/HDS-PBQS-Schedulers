"""Small, independently checkable MILP tests."""

from __future__ import annotations

import unittest

from hds.milp import MILPConfig, solve_bos_milp
from hds.models import SFC
from hds.topology import paper_base_topology
from hds.vm_pool import VMPool, paper_vm_types


def sfc(identifier: str) -> SFC:
    return SFC(
        id=identifier,
        workflow_id="manual",
        vnf_ids=(identifier,),
        workload_mi=8000,
        cpu_mips=1000,
        ram_mb=128,
        incoming_bandwidth_mbps=8,
        outgoing_bandwidth_mbps=8,
        bandwidth_mbps=8,
        deadline_s=20,
        source_data_mb=1,
    )


class MILPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.topology = paper_base_topology()
        self.pool = VMPool(self.topology, paper_vm_types())
        self.sfcs = [sfc("s1"), sfc("s2")]
        self.candidates = self.pool.candidate_instances(0, 10)
        self.inputs = {"s1": "edge", "s2": "edge"}

    def solve(self, sharable: bool):
        return solve_bos_milp(
            self.sfcs,
            self.candidates,
            self.topology,
            release_s=0,
            input_locations=self.inputs,
            residual_node_capacities=self.pool.residual_node_capacities(0),
            config=MILPConfig(sharable=sharable, time_limit_s=5),
        )

    def test_sharable_solution_is_non_overlapping(self) -> None:
        result = self.solve(True)
        self.assertEqual(result.status, "optimal")
        self.assertEqual(len(result.assignments), 2)
        by_vm: dict[str, list] = {}
        for assignment in result.assignments:
            by_vm.setdefault(assignment.candidate_id, []).append(assignment)
        for assignments in by_vm.values():
            assignments.sort(key=lambda item: item.start_s)
            for first, second in zip(assignments, assignments[1:]):
                self.assertLessEqual(
                    first.completion_s, second.start_s + 1e-7
                )
        self.assertGreater(result.big_m, 0)
        self.assertEqual(
            result.timing_policy,
            "earliest_start_reconstruction",
        )
        by_vm = {}
        for assignment in result.assignments:
            by_vm.setdefault(assignment.candidate_id, []).append(assignment)
        for assignments in by_vm.values():
            assignments.sort(key=lambda item: item.start_s)
            for first, second in zip(assignments, assignments[1:]):
                self.assertAlmostEqual(
                    first.completion_s,
                    second.start_s,
                    places=7,
                )

    def test_nonsharable_uses_distinct_candidates(self) -> None:
        result = self.solve(False)
        self.assertEqual(result.status, "optimal")
        selected = {item.candidate_id for item in result.assignments}
        self.assertEqual(len(selected), 2)

    def test_nonsharable_omits_redundant_ordering_variables(self) -> None:
        sharable = self.solve(True)
        nonsharable = self.solve(False)
        pairwise_variables = len(self.candidates)
        self.assertEqual(
            sharable.variable_count - nonsharable.variable_count,
            pairwise_variables,
        )
        self.assertGreater(
            sharable.constraint_count,
            nonsharable.constraint_count,
        )

    def test_impossible_deadline_is_infeasible(self) -> None:
        self.sfcs[0].deadline_s = 0.0001
        self.sfcs[0].vnf_deadlines_s = (0.0001,)
        result = self.solve(True)
        self.assertEqual(result.status, "optimal")
        self.assertEqual(result.admitted_count, 1)
        self.assertEqual(
            {item.sfc_id for item in result.assignments},
            {"s2"},
        )

    def test_internal_vnf_deadline_is_enforced(self) -> None:
        self.sfcs[0].vnf_ids = ("s1-a", "s1-b")
        self.sfcs[0].vnf_workloads_mi = (4000, 4000)
        self.sfcs[0].vnf_deadlines_s = (0.01, 20.0)
        result = self.solve(True)
        self.assertEqual(result.admitted_count, 1)
        self.assertEqual(
            {item.sfc_id for item in result.assignments},
            {"s2"},
        )

    def test_terminal_output_deadline_is_enforced(self) -> None:
        self.sfcs[0].is_terminal = True
        self.sfcs[0].sink_data_mb = 10
        self.sfcs[0].workflow_deadline_s = 0.1
        result = self.solve(True)
        self.assertEqual(result.admitted_count, 1)
        self.assertEqual(
            {item.sfc_id for item in result.assignments},
            {"s2"},
        )

    def test_storage_capacity_is_enforced(self) -> None:
        self.sfcs[0].storage_mb = 600_000
        result = self.solve(True)
        self.assertEqual(result.admitted_count, 1)
        self.assertEqual(
            {item.sfc_id for item in result.assignments},
            {"s2"},
        )


if __name__ == "__main__":
    unittest.main()
