"""Tests for workflow validation, SFC construction, and TASTD."""

from __future__ import annotations

import unittest

from hds.models import VNF, Workflow, WorkflowEdge
from hds.sfc_builder import SFCBuilder
from hds.tastd import InfeasibleWorkflowError, assign_tastd_deadlines
from hds.validation import validate_sfc_partition


def task(task_id: str, workload: float = 1000) -> VNF:
    return VNF(task_id, workload, 500, 128)


class StaticPhaseTests(unittest.TestCase):
    def test_linear_workflow_becomes_one_sfc(self) -> None:
        workflow = Workflow(
            "linear",
            {name: task(name) for name in "abc"},
            [
                WorkflowEdge("a", "b", 1),
                WorkflowEdge("b", "c", 1),
            ],
            deadline_s=20,
        )
        graph = SFCBuilder(1000).build(workflow)
        self.assertEqual(len(graph.sfcs), 1)
        only = next(iter(graph.sfcs.values()))
        self.assertEqual(only.vnf_ids, ("a", "b", "c"))
        validate_sfc_partition(workflow, graph)

    def test_diamond_join_starts_new_sfc_and_retains_all_parents(self) -> None:
        workflow = Workflow(
            "diamond",
            {
                "a": task("a"),
                "b": task("b", 1000),
                "c": task("c", 2000),
                "d": task("d"),
                "e": task("e"),
            },
            [
                WorkflowEdge("a", "b", 1),
                WorkflowEdge("a", "c", 1),
                WorkflowEdge("b", "d", 1),
                WorkflowEdge("c", "d", 1),
                WorkflowEdge("d", "e", 1),
            ],
            deadline_s=30,
        )
        graph = SFCBuilder(1000).build(workflow)
        validate_sfc_partition(workflow, graph)
        join_sfc = graph.sfcs[graph.vnf_to_sfc["d"]]
        self.assertEqual(join_sfc.vnf_ids, ("d", "e"))
        self.assertEqual(
            sum("d" in sfc.vnf_ids for sfc in graph.sfcs.values()), 1
        )
        branch_b = graph.vnf_to_sfc["b"]
        branch_c = graph.vnf_to_sfc["c"]
        self.assertNotEqual(branch_b, branch_c)
        self.assertNotEqual(graph.vnf_to_sfc["d"], branch_c)
        self.assertEqual(
            set(graph.predecessors(graph.vnf_to_sfc["d"])),
            {branch_b, branch_c},
        )
        self.assertEqual(
            set(graph.ready({graph.vnf_to_sfc["a"]})),
            {branch_b, branch_c},
        )

    def test_join_never_merges_with_either_parent(self) -> None:
        workflow = Workflow(
            "join-tie",
            {name: task(name) for name in "abcd"},
            [
                WorkflowEdge("a", "b", 0),
                WorkflowEdge("a", "c", 0),
                WorkflowEdge("b", "d", 0),
                WorkflowEdge("c", "d", 0),
            ],
            deadline_s=10,
        )
        graph = SFCBuilder(1000).build(workflow)
        self.assertEqual(
            graph.sfcs[graph.vnf_to_sfc["d"]].vnf_ids,
            ("d",),
        )
        self.assertEqual(
            set(graph.predecessors(graph.vnf_to_sfc["d"])),
            {
                graph.vnf_to_sfc["b"],
                graph.vnf_to_sfc["c"],
            },
        )

    def test_tastd_follows_paper_equations_at_a_join(self) -> None:
        workflow = Workflow(
            "paper-equations",
            {name: task(name) for name in "abcd"},
            [
                WorkflowEdge("a", "b", 0),
                WorkflowEdge("a", "c", 0),
                WorkflowEdge("b", "d", 0),
                WorkflowEdge("c", "d", 0),
            ],
            deadline_s=10,
        )
        graph = SFCBuilder(1000).build(workflow)
        result = assign_tastd_deadlines(
            workflow,
            graph,
            1000,
            transfer_latency=lambda _p, _c, _d: 0.0,
            mode="downstream_reserved",
        )
        self.assertAlmostEqual(result.minimum_makespan_s, 3.0)
        self.assertAlmostEqual(result.workflow_slack_s, 7.0)
        for value in result.allocated_slack_s.values():
            self.assertAlmostEqual(value, 7.0)
        self.assertAlmostEqual(result.vnf_deadlines_s["a"], 8.0)
        self.assertAlmostEqual(result.vnf_deadlines_s["b"], 9.0)
        self.assertAlmostEqual(result.vnf_deadlines_s["c"], 9.0)
        self.assertAlmostEqual(result.vnf_deadlines_s["d"], 10.0)

    def test_tastd_allocates_all_slack_and_monotone_deadlines(self) -> None:
        workflow = Workflow(
            "tastd",
            {name: task(name, 1000) for name in "abc"},
            [
                WorkflowEdge("a", "b", 0),
                WorkflowEdge("b", "c", 0),
            ],
            deadline_s=10,
        )
        graph = SFCBuilder(1000).build(workflow)
        result = assign_tastd_deadlines(
            workflow,
            graph,
            1000,
            eta=1.0,
            transfer_latency=lambda _p, _c, _d: 0.0,
        )
        self.assertAlmostEqual(result.minimum_makespan_s, 3.0)
        for value in result.allocated_slack_s.values():
            self.assertAlmostEqual(value, 7.0)
        self.assertLess(
            result.vnf_deadlines_s["a"], result.vnf_deadlines_s["b"]
        )
        self.assertLess(
            result.vnf_deadlines_s["b"], result.vnf_deadlines_s["c"]
        )
        self.assertAlmostEqual(result.vnf_deadlines_s["c"], 10.0)

    def test_tastd_rejects_infeasible_workflow(self) -> None:
        workflow = Workflow(
            "bad",
            {"a": task("a", 2000)},
            [],
            deadline_s=1,
        )
        graph = SFCBuilder(1000).build(workflow)
        with self.assertRaises(InfeasibleWorkflowError):
            assign_tastd_deadlines(workflow, graph, 1000)

    def test_allowed_infeasible_case_uses_signed_paper_slack(self) -> None:
        workflow = Workflow(
            "signed-slack",
            {name: task(name) for name in "abc"},
            [
                WorkflowEdge("a", "b", 0),
                WorkflowEdge("b", "c", 0),
            ],
            deadline_s=2.4,
        )
        graph = SFCBuilder(1000).build(workflow)
        result = assign_tastd_deadlines(
            workflow,
            graph,
            1000,
            eta=1.0,
            transfer_latency=lambda _p, _c, _d: 0.0,
            allow_infeasible=True,
        )
        self.assertFalse(result.reference_feasible)
        self.assertAlmostEqual(result.workflow_slack_s, -0.6)
        for value in result.allocated_slack_s.values():
            self.assertAlmostEqual(value, -0.6)
        self.assertAlmostEqual(result.vnf_deadlines_s["a"], 0.4)
        self.assertAlmostEqual(result.vnf_deadlines_s["b"], 1.4)
        self.assertAlmostEqual(result.vnf_deadlines_s["c"], 2.4)

    def test_tastd_rejects_zero_eta_as_specified_in_paper(self) -> None:
        workflow = Workflow(
            "eta",
            {"a": task("a")},
            [],
            deadline_s=2,
        )
        graph = SFCBuilder(1000).build(workflow)
        with self.assertRaises(ValueError):
            assign_tastd_deadlines(workflow, graph, 1000, eta=0.0)

    def test_legacy_topology_mode_uses_downstream_reserved_rule(self) -> None:
        workflow = Workflow(
            "ablation",
            {
                "a": task("a", 1000),
                "b": task("b", 2000),
                "c": task("c", 1000),
            },
            [
                WorkflowEdge("a", "b", 0),
                WorkflowEdge("a", "c", 0),
            ],
            deadline_s=10,
        )
        graph = SFCBuilder(1000).build(workflow)
        result = assign_tastd_deadlines(
            workflow,
            graph,
            1000,
            transfer_latency=lambda _p, _c, _d: 0.0,
            mode="topology",
        )
        self.assertEqual(result.mode, "downstream_reserved")
        self.assertEqual(result.weights, {})

    def test_terminal_output_time_is_reserved_at_sink(self) -> None:
        workflow = Workflow(
            "terminal-output",
            {
                "a": VNF(
                    "a",
                    workload_mi=1000,
                    cpu_mips=500,
                    ram_mb=128,
                    sink_data_mb=2,
                )
            },
            [],
            deadline_s=10,
        )
        graph = SFCBuilder(1000).build(workflow)
        result = assign_tastd_deadlines(
            workflow,
            graph,
            1000,
            transfer_latency=lambda _p, _c, data: data,
            terminal_latency=lambda _task, data: data,
        )
        self.assertAlmostEqual(result.minimum_makespan_s, 3.0)
        self.assertAlmostEqual(result.remaining_reference_s["a"], 2.0)
        self.assertAlmostEqual(result.vnf_deadlines_s["a"], 8.0)

    def test_sfc_storage_includes_retained_external_outputs(self) -> None:
        workflow = Workflow(
            "retained-output",
            {
                "a": task("a"),
                "b": task("b"),
                "c": task("c"),
            },
            [
                WorkflowEdge("a", "b", 5),
                WorkflowEdge("a", "c", 7),
            ],
            deadline_s=20,
        )
        graph = SFCBuilder(1000).build(workflow)
        source_sfc = graph.sfcs[graph.vnf_to_sfc["a"]]
        self.assertAlmostEqual(source_sfc.storage_mb, 64 + 5 + 7)

    def test_cycle_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Workflow(
                "cycle",
                {name: task(name) for name in "ab"},
                [
                    WorkflowEdge("a", "b", 1),
                    WorkflowEdge("b", "a", 1),
                ],
                deadline_s=10,
            )


if __name__ == "__main__":
    unittest.main()
