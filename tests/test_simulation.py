"""End-to-end scheduler and metric tests."""

from __future__ import annotations

import math
import unittest

from hds.controlled_workloads import independent_bos_workflow
from hds.experiment import apply_deadline_policy, run_case, run_workflow
from hds.hds_scheduler import HDSConfig, HDSScheduler
from hds.models import SFC
from hds.pbqs_scheduler import PBQSConfig, PBQSScheduler
from hds.topology import paper_base_topology, paper_scaled_topology
from hds.vm_pool import VMPool, paper_vm_types


class SimulationTests(unittest.TestCase):
    def test_scheduler_weight_constraints_match_paper(self) -> None:
        with self.assertRaises(ValueError):
            HDSConfig(alpha=0.8, beta=0.2, gamma=0.1)
        with self.assertRaises(ValueError):
            PBQSConfig(omega_u=0.8, omega_w=0.3)

    def test_all_four_scheduler_configurations_complete(self) -> None:
        for scheduler in ("hds", "pbqs"):
            for sharable in (True, False):
                with self.subTest(scheduler=scheduler, sharable=sharable):
                    result = run_case(
                        "montage",
                        25,
                        6,
                        3,
                        scheduler,
                        sharable,
                        solver_time_limit_s=5,
                    )
                    self.assertFalse(result.unfinished_sfc_ids)
                    self.assertEqual(
                        result.metrics.completed_sfc_count,
                        result.metrics.total_sfc_count,
                    )
                    self.assertTrue(
                        math.isfinite(result.metrics.provisioning_cost)
                    )

    def test_identical_seed_is_deterministic_in_decisions(self) -> None:
        first = run_case("inspiral", 25, 6, 8, "hds", True)
        second = run_case("inspiral", 25, 6, 8, "hds", True)
        key = lambda assignment: (
            assignment.sfc_id,
            assignment.node_id,
            round(assignment.start_s, 8),
            round(assignment.finish_s, 8),
        )
        self.assertEqual(
            [key(item) for item in first.assignments],
            [key(item) for item in second.assignments],
        )

    def test_reference_deadlines_create_failure_success_transition(self) -> None:
        tight = run_case(
            "montage",
            25,
            0.8,
            2,
            "hds",
            True,
            deadline_mode="reference",
        )
        relaxed = run_case(
            "montage",
            25,
            3.0,
            2,
            "hds",
            True,
            deadline_mode="reference",
        )
        self.assertEqual(tight.metrics.workflow_deadline_success, 0)
        self.assertEqual(relaxed.metrics.workflow_deadline_success, 1)
        self.assertLess(
            tight.metrics.accepted_sfc_ratio,
            relaxed.metrics.accepted_sfc_ratio,
        )

    def test_scaled_topology_has_table_three_size(self) -> None:
        for local, global_, expected in ((3, 2, 5), (8, 4, 12), (15, 10, 25)):
            with self.subTest(local=local, global_=global_):
                topology = paper_scaled_topology(local, global_)
                self.assertEqual(len(topology.nodes), expected)

    def test_scaled_topology_keeps_table_two_latency_range(self) -> None:
        topology = paper_scaled_topology(15, 10)
        local_global_latencies = [
            link.latency_s
            for link in topology.links.values()
            if link.source.startswith("local-")
            and link.destination.startswith("global-")
        ]
        self.assertTrue(local_global_latencies)
        self.assertGreaterEqual(min(local_global_latencies), 0.002)
        self.assertLessEqual(max(local_global_latencies), 0.003)

    def test_billing_rounding(self) -> None:
        topology = paper_scaled_topology(3, 2)
        pool = VMPool(topology, paper_vm_types())
        candidate = pool.candidate_instances(0, 1)[0]
        self.assertEqual(pool.periods_for_duration(candidate, 0.1), 1)
        self.assertEqual(pool.periods_for_duration(candidate, 60.0), 1)
        self.assertEqual(pool.periods_for_duration(candidate, 60.001), 2)

    def test_candidate_slots_are_unique_and_capacity_bounded(self) -> None:
        topology = paper_scaled_topology(3, 2)
        pool = VMPool(topology, paper_vm_types())
        candidates = pool.candidate_instances(0.0, 100)
        self.assertEqual(
            len(candidates), len({item.id for item in candidates})
        )
        slot_keys = {
            (
                item.node_id,
                item.vm_type.id,
                item.id.rsplit("-slot", 1)[-1],
            )
            for item in candidates
        }
        self.assertEqual(len(candidates), len(slot_keys))
        self.assertLess(len(candidates), 100)

    def test_candidate_provenance_reports_requested_and_effective_counts(
        self,
    ) -> None:
        result = run_case(
            "montage",
            25,
            6,
            1,
            "hds",
            True,
            candidate_count=20,
        )
        self.assertEqual(result.requested_candidate_count, 20)
        self.assertEqual(result.initial_effective_candidate_count, 5)
        for record in result.milp_records:
            self.assertGreater(int(record["candidate_count"]), 0)
            self.assertLessEqual(int(record["candidate_count"]), 5)

    def test_hds_idle_reuse_does_not_queue_second_ready_sfc(self) -> None:
        topology = paper_scaled_topology(3, 2)
        pool = VMPool(topology, paper_vm_types())
        candidate = pool.candidate_instances(0, 1)[0]
        instance = pool.activate(candidate, 0.0, periods=1)
        instance.reserve("earlier", 0.0, 1.0)
        scheduler = HDSScheduler(
            topology,
            pool,
            HDSConfig(sharable=True, candidate_count=5),
        )
        sfcs = [
            SFC(
                id=f"sfc-{index}",
                workflow_id="manual",
                vnf_ids=(f"v-{index}",),
                workload_mi=8000,
                cpu_mips=1000,
                ram_mb=128,
                incoming_bandwidth_mbps=8,
                outgoing_bandwidth_mbps=8,
                bandwidth_mbps=8,
                deadline_s=20,
            )
            for index in range(2)
        ]
        batch = scheduler.schedule(
            sfcs,
            release_s=1.0,
            input_locations={sfc.id: "edge" for sfc in sfcs},
            input_data_mb={sfc.id: 1.0 for sfc in sfcs},
        )
        self.assertFalse(batch.deferred_sfc_ids)
        self.assertEqual(len(batch.assignments), 2)
        self.assertEqual(
            sum(item.reused_vm for item in batch.assignments),
            1,
        )
        self.assertEqual(
            len({assignment.vm_id for assignment in batch.assignments}),
            2,
        )
        self.assertTrue(
            all(
                math.isclose(
                    item.communication_latency_s,
                    item.propagation_latency_s
                    + item.serialization_delay_s,
                )
                for item in batch.assignments
            )
        )

    def test_hds_idle_and_pbqs_queue_aware_reuse_are_distinct(self) -> None:
        topology = paper_scaled_topology(3, 2)
        outcomes = []
        for scheduler_class, config in (
            (
                HDSScheduler,
                HDSConfig(
                    sharable=True,
                    candidate_count=5,
                    reuse_policy="idle_only",
                ),
            ),
            (
                PBQSScheduler,
                PBQSConfig(
                    sharable=True,
                    candidate_count=5,
                    reuse_policy="queue_aware",
                ),
            ),
        ):
            pool = VMPool(topology, paper_vm_types())
            candidate = pool.candidate_instances(0, 1)[0]
            instance = pool.activate(candidate, 0.0, periods=1)
            instance.reserve("busy", 0.0, 5.0)
            scheduler = scheduler_class(topology, pool, config)
            item = SFC(
                id="sfc",
                workflow_id="manual",
                vnf_ids=("v",),
                workload_mi=8000,
                cpu_mips=1000,
                ram_mb=128,
                incoming_bandwidth_mbps=8,
                outgoing_bandwidth_mbps=8,
                bandwidth_mbps=8,
                deadline_s=20,
            )
            batch = scheduler.schedule(
                [item],
                release_s=1.0,
                input_locations={"sfc": "edge"},
                input_data_mb={"sfc": 1.0},
            )
            self.assertEqual(len(batch.assignments), 1)
            assignment = batch.assignments[0]
            outcomes.append(
                (
                    assignment.reused_vm,
                    round(assignment.start_s, 8),
                    round(assignment.finish_s, 8),
                )
            )
        self.assertFalse(outcomes[0][0])
        self.assertTrue(outcomes[1][0])
        self.assertLess(outcomes[0][1], outcomes[1][1])

    def test_nonsharable_reuses_idle_vm_but_not_twice_in_one_bos(self) -> None:
        topology = paper_scaled_topology(3, 2)
        for scheduler_class, config in (
            (
                HDSScheduler,
                HDSConfig(sharable=False, candidate_count=5),
            ),
            (
                PBQSScheduler,
                PBQSConfig(sharable=False, candidate_count=5),
            ),
        ):
            with self.subTest(scheduler=scheduler_class.__name__):
                pool = VMPool(topology, paper_vm_types())
                candidate = pool.candidate_instances(0.0, 1)[0]
                existing = pool.activate(candidate, 0.0, periods=1)
                existing.reserve("previous", 0.0, 1.0)
                scheduler = scheduler_class(topology, pool, config)
                items = [
                    SFC(
                        id=f"next-{index}",
                        workflow_id="manual",
                        vnf_ids=(f"v-{index}",),
                        workload_mi=8000,
                        cpu_mips=1000,
                        ram_mb=128,
                        incoming_bandwidth_mbps=8,
                        outgoing_bandwidth_mbps=8,
                        bandwidth_mbps=8,
                        deadline_s=20,
                    )
                    for index in range(2)
                ]
                batch = scheduler.schedule(
                    items,
                    release_s=1.0,
                    input_locations={
                        item.id: "edge" for item in items
                    },
                    input_data_mb={item.id: 1.0 for item in items},
                )
                self.assertEqual(len(batch.assignments), 2)
                self.assertEqual(
                    sum(item.reused_vm for item in batch.assignments),
                    1,
                )
                self.assertEqual(
                    len({item.vm_id for item in batch.assignments}),
                    2,
                )
                self.assertIn(
                    existing.id,
                    {item.vm_id for item in batch.assignments},
                )

    def test_link_capacity_is_applied_to_hds_and_pbqs_paths(self) -> None:
        topology = paper_scaled_topology(3, 2)
        for scheduler_class, config in (
            (
                HDSScheduler,
                HDSConfig(
                    sharable=True,
                    candidate_count=5,
                    enable_vm_reuse=False,
                ),
            ),
            (
                PBQSScheduler,
                PBQSConfig(
                    sharable=True,
                    candidate_count=5,
                    enable_vm_reuse=False,
                ),
            ),
        ):
            pool = VMPool(topology, paper_vm_types())
            scheduler = scheduler_class(topology, pool, config)
            sfcs = [
                SFC(
                    id=f"sfc-{index}",
                    workflow_id="manual",
                    vnf_ids=(f"v-{index}",),
                    workload_mi=1000,
                    cpu_mips=1000,
                    ram_mb=128,
                    incoming_bandwidth_mbps=60,
                    outgoing_bandwidth_mbps=60,
                    bandwidth_mbps=60,
                    deadline_s=20,
                )
                for index in range(2)
            ]
            batch = scheduler.schedule(
                sfcs,
                release_s=0.0,
                input_locations={item.id: "edge" for item in sfcs},
                input_data_mb={item.id: 10.0 for item in sfcs},
            )
            # Two transfers are legal when they use distinct edge-local links.
            self.assertEqual(len(batch.assignments), 2)
            self.assertEqual(
                len({item.node_id for item in batch.assignments}), 2
            )

    def test_separate_resource_and_network_metrics_are_bounded(self) -> None:
        result = run_case("montage", 25, 8, 2, "hds", True)
        metrics = result.metrics
        for value in (
            metrics.cpu_utilization,
            metrics.ram_utilization,
            metrics.resource_utilization,
            metrics.vm_time_utilization,
            metrics.mean_link_bandwidth_utilization,
            metrics.max_link_bandwidth_utilization,
        ):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        self.assertAlmostEqual(
            metrics.resource_utilization,
            (metrics.cpu_utilization + metrics.ram_utilization) / 2.0,
        )
        self.assertEqual(len(result.node_utilization), 5)
        self.assertEqual(
            metrics.used_link_count,
            len(result.link_utilization),
        )
        self.assertAlmostEqual(
            metrics.communication_delay_s,
            metrics.propagation_delay_s + metrics.serialization_delay_s,
        )

    def test_edf_and_hds_ablations_execute(self) -> None:
        edf = run_case("inspiral", 25, 8, 4, "edf", True)
        no_reuse = run_case(
            "inspiral",
            25,
            8,
            4,
            "hds",
            True,
            enable_vm_reuse=False,
        )
        no_joint = run_case(
            "inspiral",
            25,
            8,
            4,
            "hds",
            True,
            joint_bos_optimization=False,
        )
        self.assertFalse(edf.unfinished_sfc_ids)
        for result in (no_reuse, no_joint):
            self.assertGreater(result.metrics.completed_sfc_count, 0)
            self.assertEqual(
                result.metrics.completed_sfc_count
                + len(result.unfinished_sfc_ids),
                result.metrics.total_sfc_count,
            )

    def test_adaptive_bos_fallback_preserves_feasible_subset(self) -> None:
        topology = paper_base_topology()
        workflow = apply_deadline_policy(
            independent_bos_workflow(4, seed=1),
            deadline_factor=3.0,
            topology=topology,
            deadline_mode="reference",
        )
        adaptive = run_workflow(
            workflow,
            scheduler_name="hds",
            sharable=True,
            topology=topology,
            adaptive_bos_fallback=True,
        )
        all_or_nothing = run_workflow(
            workflow,
            scheduler_name="hds",
            sharable=True,
            topology=topology,
            adaptive_bos_fallback=False,
        )
        self.assertGreaterEqual(
            adaptive.metrics.completed_sfc_count,
            all_or_nothing.metrics.completed_sfc_count,
        )
        strategies = {
            str(record["strategy"])
            for record in adaptive.scheduling_records
        }
        self.assertTrue(
            strategies
            & {
                "joint_milp",
                "joint_milp_partial_admission",
                "adaptive_independent_after_zero_admission",
            }
        )
        for record in adaptive.scheduling_records:
            if (
                record["strategy"]
                == "adaptive_independent_after_zero_admission"
            ):
                self.assertEqual(record["joint_scheduled_count"], 0)

    def test_pbqs_defers_candidate_with_no_finite_laxity(self) -> None:
        topology = paper_base_topology()
        pool = VMPool(topology, paper_vm_types())
        scheduler = PBQSScheduler(
            topology,
            pool,
            PBQSConfig(candidate_count=5),
        )
        feasible = SFC(
            id="feasible",
            workflow_id="manual",
            vnf_ids=("f",),
            workload_mi=1000,
            cpu_mips=1000,
            ram_mb=128,
            incoming_bandwidth_mbps=8,
            outgoing_bandwidth_mbps=8,
            bandwidth_mbps=8,
            deadline_s=20,
            vnf_deadlines_s=(20,),
        )
        impossible = SFC(
            id="impossible",
            workflow_id="manual",
            vnf_ids=("i",),
            workload_mi=8000,
            cpu_mips=1000,
            ram_mb=128,
            incoming_bandwidth_mbps=8,
            outgoing_bandwidth_mbps=8,
            bandwidth_mbps=8,
            deadline_s=0.01,
            vnf_deadlines_s=(0.01,),
        )
        batch = scheduler.schedule(
            [feasible, impossible],
            release_s=0,
            input_locations={"feasible": "edge", "impossible": "edge"},
            input_data_mb={"feasible": 1.0, "impossible": 1.0},
        )
        self.assertEqual(
            {item.sfc_id for item in batch.assignments},
            {"feasible"},
        )
        self.assertEqual(batch.deferred_sfc_ids, ("impossible",))

    def test_controlled_bos_records_first_round_admission(self) -> None:
        topology = paper_base_topology()
        workflow = apply_deadline_policy(
            independent_bos_workflow(8, seed=2),
            deadline_factor=5.0,
            topology=topology,
            deadline_mode="reference",
        )
        result = run_workflow(
            workflow,
            scheduler_name="hds",
            sharable=False,
            topology=topology,
            candidate_count=5,
        )
        self.assertEqual(result.first_round_submitted_sfc_count, 8)
        self.assertLessEqual(result.first_round_admitted_sfc_count, 5)
        self.assertGreaterEqual(
            result.metrics.completed_sfc_count,
            result.first_round_admitted_sfc_count,
        )


if __name__ == "__main__":
    unittest.main()
