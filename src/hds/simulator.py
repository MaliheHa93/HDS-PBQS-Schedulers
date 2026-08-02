"""Deterministic event-driven workflow/SFC simulator."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

from .hds_scheduler import HDSScheduler
from .metrics import (
    LinkUtilization,
    NodeUtilization,
    RunMetrics,
    calculate_run_metrics,
)
from .models import Assignment, SFCGraph, Workflow
from .pbqs_scheduler import PBQSScheduler
from .tastd import TASTDResult
from .vm_pool import VMPool


@dataclass(frozen=True, slots=True)
class SimulationResult:
    workflow_id: str
    scheduler: str
    assignments: tuple[Assignment, ...]
    metrics: RunMetrics
    tastd: TASTDResult
    milp_records: tuple[dict[str, float | int | str | None], ...]
    unfinished_sfc_ids: tuple[str, ...]
    node_utilization: tuple[NodeUtilization, ...]
    link_utilization: tuple[LinkUtilization, ...]
    workflow_deadline_s: float
    workflow_arrival_s: float
    requested_candidate_count: int
    initial_effective_candidate_count: int
    first_round_submitted_sfc_count: int = 0
    first_round_admitted_sfc_count: int = 0
    first_round_scheduler_runtime_s: float = 0.0
    scheduling_records: tuple[dict[str, float | int | str], ...] = ()


class WorkflowSimulator:
    """Execute one workflow using either HDS or PBQS."""

    def __init__(
        self,
        workflow: Workflow,
        graph: SFCGraph,
        tastd: TASTDResult,
        scheduler: HDSScheduler | PBQSScheduler,
        vm_pool: VMPool,
    ) -> None:
        self.workflow = workflow
        self.graph = graph
        self.tastd = tastd
        self.scheduler = scheduler
        self.vm_pool = vm_pool

    def _input_context(
        self,
        sfc_id: str,
        assignment_by_sfc: dict[str, Assignment],
    ) -> tuple[str, float]:
        predecessors = self.graph.predecessors(sfc_id)
        sfc = self.graph.sfcs[sfc_id]
        if not predecessors:
            return self.scheduler.topology.edge_device_id, sfc.source_data_mb
        # This scalar context is retained for trace compatibility only; actual
        # scheduling routes every parent transfer independently.
        latest = max(
            predecessors,
            key=lambda item: (
                assignment_by_sfc[item].finish_s,
                item,
            ),
        )
        data_mb = sum(
            self.graph.edges_mb[(parent, sfc_id)] for parent in predecessors
        )
        return assignment_by_sfc[latest].node_id, data_mb

    def _input_transfers(
        self,
        sfc_id: str,
        assignment_by_sfc: dict[str, Assignment],
    ) -> tuple[tuple[str, float], ...]:
        """Preserve every parent input as an independently routed transfer."""

        predecessors = self.graph.predecessors(sfc_id)
        if not predecessors:
            return (
                (
                    self.scheduler.topology.edge_device_id,
                    self.graph.sfcs[sfc_id].source_data_mb,
                ),
            )
        return tuple(
            (
                assignment_by_sfc[parent].node_id,
                self.graph.edges_mb[(parent, sfc_id)],
            )
            for parent in predecessors
        )

    def run(self) -> SimulationResult:
        completed: set[str] = set()
        scheduled: set[str] = set()
        assignment_by_sfc: dict[str, Assignment] = {}
        events: list[tuple[float, str]] = []
        current_s = self.workflow.arrival_s
        scheduler_runtime = 0.0
        milp_runtime = 0.0
        milp_records: list[dict[str, float | int | str | None]] = []
        scheduling_records: list[dict[str, float | int | str]] = []
        stalled_iterations = 0
        requested_candidate_count = int(
            getattr(self.scheduler.config, "candidate_count", 0)
        )
        initial_effective_candidate_count = len(
            self.vm_pool.candidate_instances(
                self.workflow.arrival_s,
                requested_candidate_count,
            )
        )
        first_round_submitted = 0
        first_round_admitted = 0
        first_round_runtime = 0.0
        scheduling_round = 0

        while len(completed) < len(self.graph.sfcs):
            ready_ids = self.graph.ready(completed, scheduled)
            assigned_this_iteration = False
            if ready_ids and current_s <= self.workflow.deadline_s + 1e-9:
                locations: dict[str, str] = {}
                input_data: dict[str, float] = {}
                input_transfers: dict[
                    str, tuple[tuple[str, float], ...]
                ] = {}
                for sfc_id in ready_ids:
                    location, data_mb = self._input_context(
                        sfc_id, assignment_by_sfc
                    )
                    locations[sfc_id] = location
                    input_data[sfc_id] = data_mb
                    input_transfers[sfc_id] = self._input_transfers(
                        sfc_id, assignment_by_sfc
                    )
                batch = self.scheduler.schedule(
                    [self.graph.sfcs[sfc_id] for sfc_id in ready_ids],
                    current_s,
                    locations,
                    input_data,
                    input_transfers,
                )
                scheduling_round += 1
                if scheduling_round == 1:
                    first_round_submitted = len(ready_ids)
                    first_round_admitted = len(batch.assignments)
                    first_round_runtime = batch.runtime_s
                scheduler_runtime += batch.runtime_s
                if batch.milp_result is not None:
                    record = batch.milp_result
                    milp_runtime += record.runtime_s
                    milp_records.append(
                        {
                            "status": record.status,
                            "runtime_s": record.runtime_s,
                            "objective": record.objective,
                            "mip_gap": record.mip_gap,
                            "node_count": record.node_count,
                            "variables": record.variable_count,
                            "constraints": record.constraint_count,
                            "big_m": record.big_m,
                            "bos_size": record.bos_size,
                            "candidate_count": record.candidate_count,
                            "timing_policy": record.timing_policy,
                            "admitted_count": record.admitted_count,
                            "admission_status": record.admission_status,
                            "secondary_status": record.secondary_status,
                            "avoidable_idle_removed_s": (
                                record.avoidable_idle_removed_s
                            ),
                            "maximum_avoidable_idle_removed_s": (
                                record.maximum_avoidable_idle_removed_s
                            ),
                        }
                    )
                reused_count = sum(
                    assignment.reused_vm
                    for assignment in batch.assignments
                )
                new_count = len(batch.assignments) - reused_count
                joint_count = (
                    new_count
                    if batch.milp_result is not None
                    and batch.milp_result.feasible
                    else 0
                )
                scheduling_records.append(
                    {
                        "release_s": current_s,
                        "ready_bos_size": len(ready_ids),
                        "milp_input_bos_size": (
                            batch.milp_result.bos_size
                            if batch.milp_result is not None
                            else 0
                        ),
                        "assigned_count": len(batch.assignments),
                        "submitted_count": len(ready_ids),
                        "admitted_count": len(batch.assignments),
                        "deferred_count": len(batch.deferred_sfc_ids),
                        "reuse_path_count": reused_count,
                        "new_vm_path_count": new_count,
                        "joint_scheduled_count": joint_count,
                        "single_fallback_count": (
                            new_count
                            if batch.strategy
                            in {
                                "single_fallback",
                                "mixed_reuse_single",
                                "independent_fallback",
                                "adaptive_independent_after_infeasible_joint_milp",
                            }
                            else 0
                        ),
                        "strategy": batch.strategy,
                    }
                )
                for assignment in batch.assignments:
                    assignment_by_sfc[assignment.sfc_id] = assignment
                    scheduled.add(assignment.sfc_id)
                    heapq.heappush(
                        events, (assignment.finish_s, assignment.sfc_id)
                    )
                    assigned_this_iteration = True

            if events:
                next_time, sfc_id = heapq.heappop(events)
                current_s = max(current_s, next_time)
                completed.add(sfc_id)
                while events and abs(events[0][0] - next_time) <= 1e-9:
                    _, simultaneous = heapq.heappop(events)
                    completed.add(simultaneous)
                stalled_iterations = 0
                continue

            if assigned_this_iteration:
                continue

            ready_unassigned = self.graph.ready(completed, scheduled)
            if ready_unassigned:
                expiries = sorted(
                    {
                        instance.paid_until_s
                        for instance in self.vm_pool.active_instances(current_s)
                        if instance.paid_until_s is not None
                        and instance.paid_until_s > current_s + 1e-9
                    }
                )
                if expiries and expiries[0] <= self.workflow.deadline_s:
                    current_s = expiries[0]
                    stalled_iterations += 1
                    if stalled_iterations < len(self.vm_pool.instances) + 2:
                        continue
            break

        assignments = sorted(
            assignment_by_sfc.values(), key=lambda item: (item.start_s, item.sfc_id)
        )
        metrics, node_utilization, link_utilization = calculate_run_metrics(
            self.workflow,
            self.graph,
            assignments,
            self.vm_pool.instances.values(),
            scheduler_runtime,
            milp_runtime,
            self.scheduler.topology,
        )
        unfinished = tuple(sorted(set(self.graph.sfcs) - completed))
        return SimulationResult(
            workflow_id=self.workflow.id,
            scheduler=self.scheduler.name,
            assignments=tuple(assignments),
            metrics=metrics,
            tastd=self.tastd,
            milp_records=tuple(milp_records),
            unfinished_sfc_ids=unfinished,
            node_utilization=node_utilization,
            link_utilization=link_utilization,
            workflow_deadline_s=self.workflow.deadline_s,
            workflow_arrival_s=self.workflow.arrival_s,
            requested_candidate_count=requested_candidate_count,
            initial_effective_candidate_count=(
                initial_effective_candidate_count
            ),
            first_round_submitted_sfc_count=first_round_submitted,
            first_round_admitted_sfc_count=first_round_admitted,
            first_round_scheduler_runtime_s=first_round_runtime,
            scheduling_records=tuple(scheduling_records),
        )
