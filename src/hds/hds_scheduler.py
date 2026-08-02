"""Dynamic reuse-first Hybrid Deadline-Aware Scheduler (Algorithm 1)."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

from .milp import MILPConfig, MILPResult, solve_bos_milp
from .models import Assignment, SFC, VMInstance
from .network_admission import LinkCapacityLedger
from .timing import CandidateTiming, candidate_timing, transfer_routes
from .topology import FogTopology
from .vm_pool import VMPool, validate_reuse_policy


@dataclass(frozen=True, slots=True)
class SchedulerBatch:
    assignments: tuple[Assignment, ...]
    deferred_sfc_ids: tuple[str, ...]
    runtime_s: float
    milp_result: MILPResult | None = None
    strategy: str = "unspecified"


@dataclass(frozen=True, slots=True)
class HDSConfig:
    sharable: bool = True
    alpha: float = 0.7
    beta: float = 0.2
    gamma: float = 0.1
    candidate_count: int = 20
    solver_time_limit_s: float = 15.0
    solver_relative_gap: float = 0.0
    maximum_billing_periods: int = 20
    enable_vm_reuse: bool = True
    reuse_policy: str = "idle_only"
    joint_bos_optimization: bool = True
    adaptive_bos_fallback: bool = True
    reconstruct_earliest_start: bool = True

    def __post_init__(self) -> None:
        if min(self.alpha, self.beta, self.gamma) < 0:
            raise ValueError("HDS weights cannot be negative")
        if not math.isclose(
            self.alpha + self.beta + self.gamma,
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError("HDS weights must sum to one")
        if self.candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        if self.solver_time_limit_s <= 0:
            raise ValueError("solver_time_limit_s must be positive")
        if self.solver_relative_gap < 0:
            raise ValueError("solver_relative_gap cannot be negative")
        if self.maximum_billing_periods <= 0:
            raise ValueError("maximum_billing_periods must be positive")
        validate_reuse_policy(self.reuse_policy)


class HDSScheduler:
    """Paper Algorithm 1 with idle reuse and lexicographic BoS admission."""

    name = "HDS"

    def __init__(
        self,
        topology: FogTopology,
        vm_pool: VMPool,
        config: HDSConfig | None = None,
    ) -> None:
        self.topology = topology
        self.vm_pool = vm_pool
        self.config = config or HDSConfig()

    def _timing(
        self,
        sfc: SFC,
        instance: VMInstance,
        release_s: float,
        transfers: tuple[tuple[str, float], ...],
        *,
        reused: bool,
    ) -> CandidateTiming:
        return candidate_timing(
            sfc=sfc,
            instance=instance,
            topology=self.topology,
            release_s=release_s,
            transfers=transfers,
            available_at_s=(
                instance.available_at_s if reused else release_s
            ),
            startup_time_s=(
                0.0 if reused else instance.vm_type.startup_time_s
            ),
        )

    def _routes(
        self,
        sfc: SFC,
        transfers: tuple[tuple[str, float], ...],
        node_id: str,
    ) -> tuple[tuple[str, str, float], ...]:
        return transfer_routes(
            sfc=sfc,
            transfers=transfers,
            candidate_node=node_id,
            edge_device_id=self.topology.edge_device_id,
        )

    def _primary_source(
        self,
        transfers: tuple[tuple[str, float], ...],
        node_id: str,
    ) -> str:
        if not transfers:
            return self.topology.edge_device_id
        return max(
            transfers,
            key=lambda item: (
                self.topology.transfer_time_s(
                    item[0],
                    node_id,
                    item[1],
                )
                if item[1] > 0 and item[0] != node_id
                else 0.0
            ),
        )[0]

    def _assignment(
        self,
        *,
        sfc: SFC,
        instance: VMInstance,
        timing: CandidateTiming,
        transfers: tuple[tuple[str, float], ...],
        reused: bool,
        solver_status: str = "not_applicable",
    ) -> Assignment:
        return Assignment(
            sfc_id=sfc.id,
            vm_id=instance.id,
            node_id=instance.node_id,
            start_s=timing.start_s,
            finish_s=timing.finish_s,
            communication_latency_s=timing.input_delay_s,
            transferred_data_mb=(
                sum(data for _source, data in transfers)
                + (sfc.sink_data_mb if sfc.is_terminal else 0.0)
            ),
            reused_vm=reused,
            scheduler=self.name,
            solver_status=solver_status,
            source_node_id=self._primary_source(
                transfers,
                instance.node_id,
            ),
            propagation_latency_s=timing.input_propagation_s,
            serialization_delay_s=timing.input_serialization_s,
            queueing_delay_s=timing.queueing_delay_s,
            vnf_completion_times_s=timing.vnf_completion_times_s,
            terminal_delivery_s=(
                timing.terminal_delivery_s if sfc.is_terminal else None
            ),
        )

    def _reuse_first(
        self,
        sfcs: list[SFC],
        release_s: float,
        link_ledger: LinkCapacityLedger,
        input_transfers: dict[str, tuple[tuple[str, float], ...]],
    ) -> tuple[list[Assignment], list[SFC]]:
        if not self.config.enable_vm_reuse:
            return [], list(sfcs)
        reusable = self.vm_pool.reusable_instances(
            release_s,
            self.config.reuse_policy,
        )
        assignments: list[Assignment] = []
        remaining: list[SFC] = []
        for sfc in sorted(sfcs, key=lambda item: (item.deadline_s, item.id)):
            options: list[
                tuple[
                    float,
                    float,
                    str,
                    VMInstance,
                    CandidateTiming,
                    tuple[tuple[str, str, float], ...],
                ]
            ] = []
            for instance in reusable:
                transfers = input_transfers[sfc.id]
                routes = self._routes(sfc, transfers, instance.node_id)
                timing = self._timing(
                    sfc,
                    instance,
                    release_s,
                    transfers,
                    reused=True,
                )
                if (
                    self.vm_pool.resource_feasible(sfc, instance)
                    and link_ledger.feasible_routes(routes)
                    and instance.paid_until_s is not None
                    and timing.finish_s
                    <= instance.paid_until_s + 1e-9
                    and timing.deadline_feasible
                ):
                    options.append(
                        (
                            instance.vm_type.cost_per_period,
                            timing.finish_s,
                            instance.id,
                            instance,
                            timing,
                            routes,
                        )
                    )
            if not options:
                remaining.append(sfc)
                continue
            _cost, _finish, _id, selected, timing, routes = min(options)
            selected.reserve(sfc.id, timing.start_s, timing.finish_s)
            link_ledger.reserve_routes(routes)
            # HDS reuse is idle-only: an instance used in this round is no
            # longer an idle candidate for another ready SFC.
            reusable.remove(selected)
            assignments.append(
                self._assignment(
                    sfc=sfc,
                    instance=selected,
                    timing=timing,
                    transfers=input_transfers[sfc.id],
                    reused=True,
                )
            )
        return assignments, remaining

    def _single_new_vm(
        self,
        sfc: SFC,
        release_s: float,
        transfers: tuple[tuple[str, float], ...],
        link_ledger: LinkCapacityLedger,
    ) -> Assignment | None:
        options: list[
            tuple[
                float,
                float,
                str,
                VMInstance,
                CandidateTiming,
                int,
                tuple[tuple[str, str, float], ...],
            ]
        ] = []
        for candidate in self.vm_pool.candidate_instances(
            release_s,
            self.config.candidate_count,
        ):
            if not self.vm_pool.can_host_new(
                candidate.node_id,
                candidate.vm_type,
                release_s,
            ):
                continue
            if not self.vm_pool.resource_feasible(sfc, candidate):
                continue
            routes = self._routes(sfc, transfers, candidate.node_id)
            if not link_ledger.feasible_routes(routes):
                continue
            timing = self._timing(
                sfc,
                candidate,
                release_s,
                transfers,
                reused=False,
            )
            if not timing.deadline_feasible:
                continue
            periods = self.vm_pool.periods_for_duration(
                candidate,
                timing.finish_s - release_s,
            )
            if periods > self.config.maximum_billing_periods:
                continue
            cost = periods * candidate.vm_type.cost_per_period
            options.append(
                (
                    cost,
                    timing.input_delay_s + timing.output_delay_s,
                    candidate.id,
                    candidate,
                    timing,
                    periods,
                    routes,
                )
            )
        if not options:
            return None
        _cost, _network, _id, candidate, timing, periods, routes = min(options)
        instance = self.vm_pool.activate(candidate, release_s, periods)
        instance.reserve(sfc.id, timing.start_s, timing.finish_s)
        link_ledger.reserve_routes(routes)
        return self._assignment(
            sfc=sfc,
            instance=instance,
            timing=timing,
            transfers=transfers,
            reused=False,
        )

    def _independent_fallback(
        self,
        sfcs: list[SFC],
        release_s: float,
        input_transfers: dict[str, tuple[tuple[str, float], ...]],
        link_ledger: LinkCapacityLedger,
    ) -> tuple[list[Assignment], list[str]]:
        assignments: list[Assignment] = []
        deferred: list[str] = []
        for sfc in sorted(sfcs, key=lambda item: (item.deadline_s, item.id)):
            assignment = self._single_new_vm(
                sfc,
                release_s,
                input_transfers[sfc.id],
                link_ledger,
            )
            if assignment is None:
                deferred.append(sfc.id)
            else:
                assignments.append(assignment)
        return assignments, deferred

    def schedule(
        self,
        sfcs: list[SFC],
        release_s: float,
        input_locations: dict[str, str],
        input_data_mb: dict[str, float],
        input_transfers: dict[
            str,
            tuple[tuple[str, float], ...],
        ] | None = None,
    ) -> SchedulerBatch:
        started = time.perf_counter()
        input_transfers = input_transfers or {
            sfc.id: ((input_locations[sfc.id], input_data_mb[sfc.id]),)
            for sfc in sfcs
        }
        link_ledger = LinkCapacityLedger.full_capacity(self.topology)
        reuse_assignments, remaining = self._reuse_first(
            sfcs,
            release_s,
            link_ledger,
            input_transfers,
        )
        if not remaining:
            return SchedulerBatch(
                tuple(reuse_assignments),
                (),
                time.perf_counter() - started,
                strategy="reuse_only",
            )
        if len(remaining) == 1:
            assignment = self._single_new_vm(
                remaining[0],
                release_s,
                input_transfers[remaining[0].id],
                link_ledger,
            )
            new_assignments = [] if assignment is None else [assignment]
            deferred = (
                (remaining[0].id,) if assignment is None else ()
            )
            return SchedulerBatch(
                tuple(reuse_assignments + new_assignments),
                deferred,
                time.perf_counter() - started,
                strategy=(
                    "mixed_reuse_single"
                    if reuse_assignments
                    else "single_fallback"
                ),
            )

        if not self.config.joint_bos_optimization:
            independent, deferred = self._independent_fallback(
                remaining,
                release_s,
                input_transfers,
                link_ledger,
            )
            return SchedulerBatch(
                tuple(reuse_assignments + independent),
                tuple(deferred),
                time.perf_counter() - started,
                strategy="independent_fallback",
            )

        candidates = self.vm_pool.candidate_instances(
            release_s,
            self.config.candidate_count,
        )
        result = solve_bos_milp(
            sfcs=remaining,
            candidates=candidates,
            topology=self.topology,
            release_s=release_s,
            input_locations=input_locations,
            input_data_mb=input_data_mb,
            input_transfers=input_transfers,
            residual_node_capacities=self.vm_pool.residual_node_capacities(
                release_s
            ),
            residual_link_bandwidth=link_ledger.snapshot(),
            config=MILPConfig(
                alpha=self.config.alpha,
                beta=self.config.beta,
                gamma=self.config.gamma,
                sharable=self.config.sharable,
                time_limit_s=self.config.solver_time_limit_s,
                relative_gap=self.config.solver_relative_gap,
                maximum_billing_periods=self.config.maximum_billing_periods,
                reconstruct_earliest_start=(
                    self.config.reconstruct_earliest_start
                ),
            ),
        )
        if not result.feasible:
            if self.config.adaptive_bos_fallback:
                independent, deferred = self._independent_fallback(
                    remaining,
                    release_s,
                    input_transfers,
                    link_ledger,
                )
                return SchedulerBatch(
                    tuple(reuse_assignments + independent),
                    tuple(deferred),
                    time.perf_counter() - started,
                    result,
                    "adaptive_independent_after_zero_admission",
                )
            return SchedulerBatch(
                tuple(reuse_assignments),
                tuple(sfc.id for sfc in remaining),
                time.perf_counter() - started,
                result,
                "joint_milp_zero_admission",
            )

        candidate_by_id = {
            candidate.id: candidate for candidate in candidates
        }
        instance_by_candidate = {
            candidate_id: self.vm_pool.activate(
                candidate_by_id[candidate_id],
                release_s,
                periods,
            )
            for candidate_id, periods in result.activated_periods.items()
        }
        remaining_by_id = {sfc.id: sfc for sfc in remaining}
        new_assignments: list[Assignment] = []
        by_candidate: dict[str, list] = {}
        for plan in result.assignments:
            by_candidate.setdefault(plan.candidate_id, []).append(plan)
        for candidate_id, plans in sorted(by_candidate.items()):
            instance = instance_by_candidate[candidate_id]
            for plan in sorted(
                plans,
                key=lambda item: (item.start_s, item.sfc_id),
            ):
                sfc = remaining_by_id[plan.sfc_id]
                transfers = input_transfers[plan.sfc_id]
                timing = self._timing(
                    sfc,
                    instance,
                    release_s,
                    transfers,
                    reused=True,
                )
                # Preserve the MILP/reconstructed timing and its exact VNF
                # completions rather than recomputing a different queue order.
                timing = CandidateTiming(
                    start_s=plan.start_s,
                    finish_s=plan.completion_s,
                    vnf_completion_times_s=plan.vnf_completion_times_s,
                    terminal_delivery_s=plan.terminal_delivery_s,
                    input_delay_s=timing.input_delay_s,
                    input_propagation_s=timing.input_propagation_s,
                    input_serialization_s=timing.input_serialization_s,
                    output_delay_s=(
                        plan.terminal_delivery_s - plan.completion_s
                    ),
                    queueing_delay_s=max(
                        0.0,
                        plan.start_s
                        - (release_s + timing.input_delay_s),
                    ),
                    laxity_s=min(
                        [
                            deadline - completion
                            for deadline, completion in zip(
                                sfc.vnf_deadlines_s,
                                plan.vnf_completion_times_s,
                            )
                        ]
                        + (
                            [
                                sfc.workflow_deadline_s
                                - plan.terminal_delivery_s
                            ]
                            if sfc.is_terminal
                            else []
                        )
                    ),
                    deadline_feasible=True,
                )
                instance.reserve(
                    plan.sfc_id,
                    plan.start_s,
                    plan.completion_s,
                )
                link_ledger.reserve_routes(
                    self._routes(sfc, transfers, instance.node_id)
                )
                new_assignments.append(
                    self._assignment(
                        sfc=sfc,
                        instance=instance,
                        timing=timing,
                        transfers=transfers,
                        reused=False,
                        solver_status=result.status,
                    )
                )

        assigned_ids = {item.sfc_id for item in new_assignments}
        deferred = tuple(
            sfc.id for sfc in remaining if sfc.id not in assigned_ids
        )
        partial = bool(deferred)
        return SchedulerBatch(
            tuple(reuse_assignments + new_assignments),
            deferred,
            time.perf_counter() - started,
            result,
            (
                "mixed_reuse_joint_partial_admission"
                if reuse_assignments and partial
                else "mixed_reuse_joint_milp"
                if reuse_assignments
                else "joint_milp_partial_admission"
                if partial
                else "joint_milp"
            ),
        )
