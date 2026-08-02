"""Priority-Based Queuing Scheduler baseline (manuscript Algorithm 2)."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

from .hds_scheduler import SchedulerBatch
from .models import Assignment, SFC, Tier, VMInstance
from .network_admission import LinkCapacityLedger
from .timing import CandidateTiming, candidate_timing, transfer_routes
from .topology import FogTopology
from .vm_pool import VMPool, validate_reuse_policy


@dataclass(frozen=True, slots=True)
class PBQSConfig:
    omega_u: float = 0.7
    omega_w: float = 0.3
    sharable: bool = True
    candidate_count: int = 20
    maximum_billing_periods: int = 20
    epsilon: float = 1e-9
    enable_vm_reuse: bool = True
    reuse_policy: str = "queue_aware"
    placement_policy: str = "local_first"

    def __post_init__(self) -> None:
        if min(self.omega_u, self.omega_w) < 0:
            raise ValueError("PBQS weights cannot be negative")
        if abs(self.omega_u + self.omega_w - 1.0) > 1e-9:
            raise ValueError("PBQS weights must sum to one")
        if self.candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        if self.maximum_billing_periods <= 0:
            raise ValueError("maximum_billing_periods must be positive")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        validate_reuse_policy(self.reuse_policy)
        if self.placement_policy not in {"local_first", "cost_first"}:
            raise ValueError(
                "placement_policy must be local_first or cost_first"
            )


@dataclass(frozen=True, slots=True)
class _Option:
    sort_key: tuple[float | int | str, ...]
    instance: VMInstance
    timing: CandidateTiming
    periods: int
    reused: bool
    routes: tuple[tuple[str, str, float], ...]


class PBQSScheduler:
    """Greedy baseline with candidate-specific laxity (Equations 22-25)."""

    name = "PBQS"

    def __init__(
        self,
        topology: FogTopology,
        vm_pool: VMPool,
        config: PBQSConfig | None = None,
    ) -> None:
        self.topology = topology
        self.vm_pool = vm_pool
        self.config = config or PBQSConfig()
        self._priority_laxity: dict[str, float] = {}

    def _priority(self, sfcs: list[SFC], current_s: float) -> list[SFC]:
        del current_s
        finite = {
            sfc.id: self._priority_laxity[sfc.id]
            for sfc in sfcs
            if sfc.id in self._priority_laxity
            and math.isfinite(self._priority_laxity[sfc.id])
        }
        if not finite:
            return []
        inverse = {
            sfc_id: 1.0 / max(self.config.epsilon, laxity)
            for sfc_id, laxity in finite.items()
        }
        max_inverse = max(inverse.values())
        max_workload = max(
            sfc.workload_mi for sfc in sfcs if sfc.id in finite
        )
        score = {
            sfc.id: (
                self.config.omega_u * inverse[sfc.id] / max_inverse
                + self.config.omega_w * sfc.workload_mi / max_workload
            )
            for sfc in sfcs
            if sfc.id in finite
        }
        return sorted(
            (sfc for sfc in sfcs if sfc.id in score),
            key=lambda sfc: (-score[sfc.id], sfc.id),
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

    def _sort_key(
        self,
        instance: VMInstance,
        *,
        cost: float,
        network: float,
    ) -> tuple[float | int | str, ...]:
        tier = int(instance.vm_type.tier != Tier.LOCAL)
        if self.config.placement_policy == "cost_first":
            return (cost, tier, network, instance.id)
        return (tier, cost, network, instance.id)

    def _existing_options(
        self,
        sfc: SFC,
        current_s: float,
        transfers: tuple[tuple[str, float], ...],
        used_this_round: set[str],
        link_ledger: LinkCapacityLedger,
    ) -> list[_Option]:
        if not self.config.enable_vm_reuse:
            return []
        options: list[_Option] = []
        for instance in self.vm_pool.reusable_instances(
            current_s,
            self.config.reuse_policy,
        ):
            if not self.config.sharable and instance.id in used_this_round:
                continue
            if not self.vm_pool.resource_feasible(sfc, instance):
                continue
            routes = self._routes(sfc, transfers, instance.node_id)
            if not link_ledger.feasible_routes(routes):
                continue
            timing = candidate_timing(
                sfc=sfc,
                instance=instance,
                topology=self.topology,
                release_s=current_s,
                transfers=transfers,
                available_at_s=instance.available_at_s,
                startup_time_s=0.0,
            )
            if (
                timing.deadline_feasible
                and timing.finish_s
                <= (instance.paid_until_s or 0.0) + 1e-9
            ):
                options.append(
                    _Option(
                        sort_key=self._sort_key(
                            instance,
                            cost=0.0,
                            network=(
                                timing.input_delay_s
                                + timing.output_delay_s
                            ),
                        ),
                        instance=instance,
                        timing=timing,
                        periods=0,
                        reused=True,
                        routes=routes,
                    )
                )
        return options

    def _new_options(
        self,
        sfc: SFC,
        current_s: float,
        transfers: tuple[tuple[str, float], ...],
        link_ledger: LinkCapacityLedger,
    ) -> list[_Option]:
        options: list[_Option] = []
        for candidate in self.vm_pool.candidate_instances(
            current_s,
            self.config.candidate_count,
        ):
            if not self.vm_pool.can_host_new(
                candidate.node_id,
                candidate.vm_type,
                current_s,
            ):
                continue
            if not self.vm_pool.resource_feasible(sfc, candidate):
                continue
            routes = self._routes(sfc, transfers, candidate.node_id)
            if not link_ledger.feasible_routes(routes):
                continue
            timing = candidate_timing(
                sfc=sfc,
                instance=candidate,
                topology=self.topology,
                release_s=current_s,
                transfers=transfers,
                available_at_s=current_s,
                startup_time_s=candidate.vm_type.startup_time_s,
            )
            if not timing.deadline_feasible:
                continue
            periods = self.vm_pool.periods_for_duration(
                candidate,
                timing.finish_s - current_s,
            )
            if periods > self.config.maximum_billing_periods:
                continue
            cost = periods * candidate.vm_type.cost_per_period
            options.append(
                _Option(
                    sort_key=self._sort_key(
                        candidate,
                        cost=cost,
                        network=(
                            timing.input_delay_s + timing.output_delay_s
                        ),
                    ),
                    instance=candidate,
                    timing=timing,
                    periods=periods,
                    reused=False,
                    routes=routes,
                )
            )
        return options

    def _options(
        self,
        sfc: SFC,
        current_s: float,
        transfers: tuple[tuple[str, float], ...],
        used_this_round: set[str],
        link_ledger: LinkCapacityLedger,
    ) -> list[_Option]:
        return self._existing_options(
            sfc,
            current_s,
            transfers,
            used_this_round,
            link_ledger,
        ) + self._new_options(
            sfc,
            current_s,
            transfers,
            link_ledger,
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
        used_this_round: set[str] = set()

        # Equation 23: rank each SFC using the laxity of the same least-cost
        # candidate PBQS would select before any ready-set assignment.
        self._priority_laxity = {}
        initially_feasible: set[str] = set()
        for sfc in sfcs:
            options = self._options(
                sfc,
                release_s,
                input_transfers[sfc.id],
                used_this_round,
                link_ledger,
            )
            if not options:
                continue
            selected = min(options, key=lambda item: item.sort_key)
            if not math.isfinite(selected.timing.laxity_s):
                continue
            self._priority_laxity[sfc.id] = selected.timing.laxity_s
            initially_feasible.add(sfc.id)

        assignments: list[Assignment] = []
        deferred = [
            sfc.id for sfc in sfcs if sfc.id not in initially_feasible
        ]
        for sfc in self._priority(sfcs, release_s):
            transfers = input_transfers[sfc.id]
            options = self._options(
                sfc,
                release_s,
                transfers,
                used_this_round,
                link_ledger,
            )
            if not options:
                deferred.append(sfc.id)
                continue
            selected = min(options, key=lambda item: item.sort_key)
            if selected.reused:
                instance = selected.instance
            else:
                instance = self.vm_pool.activate(
                    selected.instance,
                    release_s,
                    selected.periods,
                )
            instance.reserve(
                sfc.id,
                selected.timing.start_s,
                selected.timing.finish_s,
            )
            link_ledger.reserve_routes(selected.routes)
            used_this_round.add(instance.id)
            timing = selected.timing
            assignments.append(
                Assignment(
                    sfc_id=sfc.id,
                    vm_id=instance.id,
                    node_id=instance.node_id,
                    start_s=timing.start_s,
                    finish_s=timing.finish_s,
                    communication_latency_s=timing.input_delay_s,
                    transferred_data_mb=(
                        sum(data for _source, data in transfers)
                        + (
                            sfc.sink_data_mb
                            if sfc.is_terminal
                            else 0.0
                        )
                    ),
                    reused_vm=selected.reused,
                    scheduler=self.name,
                    source_node_id=self._primary_source(
                        transfers,
                        instance.node_id,
                    ),
                    propagation_latency_s=timing.input_propagation_s,
                    serialization_delay_s=timing.input_serialization_s,
                    queueing_delay_s=timing.queueing_delay_s,
                    vnf_completion_times_s=(
                        timing.vnf_completion_times_s
                    ),
                    terminal_delivery_s=(
                        timing.terminal_delivery_s
                        if sfc.is_terminal
                        else None
                    ),
                )
            )
        return SchedulerBatch(
            tuple(assignments),
            tuple(sorted(set(deferred))),
            time.perf_counter() - started,
            strategy=(
                "cost_greedy"
                if self.config.placement_policy == "cost_first"
                else "priority_greedy"
            ),
        )
