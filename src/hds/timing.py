"""Candidate-dependent processing, transfer, and deadline calculations."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .models import SFC, VMInstance
from .topology import FogTopology


@dataclass(frozen=True, slots=True)
class CandidateTiming:
    start_s: float
    finish_s: float
    vnf_completion_times_s: tuple[float, ...]
    terminal_delivery_s: float
    input_delay_s: float
    input_propagation_s: float
    input_serialization_s: float
    output_delay_s: float
    queueing_delay_s: float
    laxity_s: float
    deadline_feasible: bool


def transfer_profile(
    topology: FogTopology,
    transfers: tuple[tuple[str, float], ...],
    destination: str,
) -> tuple[float, float, float]:
    """Return the latest input arrival's total/propagation/serialization."""

    profiles: list[tuple[float, float, float]] = []
    for source, data_mb in transfers:
        if data_mb <= 0 or source == destination:
            profiles.append((0.0, 0.0, 0.0))
            continue
        path = topology.shortest_path(source, destination)
        serialization = data_mb * 8.0 / path.bottleneck_bandwidth_mbps
        profiles.append(
            (
                path.propagation_latency_s + serialization,
                path.propagation_latency_s,
                serialization,
            )
        )
    return max(profiles, key=lambda item: item[0], default=(0.0, 0.0, 0.0))


def output_delay_s(
    topology: FogTopology,
    sfc: SFC,
    source_node: str,
) -> float:
    if not sfc.is_terminal or sfc.sink_data_mb <= 0:
        return 0.0
    return topology.transfer_time_s(
        source_node,
        topology.edge_device_id,
        sfc.sink_data_mb,
    )


def cumulative_processing_s(
    sfc: SFC,
    speed_mips: float,
) -> tuple[float, ...]:
    total = 0.0
    result = []
    for workload in sfc.vnf_workloads_mi:
        total += workload / speed_mips
        result.append(total)
    return tuple(result)


def candidate_timing(
    *,
    sfc: SFC,
    instance: VMInstance,
    topology: FogTopology,
    release_s: float,
    transfers: tuple[tuple[str, float], ...],
    available_at_s: float,
    startup_time_s: float,
) -> CandidateTiming:
    """Evaluate Equations 22-23 for one SFC/candidate pair."""

    network, propagation, serialization = transfer_profile(
        topology,
        transfers,
        instance.node_id,
    )
    start = max(
        available_at_s,
        release_s + network,
        release_s + startup_time_s,
    )
    cumulative = cumulative_processing_s(sfc, instance.vm_type.speed_mips)
    completions = tuple(start + value for value in cumulative)
    finish = completions[-1]
    output = output_delay_s(topology, sfc, instance.node_id)
    terminal_delivery = finish + output
    internal_slacks = [
        deadline - completion
        for deadline, completion in zip(
            sfc.vnf_deadlines_s,
            completions,
        )
    ]
    if sfc.is_terminal:
        internal_slacks.append(
            sfc.workflow_deadline_s - terminal_delivery
        )
    laxity = min(internal_slacks, default=math.inf)
    earliest_input_arrival = release_s + network
    return CandidateTiming(
        start_s=start,
        finish_s=finish,
        vnf_completion_times_s=completions,
        terminal_delivery_s=terminal_delivery,
        input_delay_s=network,
        input_propagation_s=propagation,
        input_serialization_s=serialization,
        output_delay_s=output,
        queueing_delay_s=max(0.0, start - earliest_input_arrival),
        laxity_s=laxity,
        deadline_feasible=laxity >= -1e-9,
    )


def transfer_routes(
    *,
    sfc: SFC,
    transfers: tuple[tuple[str, float], ...],
    candidate_node: str,
    edge_device_id: str,
) -> tuple[tuple[str, str, float], ...]:
    routes = [
        (source, candidate_node, data_mb)
        for source, data_mb in transfers
        if data_mb > 0 and source != candidate_node
    ]
    if (
        sfc.is_terminal
        and sfc.sink_data_mb > 0
        and candidate_node != edge_device_id
    ):
        routes.append(
            (candidate_node, edge_device_id, sfc.sink_data_mb)
        )
    return tuple(routes)
