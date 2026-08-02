"""Evaluation metrics used by Section VII of the paper.

The utilization metrics are time-integrated capacity ratios:

* CPU and RAM utilization divide executed VNF demand by purchased VM
  capacity-time.
* Per-node physical utilization divides the same executed demand by the
  fog-node capacity over the observed workflow makespan.
* Link bandwidth utilization divides traversing megabits by link
  capacity-times-observation-duration.

These definitions keep data volume, bandwidth, VM busy time, CPU, and RAM as
separate measurements instead of using one ambiguous "resource utilization"
value for all of them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics
from typing import Iterable

from scipy.stats import t as student_t

from .models import Assignment, SFCGraph, VMInstance, Workflow
from .topology import FogTopology


@dataclass(frozen=True, slots=True)
class NodeUtilization:
    """Time-integrated utilization details for one physical fog node."""

    node_id: str
    tier: str
    assigned_sfc_count: int
    provisioned_vm_count: int
    cpu_demand_mips_s: float
    ram_demand_mb_s: float
    purchased_cpu_capacity_mips_s: float
    purchased_ram_capacity_mb_s: float
    physical_cpu_capacity_mips_s: float
    physical_ram_capacity_mb_s: float
    cpu_utilization: float
    ram_utilization: float
    physical_cpu_utilization: float
    physical_ram_utilization: float

    def to_dict(self) -> dict[str, str | float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LinkUtilization:
    """Observed average utilization for one used directed physical link."""

    source: str
    destination: str
    capacity_mbps: float
    transferred_data_mb: float
    average_throughput_mbps: float
    bandwidth_utilization: float

    @property
    def link_id(self) -> str:
        return f"{self.source}->{self.destination}"

    def to_dict(self) -> dict[str, str | float]:
        result: dict[str, str | float] = asdict(self)
        result["link_id"] = self.link_id
        return result


@dataclass(frozen=True, slots=True)
class RunMetrics:
    provisioning_cost: float
    end_to_end_delay_s: float
    network_data_mb: float
    network_hop_data_mb: float
    resource_utilization: float
    cpu_utilization: float
    ram_utilization: float
    vm_time_utilization: float
    mean_node_cpu_utilization: float
    max_node_cpu_utilization: float
    mean_node_ram_utilization: float
    max_node_ram_utilization: float
    mean_link_bandwidth_utilization: float
    max_link_bandwidth_utilization: float
    used_link_count: int
    workflow_deadline_success: int
    workflow_completed: int
    global_deadline_success: int
    sfc_subdeadline_success: int
    sfc_deadline_miss_rate: float
    makespan_s: float
    vm_reuse_rate: float
    provisioned_vm_count: int
    scheduler_runtime_s: float
    scheduler_overhead_ratio: float
    milp_runtime_s: float
    communication_delay_s: float
    propagation_delay_s: float
    serialization_delay_s: float
    queueing_delay_s: float
    execution_time_s: float
    measurement_duration_s: float
    completed_sfc_count: int
    total_sfc_count: int
    accepted_sfc_ratio: float
    realized_workflow_slack_s: float
    workflow_deadline_consumption: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _measurement_end(
    workflow: Workflow,
    assignments: list[Assignment],
) -> float:
    if not assignments:
        return workflow.arrival_s
    return max(
        (
            assignment.terminal_delivery_s
            if assignment.terminal_delivery_s is not None
            else assignment.finish_s
        )
        for assignment in assignments
    )


def _meets_vnf_deadlines(assignment: Assignment, sfc) -> bool:
    if assignment.vnf_completion_times_s:
        return all(
            completion <= deadline + 1e-9
            for completion, deadline in zip(
                assignment.vnf_completion_times_s,
                sfc.vnf_deadlines_s,
            )
        )
    return assignment.finish_s <= sfc.deadline_s + 1e-9


def calculate_node_utilization(
    workflow: Workflow,
    graph: SFCGraph,
    assignments: list[Assignment],
    instances: Iterable[VMInstance],
    topology: FogTopology,
) -> tuple[NodeUtilization, ...]:
    """Calculate used/purchased and used/physical CPU and RAM ratios."""

    instance_list = list(instances)
    instance_by_id = {instance.id: instance for instance in instance_list}
    observation_s = max(
        0.0,
        _measurement_end(workflow, assignments) - workflow.arrival_s,
    )
    cpu_demand = {node_id: 0.0 for node_id in topology.nodes}
    ram_demand = {node_id: 0.0 for node_id in topology.nodes}
    assignment_count = {node_id: 0 for node_id in topology.nodes}

    for assignment in assignments:
        instance = instance_by_id[assignment.vm_id]
        assignment_count[assignment.node_id] += 1
        # VNFs inside one SFC run sequentially. Integrating each VNF's demand
        # is more accurate than applying the SFC's maximum CPU/RAM to the
        # entire chain duration.
        for vnf_id in graph.sfcs[assignment.sfc_id].vnf_ids:
            vnf = workflow.vnfs[vnf_id]
            duration_s = vnf.workload_mi / instance.vm_type.speed_mips
            cpu_demand[assignment.node_id] += vnf.cpu_mips * duration_s
            ram_demand[assignment.node_id] += vnf.ram_mb * duration_s

    rows: list[NodeUtilization] = []
    for node_id, node in sorted(topology.nodes.items()):
        node_instances = [
            instance
            for instance in instance_list
            if instance.node_id == node_id
        ]
        purchased_cpu = sum(
            instance.vm_type.cpu_capacity_mips
            * max(
                0.0,
                (instance.paid_until_s or 0.0)
                - (instance.provisioned_at_s or 0.0),
            )
            for instance in node_instances
        )
        purchased_ram = sum(
            instance.vm_type.ram_mb
            * max(
                0.0,
                (instance.paid_until_s or 0.0)
                - (instance.provisioned_at_s or 0.0),
            )
            for instance in node_instances
        )
        physical_cpu = node.cpu_mips * observation_s
        physical_ram = node.ram_mb * observation_s
        rows.append(
            NodeUtilization(
                node_id=node_id,
                tier=node.tier.value,
                assigned_sfc_count=assignment_count[node_id],
                provisioned_vm_count=len(node_instances),
                cpu_demand_mips_s=cpu_demand[node_id],
                ram_demand_mb_s=ram_demand[node_id],
                purchased_cpu_capacity_mips_s=purchased_cpu,
                purchased_ram_capacity_mb_s=purchased_ram,
                physical_cpu_capacity_mips_s=physical_cpu,
                physical_ram_capacity_mb_s=physical_ram,
                cpu_utilization=_safe_ratio(
                    cpu_demand[node_id], purchased_cpu
                ),
                ram_utilization=_safe_ratio(
                    ram_demand[node_id], purchased_ram
                ),
                physical_cpu_utilization=_safe_ratio(
                    cpu_demand[node_id], physical_cpu
                ),
                physical_ram_utilization=_safe_ratio(
                    ram_demand[node_id], physical_ram
                ),
            )
        )
    return tuple(rows)


def calculate_link_utilization(
    workflow: Workflow,
    graph: SFCGraph,
    assignments: list[Assignment],
    topology: FogTopology,
) -> tuple[LinkUtilization, ...]:
    """Calculate average capacity utilization for every used path link."""

    duration_s = max(
        0.0,
        _measurement_end(workflow, assignments) - workflow.arrival_s,
    )
    data_by_link: dict[tuple[str, str], float] = {}
    for source, destination, data_mb in _transfer_legs(
        graph, assignments, topology.edge_device_id
    ):
        if data_mb <= 0 or source == destination:
            continue
        path = topology.shortest_path(source, destination)
        for link_id in path.links:
            data_by_link[link_id] = (
                data_by_link.get(link_id, 0.0) + data_mb
            )

    rows: list[LinkUtilization] = []
    for link_id, data_mb in sorted(data_by_link.items()):
        link = topology.links[link_id]
        throughput = _safe_ratio(data_mb * 8.0, duration_s)
        rows.append(
            LinkUtilization(
                source=link.source,
                destination=link.destination,
                capacity_mbps=link.bandwidth_mbps,
                transferred_data_mb=data_mb,
                average_throughput_mbps=throughput,
                bandwidth_utilization=_safe_ratio(
                    throughput, link.bandwidth_mbps
                ),
            )
        )
    return tuple(rows)


def _transfer_legs(
    graph: SFCGraph,
    assignments: list[Assignment],
    edge_device_id: str,
) -> list[tuple[str, str, float]]:
    """Return every source, parent, and terminal-output transfer separately."""

    by_sfc = {assignment.sfc_id: assignment for assignment in assignments}
    legs: list[tuple[str, str, float]] = []
    for sfc_id, assignment in by_sfc.items():
        sfc = graph.sfcs[sfc_id]
        if not graph.predecessors(sfc_id) and sfc.source_data_mb > 0:
            legs.append(
                (edge_device_id, assignment.node_id, sfc.source_data_mb)
            )
        for parent_id in graph.predecessors(sfc_id):
            parent = by_sfc.get(parent_id)
            if parent is not None:
                legs.append(
                    (
                        parent.node_id,
                        assignment.node_id,
                        graph.edges_mb[(parent_id, sfc_id)],
                    )
                )
        if not graph.successors(sfc_id) and sfc.sink_data_mb > 0:
            legs.append(
                (assignment.node_id, edge_device_id, sfc.sink_data_mb)
            )
    return legs


def calculate_run_metrics(
    workflow: Workflow,
    graph: SFCGraph,
    assignments: list[Assignment],
    instances: Iterable[VMInstance],
    scheduler_runtime_s: float,
    milp_runtime_s: float,
    topology: FogTopology,
) -> tuple[
    RunMetrics,
    tuple[NodeUtilization, ...],
    tuple[LinkUtilization, ...],
]:
    """Calculate aggregate metrics and their node/link detail tables."""

    by_sfc = {assignment.sfc_id: assignment for assignment in assignments}
    instance_list = list(instances)
    completed = len(by_sfc)
    total = len(graph.sfcs)
    completed_workflow = completed == total
    if completed_workflow and assignments:
        terminal_completions = []
        for sfc_id, assignment in by_sfc.items():
            if graph.successors(sfc_id):
                continue
            terminal_completions.append(
                assignment.terminal_delivery_s
                if assignment.terminal_delivery_s is not None
                else (
                    assignment.finish_s
                    + topology.transfer_time_s(
                        assignment.node_id,
                        topology.edge_device_id,
                        graph.sfcs[sfc_id].sink_data_mb,
                    )
                )
            )
        makespan = max(terminal_completions) - workflow.arrival_s
        delay = makespan
    else:
        makespan = math.nan
        delay = math.nan
    deadline_misses = sum(
        1
        for sfc_id, sfc in graph.sfcs.items()
        if sfc_id not in by_sfc
        or not _meets_vnf_deadlines(by_sfc[sfc_id], sfc)
    )
    paid_time = sum(
        max(
            0.0,
            (vm.paid_until_s or 0.0) - (vm.provisioned_at_s or 0.0),
        )
        for vm in instance_list
    )
    busy_time = sum(vm.busy_time_s for vm in instance_list)
    vm_time_utilization = _safe_ratio(busy_time, paid_time)
    reuse_count = sum(assignment.reused_vm for assignment in assignments)
    subdeadline_success = int(
        completed_workflow
        and all(
            _meets_vnf_deadlines(by_sfc[sfc_id], sfc)
            for sfc_id, sfc in graph.sfcs.items()
        )
    )
    global_deadline_success = int(
        completed_workflow
        and makespan
        <= workflow.deadline_s - workflow.arrival_s + 1e-9
    )
    workflow_success = int(
        global_deadline_success and subdeadline_success
    )
    accepted_sfc_ratio = _safe_ratio(completed, total)
    completed_workflow = completed == total and math.isfinite(makespan)
    deadline_duration = workflow.deadline_s - workflow.arrival_s
    realized_slack = (
        deadline_duration - makespan
        if completed_workflow
        else math.nan
    )
    deadline_consumption = (
        _safe_ratio(makespan, deadline_duration)
        if completed_workflow
        else math.nan
    )

    node_rows = calculate_node_utilization(
        workflow,
        graph,
        assignments,
        instance_list,
        topology,
    )
    link_rows = calculate_link_utilization(
        workflow,
        graph,
        assignments,
        topology,
    )
    total_cpu_demand = sum(row.cpu_demand_mips_s for row in node_rows)
    total_ram_demand = sum(row.ram_demand_mb_s for row in node_rows)
    total_purchased_cpu = sum(
        row.purchased_cpu_capacity_mips_s for row in node_rows
    )
    total_purchased_ram = sum(
        row.purchased_ram_capacity_mb_s for row in node_rows
    )
    cpu_utilization = _safe_ratio(total_cpu_demand, total_purchased_cpu)
    ram_utilization = _safe_ratio(total_ram_demand, total_purchased_ram)
    resource_utilization = (cpu_utilization + ram_utilization) / 2.0

    used_nodes = [
        row
        for row in node_rows
        if row.assigned_sfc_count > 0 or row.provisioned_vm_count > 0
    ]
    node_cpu = [row.physical_cpu_utilization for row in used_nodes]
    node_ram = [row.physical_ram_utilization for row in used_nodes]
    link_utilization = [row.bandwidth_utilization for row in link_rows]

    transfer_legs = _transfer_legs(
        graph, assignments, topology.edge_device_id
    )
    network_data = sum(
        data_mb
        for source, destination, data_mb in transfer_legs
        if source != destination
    )
    network_hop_data = sum(
        row.transferred_data_mb for row in link_rows
    )
    propagation_delay = sum(
        topology.shortest_path(source, destination).propagation_latency_s
        for source, destination, data_mb in transfer_legs
        if data_mb > 0 and source != destination
    )
    communication_delay = sum(
        topology.transfer_time_s(source, destination, data_mb)
        for source, destination, data_mb in transfer_legs
        if data_mb > 0 and source != destination
    )
    serialization_delay = communication_delay - propagation_delay
    queueing_delay = sum(
        assignment.queueing_delay_s for assignment in assignments
    )
    execution_time = sum(
        assignment.execution_time_s for assignment in assignments
    )
    measurement_duration = (
        makespan if math.isfinite(makespan) else 0.0
    )

    metrics = RunMetrics(
        provisioning_cost=sum(
            vm.provisioning_cost for vm in instance_list
        ),
        end_to_end_delay_s=delay,
        network_data_mb=network_data,
        network_hop_data_mb=network_hop_data,
        resource_utilization=resource_utilization,
        cpu_utilization=cpu_utilization,
        ram_utilization=ram_utilization,
        vm_time_utilization=vm_time_utilization,
        mean_node_cpu_utilization=(
            statistics.fmean(node_cpu) if node_cpu else 0.0
        ),
        max_node_cpu_utilization=max(node_cpu, default=0.0),
        mean_node_ram_utilization=(
            statistics.fmean(node_ram) if node_ram else 0.0
        ),
        max_node_ram_utilization=max(node_ram, default=0.0),
        mean_link_bandwidth_utilization=(
            statistics.fmean(link_utilization)
            if link_utilization
            else 0.0
        ),
        max_link_bandwidth_utilization=max(
            link_utilization,
            default=0.0,
        ),
        used_link_count=len(link_rows),
        workflow_deadline_success=workflow_success,
        workflow_completed=int(completed_workflow),
        global_deadline_success=global_deadline_success,
        sfc_subdeadline_success=subdeadline_success,
        sfc_deadline_miss_rate=deadline_misses / total,
        makespan_s=makespan,
        vm_reuse_rate=_safe_ratio(reuse_count, completed),
        provisioned_vm_count=len(instance_list),
        scheduler_runtime_s=scheduler_runtime_s,
        scheduler_overhead_ratio=(
            _safe_ratio(scheduler_runtime_s, measurement_duration)
        ),
        milp_runtime_s=milp_runtime_s,
        communication_delay_s=communication_delay,
        propagation_delay_s=propagation_delay,
        serialization_delay_s=serialization_delay,
        queueing_delay_s=queueing_delay,
        execution_time_s=execution_time,
        measurement_duration_s=measurement_duration,
        completed_sfc_count=completed,
        total_sfc_count=total,
        accepted_sfc_ratio=accepted_sfc_ratio,
        realized_workflow_slack_s=realized_slack,
        workflow_deadline_consumption=deadline_consumption,
    )
    return metrics, node_rows, link_rows


def confidence_interval_95(values: list[float]) -> tuple[float, float]:
    """Return a two-sided 95% Student-t confidence interval for the mean."""

    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan, math.nan
    mean = statistics.fmean(finite)
    if len(finite) == 1:
        return mean, mean
    standard_error = statistics.stdev(finite) / math.sqrt(len(finite))
    critical_value = float(student_t.ppf(0.975, df=len(finite) - 1))
    margin = critical_value * standard_error
    return mean - margin, mean + margin
