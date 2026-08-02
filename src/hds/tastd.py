"""Downstream-reserved VNF deadline assignment (manuscript Equations 7-9)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from .models import SFCGraph, Workflow


TransferLatency = Callable[[str, str, float], float]
ExternalLatency = Callable[[str, float], float]


@dataclass(frozen=True, slots=True)
class TASTDResult:
    """Traceable reference times and the resulting absolute VNF deadlines."""

    mode: str
    minimum_makespan_s: float
    workflow_slack_s: float
    reference_execution_s: dict[str, float]
    weights: dict[str, float]
    allocated_slack_s: dict[str, float]
    vnf_deadlines_s: dict[str, float]
    earliest_finish_s: dict[str, float]
    remaining_reference_s: dict[str, float]
    reference_feasible: bool = True
    deadline_kappa: float = 1.0


class InfeasibleWorkflowError(ValueError):
    """Raised when the workflow deadline is below its reference makespan."""


def _edge_reference_latency(
    workflow: Workflow,
    sfc_graph: SFCGraph | None,
    parent: str,
    child: str,
    transfer_latency: TransferLatency,
) -> float:
    """Return zero for intra-SFC edges and a path reference otherwise."""

    if (
        sfc_graph is not None
        and sfc_graph.vnf_to_sfc[parent] == sfc_graph.vnf_to_sfc[child]
    ):
        return 0.0
    return transfer_latency(
        parent,
        child,
        workflow.edge(parent, child).data_mb,
    )


def reference_timing(
    workflow: Workflow,
    fastest_mips: float,
    transfer_latency: TransferLatency,
    *,
    sfc_graph: SFCGraph | None = None,
    source_latency: ExternalLatency | None = None,
    terminal_latency: ExternalLatency | None = None,
) -> tuple[
    float,
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    """Compute Equations 7-8 and the terminal-delivery makespan."""

    if fastest_mips <= 0:
        raise ValueError("fastest_mips must be positive")
    source_latency = source_latency or (
        lambda task_id, data_mb: (
            transfer_latency("__source__", task_id, data_mb)
            if data_mb > 0
            else 0.0
        )
    )
    terminal_latency = terminal_latency or (
        lambda task_id, data_mb: (
            transfer_latency(task_id, "__destination__", data_mb)
            if data_mb > 0
            else 0.0
        )
    )
    reference_execution = {
        task_id: task.workload_mi / fastest_mips
        for task_id, task in workflow.vnfs.items()
    }

    earliest_finish: dict[str, float] = {}
    for task_id in workflow.topological_order():
        predecessors = workflow.predecessors(task_id)
        if predecessors:
            arrival = max(
                earliest_finish[parent]
                + _edge_reference_latency(
                    workflow,
                    sfc_graph,
                    parent,
                    task_id,
                    transfer_latency,
                )
                for parent in predecessors
            )
        else:
            task = workflow.vnfs[task_id]
            arrival = workflow.arrival_s + source_latency(
                task_id,
                task.source_data_mb,
            )
        earliest_finish[task_id] = (
            arrival + reference_execution[task_id]
        )

    minimum_makespan = max(
        earliest_finish[task_id]
        + terminal_latency(
            task_id,
            workflow.vnfs[task_id].sink_data_mb,
        )
        for task_id in workflow.sinks
    )

    remaining: dict[str, float] = {}
    for task_id in reversed(workflow.topological_order()):
        successors = workflow.successors(task_id)
        if successors:
            remaining[task_id] = max(
                _edge_reference_latency(
                    workflow,
                    sfc_graph,
                    task_id,
                    child,
                    transfer_latency,
                )
                + reference_execution[child]
                + remaining[child]
                for child in successors
            )
        else:
            remaining[task_id] = terminal_latency(
                task_id,
                workflow.vnfs[task_id].sink_data_mb,
            )
    return (
        minimum_makespan,
        reference_execution,
        earliest_finish,
        remaining,
    )


def minimum_reference_makespan(
    workflow: Workflow,
    fastest_mips: float,
    transfer_latency: TransferLatency,
    *,
    sfc_graph: SFCGraph | None = None,
    source_latency: ExternalLatency | None = None,
    terminal_latency: ExternalLatency | None = None,
) -> tuple[float, dict[str, float]]:
    """Backward-compatible wrapper returning makespan and earliest finishes."""

    minimum, _execution, earliest, _remaining = reference_timing(
        workflow,
        fastest_mips,
        transfer_latency,
        sfc_graph=sfc_graph,
        source_latency=source_latency,
        terminal_latency=terminal_latency,
    )
    return minimum, earliest


def assign_tastd_deadlines(
    workflow: Workflow,
    sfc_graph: SFCGraph,
    fastest_mips: float,
    eta: float = 0.7,
    transfer_latency: TransferLatency | None = None,
    mode: str = "downstream_reserved",
    allow_infeasible: bool = False,
    source_latency: ExternalLatency | None = None,
    terminal_latency: ExternalLatency | None = None,
) -> TASTDResult:
    """Assign absolute deadlines by reserving all downstream reference time.

    ``eta`` is retained only for source compatibility with v0.6.x. The current
    manuscript no longer distributes one global slack value by weights; it
    assigns ``deadline[f] = workflow_deadline - remaining_reference[f]``.
    ``topology`` is accepted as a legacy alias for ``downstream_reserved``.
    """

    if not 0 < eta <= 1:
        raise ValueError("eta must be in (0, 1]")
    if mode not in {"downstream_reserved", "topology"}:
        raise ValueError("mode must be downstream_reserved")
    if set(sfc_graph.vnf_to_sfc) != set(workflow.vnfs):
        raise ValueError("SFC graph and workflow do not have equal VNF coverage")

    transfer_latency = transfer_latency or (
        lambda _parent, _child, data_mb: (
            0.0005 + data_mb * 8.0 / 1000.0
            if data_mb > 0
            else 0.0
        )
    )
    minimum, reference, earliest, remaining = reference_timing(
        workflow,
        fastest_mips,
        transfer_latency,
        sfc_graph=sfc_graph,
        source_latency=source_latency,
        terminal_latency=terminal_latency,
    )
    reference_duration = minimum - workflow.arrival_s
    deadline_duration = workflow.deadline_s - workflow.arrival_s
    deadline_kappa = deadline_duration / reference_duration
    reference_feasible = workflow.deadline_s + 1e-9 >= minimum
    if not reference_feasible and not allow_infeasible:
        raise InfeasibleWorkflowError(
            f"Workflow deadline {workflow.deadline_s:.6f}s is below "
            f"minimum reference makespan {minimum:.6f}s"
        )

    deadlines = {
        task_id: workflow.deadline_s - remaining[task_id]
        for task_id in workflow.vnfs
    }
    local_slack = {
        task_id: deadlines[task_id] - earliest[task_id]
        for task_id in workflow.vnfs
    }
    for sfc_id, sfc in list(sfc_graph.sfcs.items()):
        sfc_graph.sfcs[sfc_id] = replace(
            sfc,
            deadline_s=deadlines[sfc.vnf_ids[-1]],
            vnf_deadlines_s=tuple(
                deadlines[vnf_id] for vnf_id in sfc.vnf_ids
            ),
            workflow_deadline_s=workflow.deadline_s,
        )

    return TASTDResult(
        mode="downstream_reserved",
        minimum_makespan_s=minimum,
        workflow_slack_s=workflow.deadline_s - minimum,
        reference_execution_s=reference,
        weights={},
        allocated_slack_s=local_slack,
        vnf_deadlines_s=deadlines,
        earliest_finish_s=earliest,
        remaining_reference_s=remaining,
        reference_feasible=reference_feasible,
        deadline_kappa=deadline_kappa,
    )
