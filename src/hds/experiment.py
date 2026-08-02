"""Programmatic experiment API shared by scripts and tests."""

from __future__ import annotations

from dataclasses import asdict, replace
import json

from .cost_scheduler import CostFirstScheduler
from .edf_scheduler import EDFScheduler
from .hds_scheduler import HDSConfig, HDSScheduler
from .models import Workflow
from .pbqs_scheduler import PBQSConfig, PBQSScheduler
from .sfc_builder import SFCBuilder
from .simulator import SimulationResult, WorkflowSimulator
from .tastd import assign_tastd_deadlines, minimum_reference_makespan
from .topology import FogTopology, paper_base_topology
from .vm_pool import VMPool, paper_vm_types
from .workflow_generator import generate_workflow


def apply_deadline_policy(
    workflow: Workflow,
    *,
    deadline_factor: float,
    topology: FogTopology,
    deadline_mode: str,
) -> Workflow:
    """Return a workflow with a documented serial or reference deadline."""

    if deadline_factor <= 0:
        raise ValueError("deadline_factor must be positive")
    if deadline_mode not in {"reference", "serial"}:
        raise ValueError("deadline_mode must be reference or serial")
    if deadline_mode == "serial":
        return workflow
    vm_types = paper_vm_types()
    fastest_mips = max(vm_type.speed_mips for vm_type in vm_types)
    graph = SFCBuilder(fastest_mips).build(workflow)
    fastest_bandwidth = max(
        link.bandwidth_mbps for link in topology.links.values()
    )
    minimum_latency = topology.min_propagation_latency_s()
    reference_transfer = lambda data: (
        minimum_latency + data * 8.0 / fastest_bandwidth
        if data > 0
        else 0.0
    )
    reference, _ = minimum_reference_makespan(
        workflow,
        fastest_mips,
        lambda _p, _c, data: reference_transfer(data),
        sfc_graph=graph,
        source_latency=lambda _task, data: reference_transfer(data),
        terminal_latency=lambda _task, data: reference_transfer(data),
    )
    reference_duration = reference - workflow.arrival_s
    return replace(
        workflow,
        deadline_s=workflow.arrival_s + deadline_factor * reference_duration,
    )


def run_case(
    family: str,
    size: int,
    deadline_factor: float,
    seed: int,
    scheduler_name: str,
    sharable: bool,
    topology: FogTopology | None = None,
    candidate_count: int = 20,
    solver_time_limit_s: float = 15.0,
    eta: float = 0.7,
    alpha: float = 0.7,
    beta: float = 0.2,
    hds_gamma: float = 0.1,
    omega_u: float = 0.7,
    omega_w: float = 0.3,
    tastd_mode: str = "downstream_reserved",
    enable_vm_reuse: bool = True,
    reuse_policy: str | None = None,
    joint_bos_optimization: bool = True,
    adaptive_bos_fallback: bool = True,
    deadline_mode: str = "reference",
    reconstruct_earliest_start: bool = True,
) -> SimulationResult:
    """Run one reproducible HDS/PBQS case."""

    topology = topology or paper_base_topology()
    workflow = generate_workflow(
        family,
        size,
        deadline_factor if deadline_mode == "serial" else 1.0,
        seed,
    )
    workflow = apply_deadline_policy(
        workflow,
        deadline_factor=deadline_factor,
        topology=topology,
        deadline_mode=deadline_mode,
    )
    return run_workflow(
        workflow=workflow,
        scheduler_name=scheduler_name,
        sharable=sharable,
        topology=topology,
        candidate_count=candidate_count,
        solver_time_limit_s=solver_time_limit_s,
        eta=eta,
        alpha=alpha,
        beta=beta,
        hds_gamma=hds_gamma,
        omega_u=omega_u,
        omega_w=omega_w,
        tastd_mode=tastd_mode,
        enable_vm_reuse=enable_vm_reuse,
        reuse_policy=reuse_policy,
        joint_bos_optimization=joint_bos_optimization,
        adaptive_bos_fallback=adaptive_bos_fallback,
        reconstruct_earliest_start=reconstruct_earliest_start,
    )


def run_workflow(
    workflow: Workflow,
    scheduler_name: str,
    sharable: bool,
    topology: FogTopology | None = None,
    candidate_count: int = 20,
    solver_time_limit_s: float = 15.0,
    eta: float = 0.7,
    alpha: float = 0.7,
    beta: float = 0.2,
    hds_gamma: float = 0.1,
    omega_u: float = 0.7,
    omega_w: float = 0.3,
    tastd_mode: str = "downstream_reserved",
    enable_vm_reuse: bool = True,
    reuse_policy: str | None = None,
    joint_bos_optimization: bool = True,
    adaptive_bos_fallback: bool = True,
    reconstruct_earliest_start: bool = True,
) -> SimulationResult:
    """Run one supplied workflow, including JSON or Pegasus DAX traces."""

    topology = topology or paper_base_topology()
    vm_types = paper_vm_types()
    fastest_mips = max(vm_type.speed_mips for vm_type in vm_types)
    graph = SFCBuilder(fastest_mips).build(workflow)
    fastest_bandwidth = max(
        link.bandwidth_mbps for link in topology.links.values()
    )
    minimum_latency = topology.min_propagation_latency_s()
    reference_transfer = lambda data: (
        minimum_latency + data * 8.0 / fastest_bandwidth
        if data > 0
        else 0.0
    )
    tastd = assign_tastd_deadlines(
        workflow,
        graph,
        fastest_mips,
        eta=eta,
        transfer_latency=lambda _p, _c, data: reference_transfer(data),
        mode=tastd_mode,
        allow_infeasible=True,
        source_latency=lambda _task, data: reference_transfer(data),
        terminal_latency=lambda _task, data: reference_transfer(data),
    )
    pool = VMPool(topology, vm_types)
    key = scheduler_name.strip().lower()
    if key == "hds":
        scheduler = HDSScheduler(
            topology,
            pool,
            HDSConfig(
                sharable=sharable,
                alpha=alpha,
                beta=beta,
                gamma=hds_gamma,
                candidate_count=candidate_count,
                solver_time_limit_s=solver_time_limit_s,
                enable_vm_reuse=enable_vm_reuse,
                reuse_policy=reuse_policy or "idle_only",
                joint_bos_optimization=joint_bos_optimization,
                adaptive_bos_fallback=adaptive_bos_fallback,
                reconstruct_earliest_start=reconstruct_earliest_start,
            ),
        )
    elif key == "pbqs":
        scheduler = PBQSScheduler(
            topology,
            pool,
            PBQSConfig(
                sharable=sharable,
                omega_u=omega_u,
                omega_w=omega_w,
                candidate_count=candidate_count,
                enable_vm_reuse=enable_vm_reuse,
                reuse_policy=reuse_policy or "queue_aware",
            ),
        )
    elif key == "edf":
        scheduler = EDFScheduler(
            topology,
            pool,
            config=PBQSConfig(
                sharable=sharable,
                omega_u=omega_u,
                omega_w=omega_w,
                candidate_count=candidate_count,
                enable_vm_reuse=enable_vm_reuse,
                reuse_policy=reuse_policy or "queue_aware",
            ),
        )
    elif key == "cost":
        scheduler = CostFirstScheduler(
            topology,
            pool,
            config=PBQSConfig(
                sharable=sharable,
                omega_u=omega_u,
                omega_w=omega_w,
                candidate_count=candidate_count,
                enable_vm_reuse=enable_vm_reuse,
                reuse_policy=reuse_policy or "queue_aware",
                placement_policy="cost_first",
            ),
        )
    else:
        raise ValueError("scheduler_name must be HDS, PBQS, EDF, or COST")
    return WorkflowSimulator(workflow, graph, tastd, scheduler, pool).run()


def result_record(
    result: SimulationResult,
    *,
    family: str,
    size: int,
    deadline_factor: float,
    seed: int,
    scheduler: str,
    sharable: bool,
    enable_vm_reuse: bool = True,
    reuse_policy: str | None = None,
    joint_bos_optimization: bool = True,
    adaptive_bos_fallback: bool = True,
    deadline_mode: str = "reference",
    reconstruct_earliest_start: bool = True,
) -> dict[str, str | int | float]:
    """Flatten a simulation result for raw CSV export."""

    actual_reuse_policy = reuse_policy or (
        "idle_only" if scheduler.strip().lower() == "hds" else "queue_aware"
    )
    scheduling_records = list(result.scheduling_records)
    bos_sizes = [
        int(item["ready_bos_size"])
        for item in scheduling_records
    ]
    bos_histogram: dict[int, int] = {}
    for size_value in bos_sizes:
        bos_histogram[size_value] = bos_histogram.get(size_value, 0) + 1
    total_sfc_count = result.metrics.total_sfc_count
    jointly_scheduled = sum(
        int(item["joint_scheduled_count"])
        for item in scheduling_records
    )
    record: dict[str, str | int | float] = {
        "family": family,
        "workflow_size": size,
        "deadline_factor": deadline_factor,
        "deadline_mode": deadline_mode,
        "seed": seed,
        "scheduler": scheduler.upper(),
        "sharing": "Sharable" if sharable else "NonSharable",
        "configuration": (
            f"{scheduler.upper()}-"
            f"{'Sharable' if sharable else 'NonSharable'}"
        ),
        "minimum_makespan_s": result.tastd.minimum_makespan_s,
        "workflow_slack_s": result.tastd.workflow_slack_s,
        "reference_deadline_feasible": int(
            result.tastd.reference_feasible
        ),
        "deadline_kappa": result.tastd.deadline_kappa,
        "tastd_mode": result.tastd.mode,
        "enable_vm_reuse": int(enable_vm_reuse),
        "reuse_policy": actual_reuse_policy,
        "joint_bos_optimization": int(joint_bos_optimization),
        "adaptive_bos_fallback": int(adaptive_bos_fallback),
        "reconstruct_earliest_start": int(reconstruct_earliest_start),
        "workflow_arrival_s": result.workflow_arrival_s,
        "workflow_deadline_s": result.workflow_deadline_s,
        "workflow_deadline_duration_s": (
            result.workflow_deadline_s - result.workflow_arrival_s
        ),
        "deadline_to_minimum_makespan_ratio": (
            result.tastd.deadline_kappa
        ),
        "requested_candidate_count": result.requested_candidate_count,
        "initial_effective_candidate_count": (
            result.initial_effective_candidate_count
        ),
        "first_round_submitted_sfc_count": (
            result.first_round_submitted_sfc_count
        ),
        "first_round_admitted_sfc_count": (
            result.first_round_admitted_sfc_count
        ),
        "first_round_accepted_sfc_ratio": (
            result.first_round_admitted_sfc_count
            / result.first_round_submitted_sfc_count
            if result.first_round_submitted_sfc_count
            else 0.0
        ),
        "first_round_scheduler_runtime_s": (
            result.first_round_scheduler_runtime_s
        ),
        "unfinished_sfc_count": len(result.unfinished_sfc_ids),
        "scheduling_round_count": len(scheduling_records),
        "bos_max_size": max(bos_sizes, default=0),
        "bos_mean_size": (
            sum(bos_sizes) / len(bos_sizes) if bos_sizes else 0.0
        ),
        "bos_size_histogram": json.dumps(
            bos_histogram,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "jointly_scheduled_sfc_count": jointly_scheduled,
        "jointly_scheduled_sfc_ratio": (
            jointly_scheduled / total_sfc_count
            if total_sfc_count
            else 0.0
        ),
        "single_sfc_fallback_count": sum(
            int(item["single_fallback_count"])
            for item in scheduling_records
        ),
        "reuse_path_count": sum(
            int(item["reuse_path_count"])
            for item in scheduling_records
        ),
        "new_vm_path_count": sum(
            int(item["new_vm_path_count"])
            for item in scheduling_records
        ),
        "milp_invocations": len(result.milp_records),
        "milp_max_bos_size": max(
            (int(item["bos_size"]) for item in result.milp_records),
            default=0,
        ),
        "milp_min_effective_candidate_count": min(
            (
                int(item["candidate_count"])
                for item in result.milp_records
            ),
            default=0,
        ),
        "milp_max_effective_candidate_count": max(
            (
                int(item["candidate_count"])
                for item in result.milp_records
            ),
            default=0,
        ),
        "milp_max_variables": max(
            (int(item["variables"]) for item in result.milp_records),
            default=0,
        ),
        "milp_max_constraints": max(
            (int(item["constraints"]) for item in result.milp_records),
            default=0,
        ),
        "milp_max_single_solve_s": max(
            (float(item["runtime_s"]) for item in result.milp_records),
            default=0.0,
        ),
        "milp_max_gap": max(
            (
                float(item["mip_gap"])
                for item in result.milp_records
                if item["mip_gap"] is not None
            ),
            default=0.0,
        ),
        "milp_limit_reached_count": sum(
            item["status"] == "limit_reached" for item in result.milp_records
        ),
        "milp_admission_limit_reached_count": sum(
            item["admission_status"] == "limit_reached"
            for item in result.milp_records
        ),
        "milp_secondary_limit_reached_count": sum(
            item["secondary_status"] == "limit_reached"
            for item in result.milp_records
        ),
        "milp_total_admitted_count": sum(
            int(item["admitted_count"]) for item in result.milp_records
        ),
        "milp_avoidable_idle_removed_s": sum(
            float(item["avoidable_idle_removed_s"])
            for item in result.milp_records
        ),
        "milp_max_avoidable_idle_removed_s": max(
            (
                float(item["maximum_avoidable_idle_removed_s"])
                for item in result.milp_records
            ),
            default=0.0,
        ),
    }
    record.update(result.metrics.to_dict())
    return record


def _detail_identity(
    *,
    family: str,
    size: int,
    deadline_factor: float,
    seed: int,
    scheduler: str,
    sharable: bool,
) -> dict[str, str | int | float]:
    return {
        "family": family,
        "workflow_size": size,
        "deadline_factor": deadline_factor,
        "seed": seed,
        "scheduler": scheduler.upper(),
        "sharing": "Sharable" if sharable else "NonSharable",
        "configuration": (
            f"{scheduler.upper()}-"
            f"{'Sharable' if sharable else 'NonSharable'}"
        ),
    }


def node_records(
    result: SimulationResult,
    *,
    family: str,
    size: int,
    deadline_factor: float,
    seed: int,
    scheduler: str,
    sharable: bool,
) -> list[dict[str, str | int | float]]:
    """Flatten per-node utilization for long-format CSV export."""

    identity = _detail_identity(
        family=family,
        size=size,
        deadline_factor=deadline_factor,
        seed=seed,
        scheduler=scheduler,
        sharable=sharable,
    )
    return [
        {**identity, **row.to_dict()}
        for row in result.node_utilization
    ]


def link_records(
    result: SimulationResult,
    *,
    family: str,
    size: int,
    deadline_factor: float,
    seed: int,
    scheduler: str,
    sharable: bool,
) -> list[dict[str, str | int | float]]:
    """Flatten used-link utilization for long-format CSV export."""

    identity = _detail_identity(
        family=family,
        size=size,
        deadline_factor=deadline_factor,
        seed=seed,
        scheduler=scheduler,
        sharable=sharable,
    )
    return [
        {**identity, **row.to_dict()}
        for row in result.link_utilization
    ]


def assignment_records(
    result: SimulationResult,
    *,
    family: str,
    size: int,
    deadline_factor: float,
    seed: int,
    scheduler: str,
    sharable: bool,
) -> list[dict[str, str | int | float]]:
    """Flatten the SFC execution trace for reproducibility audits."""

    identity = _detail_identity(
        family=family,
        size=size,
        deadline_factor=deadline_factor,
        seed=seed,
        scheduler=scheduler,
        sharable=sharable,
    )
    return [
        {**identity, **asdict(assignment)}
        for assignment in result.assignments
    ]
