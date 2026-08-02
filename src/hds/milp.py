"""Lexicographic BoS MILP for manuscript Equations 10-21."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .models import SFC, VMInstance
from .timing import cumulative_processing_s, output_delay_s, transfer_routes
from .topology import FogTopology


@dataclass(frozen=True, slots=True)
class MILPConfig:
    alpha: float = 0.7
    beta: float = 0.2
    gamma: float = 0.1
    sharable: bool = True
    time_limit_s: float = 15.0
    relative_gap: float = 0.0
    maximum_billing_periods: int = 20
    reconstruct_earliest_start: bool = True

    def __post_init__(self) -> None:
        if min(self.alpha, self.beta, self.gamma) < 0:
            raise ValueError("MILP weights cannot be negative")
        if not math.isclose(
            self.alpha + self.beta + self.gamma,
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError("MILP weights must sum to one")
        if self.time_limit_s <= 0 or self.maximum_billing_periods <= 0:
            raise ValueError("Invalid MILP limit")
        if self.relative_gap < 0:
            raise ValueError("relative_gap cannot be negative")


@dataclass(frozen=True, slots=True)
class MILPAssignment:
    sfc_id: str
    candidate_id: str
    start_s: float
    completion_s: float
    vnf_completion_times_s: tuple[float, ...]
    terminal_delivery_s: float


@dataclass(frozen=True, slots=True)
class MILPResult:
    status: str
    objective: float | None
    assignments: tuple[MILPAssignment, ...]
    activated_periods: dict[str, int]
    runtime_s: float
    mip_gap: float | None
    node_count: int | None
    variable_count: int
    constraint_count: int
    big_m: float
    message: str
    timing_policy: str = "solver_times"
    avoidable_idle_removed_s: float = 0.0
    maximum_avoidable_idle_removed_s: float = 0.0
    bos_size: int = 0
    candidate_count: int = 0
    admitted_count: int = 0
    admission_status: str = "not_solved"
    secondary_status: str = "not_solved"

    @property
    def feasible(self) -> bool:
        return bool(self.assignments)

    @property
    def solver_limit_reached(self) -> bool:
        return (
            self.admission_status == "limit_reached"
            or self.secondary_status == "limit_reached"
        )


class _Variables:
    def __init__(
        self,
        sfcs: list[SFC],
        candidate_ids: list[str],
        *,
        include_ordering: bool,
    ) -> None:
        cursor = 0
        sfc_ids = [sfc.id for sfc in sfcs]
        self.x: dict[tuple[str, str], int] = {}
        for sfc_id in sfc_ids:
            for candidate_id in candidate_ids:
                self.x[(sfc_id, candidate_id)] = cursor
                cursor += 1
        self.y: dict[str, int] = {}
        for candidate_id in candidate_ids:
            self.y[candidate_id] = cursor
            cursor += 1
        self.nu: dict[str, int] = {}
        for sfc_id in sfc_ids:
            self.nu[sfc_id] = cursor
            cursor += 1
        self.t: dict[str, int] = {}
        for sfc_id in sfc_ids:
            self.t[sfc_id] = cursor
            cursor += 1
        self.completion: dict[tuple[str, int], int] = {}
        for sfc in sfcs:
            for index in range(len(sfc.vnf_ids)):
                self.completion[(sfc.id, index)] = cursor
                cursor += 1
        self.phi: dict[str, int] = {}
        for sfc_id in sfc_ids:
            self.phi[sfc_id] = cursor
            cursor += 1
        self.z: dict[tuple[str, str, str], int] = {}
        if include_ordering:
            for first_index, sfc_id in enumerate(sfc_ids):
                for other in sfc_ids[first_index + 1 :]:
                    for candidate_id in candidate_ids:
                        self.z[(sfc_id, other, candidate_id)] = cursor
                        cursor += 1
        self.g: dict[str, int] = {}
        for candidate_id in candidate_ids:
            self.g[candidate_id] = cursor
            cursor += 1
        self.h: dict[str, int] = {}
        for candidate_id in candidate_ids:
            self.h[candidate_id] = cursor
            cursor += 1
        self.count = cursor


class _ConstraintBuilder:
    def __init__(self, variable_count: int) -> None:
        self.variable_count = variable_count
        self.rows: list[dict[int, float]] = []
        self.lower: list[float] = []
        self.upper: list[float] = []

    def add(
        self,
        coefficients: dict[int, float],
        lower: float = -math.inf,
        upper: float = math.inf,
    ) -> None:
        self.rows.append(coefficients)
        self.lower.append(lower)
        self.upper.append(upper)

    def build(self) -> LinearConstraint:
        matrix = lil_matrix((len(self.rows), self.variable_count), dtype=float)
        for row_index, coefficients in enumerate(self.rows):
            for column, value in coefficients.items():
                if value:
                    matrix[row_index, column] = value
        return LinearConstraint(
            matrix.tocsr(),
            np.asarray(self.lower, dtype=float),
            np.asarray(self.upper, dtype=float),
        )


def _status_name(status: int) -> str:
    return {
        0: "optimal",
        1: "limit_reached",
        2: "infeasible",
        3: "unbounded",
        4: "solver_error",
    }.get(status, "unknown")


def _effective_workflow_deadline(sfc: SFC) -> float:
    if math.isfinite(sfc.workflow_deadline_s):
        return sfc.workflow_deadline_s
    return sfc.deadline_s


def _calculated_big_m(
    sfcs: Iterable[SFC],
    candidates: Iterable[VMInstance],
    release_s: float,
    maximum_billing_periods: int,
) -> float:
    sfc_list = list(sfcs)
    candidate_list = list(candidates)
    workflow_deadline = max(
        _effective_workflow_deadline(sfc) for sfc in sfc_list
    )
    slowest_execution = max(
        sfc.workload_mi / candidate.vm_type.speed_mips
        for sfc in sfc_list
        for candidate in candidate_list
    )
    maximum_startup = max(
        candidate.vm_type.startup_time_s for candidate in candidate_list
    )
    billing_horizon = max(
        candidate.vm_type.billing_period_s * maximum_billing_periods
        for candidate in candidate_list
    )
    return max(
        1.0,
        workflow_deadline - release_s + slowest_execution + maximum_startup,
        billing_horizon + slowest_execution + maximum_startup,
    )


def _extract_assignments(
    solution: np.ndarray,
    *,
    sfcs: list[SFC],
    candidates: list[VMInstance],
    variables: _Variables,
) -> tuple[list[MILPAssignment], dict[str, int]]:
    candidate_ids = [candidate.id for candidate in candidates]
    assignments: list[MILPAssignment] = []
    periods: dict[str, int] = {}
    for candidate_id in candidate_ids:
        count = int(round(solution[variables.h[candidate_id]]))
        if count > 0:
            periods[candidate_id] = count
    for sfc in sfcs:
        if solution[variables.nu[sfc.id]] < 0.5:
            continue
        selected = max(
            candidate_ids,
            key=lambda candidate_id: solution[
                variables.x[(sfc.id, candidate_id)]
            ],
        )
        if solution[variables.x[(sfc.id, selected)]] < 0.5:
            continue
        completions = tuple(
            max(
                0.0,
                float(solution[variables.completion[(sfc.id, index)]]),
            )
            for index in range(len(sfc.vnf_ids))
        )
        assignments.append(
            MILPAssignment(
                sfc_id=sfc.id,
                candidate_id=selected,
                start_s=max(0.0, float(solution[variables.t[sfc.id]])),
                completion_s=completions[-1],
                vnf_completion_times_s=completions,
                terminal_delivery_s=max(
                    0.0,
                    float(solution[variables.phi[sfc.id]]),
                ),
            )
        )
    return assignments, periods


def _reconstruct_earliest_start(
    assignments: list[MILPAssignment],
    *,
    release_s: float,
    sfc_by_id: dict[str, SFC],
    candidate_by_id: dict[str, VMInstance],
    input_delay: dict[tuple[str, str], float],
    output_delay: dict[tuple[str, str], float],
    cumulative_processing: dict[tuple[str, str], tuple[float, ...]],
) -> tuple[list[MILPAssignment], dict[str, int], float, float]:
    """Left-shift solved placement/order without changing either decision."""

    by_candidate: dict[str, list[MILPAssignment]] = {}
    for assignment in assignments:
        by_candidate.setdefault(assignment.candidate_id, []).append(assignment)

    rebuilt: list[MILPAssignment] = []
    periods: dict[str, int] = {}
    removed: list[float] = []
    for candidate_id, plans in sorted(by_candidate.items()):
        candidate = candidate_by_id[candidate_id]
        previous_completion = release_s
        candidate_plans: list[MILPAssignment] = []
        for plan in sorted(
            plans,
            key=lambda item: (item.start_s, item.completion_s, item.sfc_id),
        ):
            sfc = sfc_by_id[plan.sfc_id]
            key = (plan.sfc_id, candidate_id)
            start = max(
                release_s + input_delay[key],
                release_s + candidate.vm_type.startup_time_s,
                previous_completion,
            )
            completions = tuple(
                start + value for value in cumulative_processing[key]
            )
            terminal_delivery = completions[-1] + output_delay[key]
            if any(
                completion > deadline + 1e-7
                for completion, deadline in zip(
                    completions,
                    sfc.vnf_deadlines_s,
                )
            ):
                raise RuntimeError(
                    "Earliest-start reconstruction violated a VNF deadline"
                )
            if (
                sfc.is_terminal
                and terminal_delivery
                > _effective_workflow_deadline(sfc) + 1e-7
            ):
                raise RuntimeError(
                    "Earliest-start reconstruction violated terminal delivery"
                )
            removed.append(max(0.0, plan.start_s - start))
            candidate_plans.append(
                MILPAssignment(
                    sfc_id=plan.sfc_id,
                    candidate_id=candidate_id,
                    start_s=start,
                    completion_s=completions[-1],
                    vnf_completion_times_s=completions,
                    terminal_delivery_s=terminal_delivery,
                )
            )
            previous_completion = completions[-1]
        rebuilt.extend(candidate_plans)
        active_duration = max(
            plan.completion_s for plan in candidate_plans
        ) - release_s
        periods[candidate_id] = max(
            1,
            math.ceil(active_duration / candidate.vm_type.billing_period_s),
        )
    return rebuilt, periods, sum(removed), max(removed, default=0.0)


def solve_bos_milp(
    sfcs: list[SFC],
    candidates: list[VMInstance],
    topology: FogTopology,
    release_s: float,
    input_locations: dict[str, str],
    residual_node_capacities: dict[
        str,
        tuple[float, float] | tuple[float, float, float],
    ],
    config: MILPConfig,
    residual_link_bandwidth: dict[tuple[str, str], float] | None = None,
    input_data_mb: dict[str, float] | None = None,
    input_transfers: dict[str, tuple[tuple[str, float], ...]] | None = None,
) -> MILPResult:
    """Maximize admitted SFCs, then minimize Eq. 13 at that admission count."""

    started = time.perf_counter()
    if not sfcs or not candidates:
        return MILPResult(
            "invalid_input",
            None,
            (),
            {},
            0.0,
            None,
            None,
            0,
            0,
            0.0,
            "Both SFCs and candidates are required",
            bos_size=len(sfcs),
            candidate_count=len(candidates),
        )

    residual_link_bandwidth = residual_link_bandwidth or {
        key: link.bandwidth_mbps for key, link in topology.links.items()
    }
    input_data_mb = input_data_mb or {}
    input_transfers = input_transfers or {
        sfc.id: (
            (
                input_locations.get(sfc.id, topology.edge_device_id),
                input_data_mb.get(sfc.id, sfc.source_data_mb),
            ),
        )
        for sfc in sfcs
    }
    sfc_ids = [sfc.id for sfc in sfcs]
    candidate_ids = [candidate.id for candidate in candidates]
    sfc_by_id = {sfc.id: sfc for sfc in sfcs}
    candidate_by_id = {candidate.id: candidate for candidate in candidates}
    variables = _Variables(
        sfcs,
        candidate_ids,
        include_ordering=config.sharable,
    )
    constraints = _ConstraintBuilder(variables.count)
    big_m = _calculated_big_m(
        sfcs,
        candidates,
        release_s,
        config.maximum_billing_periods,
    )
    workflow_deadline = max(
        _effective_workflow_deadline(sfc) for sfc in sfcs
    )

    input_delay: dict[tuple[str, str], float] = {}
    output_delay: dict[tuple[str, str], float] = {}
    cumulative_processing: dict[
        tuple[str, str],
        tuple[float, ...],
    ] = {}
    link_demands: dict[
        tuple[str, str],
        dict[tuple[str, str], float],
    ] = {}
    feasible: dict[tuple[str, str], bool] = {}
    for sfc in sfcs:
        for candidate in candidates:
            key = (sfc.id, candidate.id)
            path_ok = True
            try:
                input_delay[key] = max(
                    (
                        topology.transfer_time_s(
                            source,
                            candidate.node_id,
                            data_mb,
                        )
                        if data_mb > 0 and source != candidate.node_id
                        else 0.0
                        for source, data_mb in input_transfers[sfc.id]
                    ),
                    default=0.0,
                )
                output_delay[key] = output_delay_s(
                    topology,
                    sfc,
                    candidate.node_id,
                )
                routes = transfer_routes(
                    sfc=sfc,
                    transfers=input_transfers[sfc.id],
                    candidate_node=candidate.node_id,
                    edge_device_id=topology.edge_device_id,
                )
                demands: dict[tuple[str, str], float] = {}
                for source, destination, data_mb in routes:
                    for link_id in topology.shortest_path(
                        source,
                        destination,
                    ).links:
                        demands[link_id] = (
                            demands.get(link_id, 0.0) + data_mb * 8.0
                        )
                link_demands[key] = demands
            except (KeyError, ValueError):
                path_ok = False
                input_delay[key] = math.inf
                output_delay[key] = math.inf
                link_demands[key] = {}
            cumulative_processing[key] = cumulative_processing_s(
                sfc,
                candidate.vm_type.speed_mips,
            )
            earliest_start = release_s + max(
                input_delay[key],
                candidate.vm_type.startup_time_s,
            )
            earliest_completions = tuple(
                earliest_start + value
                for value in cumulative_processing[key]
            )
            terminal_delivery = (
                earliest_completions[-1] + output_delay[key]
            )
            feasible[key] = (
                path_ok
                and sfc.cpu_mips
                <= candidate.vm_type.cpu_capacity_mips + 1e-9
                and sfc.ram_mb <= candidate.vm_type.ram_mb + 1e-9
                and sfc.storage_mb <= candidate.vm_type.storage_mb + 1e-9
                and sfc.bandwidth_mbps
                <= candidate.vm_type.access_bandwidth_mbps + 1e-9
                and all(
                    completion <= deadline + 1e-9
                    for completion, deadline in zip(
                        earliest_completions,
                        sfc.vnf_deadlines_s,
                    )
                )
                and (
                    not sfc.is_terminal
                    or terminal_delivery
                    <= _effective_workflow_deadline(sfc) + 1e-9
                )
            )

    lower_bounds = np.zeros(variables.count, dtype=float)
    upper_bounds = np.full(variables.count, np.inf, dtype=float)
    integrality = np.zeros(variables.count, dtype=int)
    for mapping in (variables.x, variables.y, variables.nu, variables.z):
        for index in mapping.values():
            upper_bounds[index] = 1.0
            integrality[index] = 1
    for index in variables.h.values():
        upper_bounds[index] = config.maximum_billing_periods
        integrality[index] = 1
    for index in variables.t.values():
        upper_bounds[index] = workflow_deadline
    for index in variables.completion.values():
        upper_bounds[index] = workflow_deadline
    for index in variables.phi.values():
        upper_bounds[index] = workflow_deadline
    for index in variables.g.values():
        upper_bounds[index] = big_m
    for key, is_feasible in feasible.items():
        if not is_feasible:
            upper_bounds[variables.x[key]] = 0.0

    # Equation 14: assignment, admission, feasibility, and activation.
    for sfc_id in sfc_ids:
        row = {
            variables.x[(sfc_id, candidate_id)]: 1.0
            for candidate_id in candidate_ids
        }
        row[variables.nu[sfc_id]] = -1.0
        constraints.add(row, 0.0, 0.0)
    for candidate_id in candidate_ids:
        for sfc_id in sfc_ids:
            constraints.add(
                {
                    variables.x[(sfc_id, candidate_id)]: 1.0,
                    variables.y[candidate_id]: -1.0,
                },
                upper=0.0,
            )
        row = {variables.y[candidate_id]: 1.0}
        row.update(
            {
                variables.x[(sfc_id, candidate_id)]: -1.0
                for sfc_id in sfc_ids
            }
        )
        constraints.add(row, upper=0.0)

    # Equation 15: candidate-VM reservations fit residual node capacity.
    for node_id, residual in residual_node_capacities.items():
        cpu_residual, ram_residual = residual[:2]
        storage_residual = (
            residual[2]
            if len(residual) >= 3
            else topology.nodes[node_id].storage_mb
        )
        hosted = [
            candidate
            for candidate in candidates
            if candidate.node_id == node_id
        ]
        constraints.add(
            {
                variables.y[candidate.id]:
                candidate.vm_type.cpu_capacity_mips
                for candidate in hosted
            },
            upper=cpu_residual,
        )
        constraints.add(
            {
                variables.y[candidate.id]: candidate.vm_type.ram_mb
                for candidate in hosted
            },
            upper=ram_residual,
        )
        constraints.add(
            {
                variables.y[candidate.id]: candidate.vm_type.storage_mb
                for candidate in hosted
            },
            upper=storage_residual,
        )

    # Equation 16: each assigned SFC individually fits its VM.
    for sfc in sfcs:
        for candidate in candidates:
            x = variables.x[(sfc.id, candidate.id)]
            constraints.add(
                {x: sfc.cpu_mips},
                upper=candidate.vm_type.cpu_capacity_mips,
            )
            constraints.add(
                {x: sfc.ram_mb},
                upper=candidate.vm_type.ram_mb,
            )
            constraints.add(
                {x: sfc.storage_mb},
                upper=candidate.vm_type.storage_mb,
            )
            constraints.add(
                {x: sfc.bandwidth_mbps},
                upper=candidate.vm_type.access_bandwidth_mbps,
            )

    # Equation 17: conservative simultaneous fixed-path reservations.
    for link_id, link in topology.links.items():
        row: dict[int, float] = {}
        for sfc in sfcs:
            for candidate in candidates:
                demand = link_demands[(sfc.id, candidate.id)].get(
                    link_id,
                    0.0,
                )
                if demand:
                    index = variables.x[(sfc.id, candidate.id)]
                    row[index] = row.get(index, 0.0) + demand
        constraints.add(
            row,
            upper=residual_link_bandwidth.get(
                link_id,
                link.bandwidth_mbps,
            ),
        )

    # Equations 18-19: admission-linked bounds, arrival/startup, all VNF
    # completions, internal deadlines, and terminal delivery.
    for sfc in sfcs:
        nu = variables.nu[sfc.id]
        constraints.add(
            {variables.t[sfc.id]: 1.0, nu: -workflow_deadline},
            upper=0.0,
        )
        start_input = {
            variables.t[sfc.id]: 1.0,
            nu: -release_s,
        }
        start_input.update(
            {
                variables.x[(sfc.id, candidate.id)]:
                -input_delay[(sfc.id, candidate.id)]
                for candidate in candidates
                if math.isfinite(input_delay[(sfc.id, candidate.id)])
            }
        )
        constraints.add(start_input, lower=0.0)
        start_vm = {
            variables.t[sfc.id]: 1.0,
            nu: -release_s,
        }
        start_vm.update(
            {
                variables.x[(sfc.id, candidate.id)]:
                -candidate.vm_type.startup_time_s
                for candidate in candidates
            }
        )
        constraints.add(start_vm, lower=0.0)

        for index, deadline in enumerate(sfc.vnf_deadlines_s):
            completion = variables.completion[(sfc.id, index)]
            constraints.add(
                {completion: 1.0, nu: -workflow_deadline},
                upper=0.0,
            )
            row = {
                completion: 1.0,
                variables.t[sfc.id]: -1.0,
            }
            row.update(
                {
                    variables.x[(sfc.id, candidate.id)]:
                    -cumulative_processing[(sfc.id, candidate.id)][index]
                    for candidate in candidates
                }
            )
            constraints.add(row, 0.0, 0.0)
            constraints.add(
                {completion: 1.0, nu: -deadline},
                upper=0.0,
            )

        final_completion = variables.completion[
            (sfc.id, len(sfc.vnf_ids) - 1)
        ]
        phi = variables.phi[sfc.id]
        constraints.add(
            {phi: 1.0, nu: -workflow_deadline},
            upper=0.0,
        )
        row = {phi: 1.0, final_completion: -1.0}
        row.update(
            {
                variables.x[(sfc.id, candidate.id)]:
                -output_delay[(sfc.id, candidate.id)]
                for candidate in candidates
                if math.isfinite(output_delay[(sfc.id, candidate.id)])
            }
        )
        constraints.add(row, 0.0, 0.0)
        if sfc.is_terminal:
            constraints.add(
                {
                    phi: 1.0,
                    nu: -_effective_workflow_deadline(sfc),
                },
                upper=0.0,
            )

    # Equation 20: SpaceShared ordering on a reused candidate.
    if config.sharable:
        for first_index, sfc_id in enumerate(sfc_ids):
            first = sfc_by_id[sfc_id]
            first_completion = variables.completion[
                (sfc_id, len(first.vnf_ids) - 1)
            ]
            for other_id in sfc_ids[first_index + 1 :]:
                other = sfc_by_id[other_id]
                other_completion = variables.completion[
                    (other_id, len(other.vnf_ids) - 1)
                ]
                for candidate_id in candidate_ids:
                    z = variables.z[(sfc_id, other_id, candidate_id)]
                    x_first = variables.x[(sfc_id, candidate_id)]
                    x_other = variables.x[(other_id, candidate_id)]
                    constraints.add(
                        {
                            first_completion: 1.0,
                            variables.t[other_id]: -1.0,
                            z: big_m,
                            x_first: big_m,
                            x_other: big_m,
                        },
                        upper=3.0 * big_m,
                    )
                    constraints.add(
                        {
                            other_completion: 1.0,
                            variables.t[sfc_id]: -1.0,
                            z: -big_m,
                            x_first: big_m,
                            x_other: big_m,
                        },
                        upper=2.0 * big_m,
                    )

    # Equation 21: active duration and purchased billing periods.
    for candidate in candidates:
        candidate_id = candidate.id
        for sfc in sfcs:
            final_completion = variables.completion[
                (sfc.id, len(sfc.vnf_ids) - 1)
            ]
            constraints.add(
                {
                    final_completion: 1.0,
                    variables.g[candidate_id]: -1.0,
                    variables.x[(sfc.id, candidate_id)]: big_m,
                },
                upper=release_s + big_m,
            )
        constraints.add(
            {
                variables.g[candidate_id]: 1.0,
                variables.y[candidate_id]: -big_m,
            },
            upper=0.0,
        )
        constraints.add(
            {
                variables.g[candidate_id]: 1.0,
                variables.h[candidate_id]:
                -candidate.vm_type.billing_period_s,
            },
            upper=0.0,
        )
        constraints.add(
            {
                variables.y[candidate_id]: 1.0,
                variables.h[candidate_id]: -1.0,
            },
            upper=0.0,
        )
        constraints.add(
            {
                variables.h[candidate_id]: 1.0,
                variables.y[candidate_id]:
                -config.maximum_billing_periods,
            },
            upper=0.0,
        )
        if not config.sharable:
            constraints.add(
                {
                    variables.x[(sfc_id, candidate_id)]: 1.0
                    for sfc_id in sfc_ids
                },
                upper=1.0,
            )

    scipy_options = {
        "time_limit": config.time_limit_s,
        "mip_rel_gap": config.relative_gap,
        "presolve": True,
    }

    # Equation 12: first maximize the number admitted.
    admission_objective = np.zeros(variables.count, dtype=float)
    for index in variables.nu.values():
        admission_objective[index] = -1.0
    admission = milp(
        c=admission_objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=constraints.build(),
        options=scipy_options,
    )
    admission_status = _status_name(admission.status)
    admission_solution = admission.x
    admitted_count = (
        int(
            round(
                sum(
                    admission_solution[index]
                    for index in variables.nu.values()
                )
            )
        )
        if admission_solution is not None
        else 0
    )
    if admitted_count <= 0:
        runtime = time.perf_counter() - started
        return MILPResult(
            status="infeasible",
            objective=None,
            assignments=(),
            activated_periods={},
            runtime_s=runtime,
            mip_gap=(
                float(admission.mip_gap)
                if getattr(admission, "mip_gap", None) is not None
                else None
            ),
            node_count=(
                int(admission.mip_node_count)
                if getattr(admission, "mip_node_count", None) is not None
                else None
            ),
            variable_count=variables.count,
            constraint_count=len(constraints.rows),
            big_m=big_m,
            message=str(admission.message),
            bos_size=len(sfcs),
            candidate_count=len(candidates),
            admitted_count=0,
            admission_status=admission_status,
        )

    # Equation 13: preserve A* and minimize normalized cost, communication,
    # and final-SFC completion.
    constraints.add(
        {index: 1.0 for index in variables.nu.values()},
        float(admitted_count),
        float(admitted_count),
    )
    secondary_objective = np.zeros(variables.count, dtype=float)
    max_cost = max(
        1e-12,
        sum(
            candidate.vm_type.cost_per_period
            * config.maximum_billing_periods
            for candidate in candidates
        ),
    )
    max_latency = max(
        1e-12,
        sum(
            max(
                input_delay[(sfc.id, candidate.id)]
                + output_delay[(sfc.id, candidate.id)]
                for candidate in candidates
                if math.isfinite(input_delay[(sfc.id, candidate.id)])
                and math.isfinite(output_delay[(sfc.id, candidate.id)])
            )
            for sfc in sfcs
        ),
    )
    max_completion = max(1e-12, len(sfcs) * workflow_deadline)
    for candidate in candidates:
        secondary_objective[variables.h[candidate.id]] = (
            config.alpha
            * candidate.vm_type.cost_per_period
            / max_cost
        )
    for sfc in sfcs:
        final_completion = variables.completion[
            (sfc.id, len(sfc.vnf_ids) - 1)
        ]
        secondary_objective[final_completion] = (
            config.gamma / max_completion
        )
        for candidate_index, candidate in enumerate(candidates):
            key = (sfc.id, candidate.id)
            if math.isfinite(input_delay[key]) and math.isfinite(
                output_delay[key]
            ):
                secondary_objective[
                    variables.x[(sfc.id, candidate.id)]
                ] = (
                    config.beta
                    * (input_delay[key] + output_delay[key])
                    / max_latency
                    + 1e-12 * (candidate_index + 1)
                )
    secondary = milp(
        c=secondary_objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=constraints.build(),
        options=scipy_options,
    )
    secondary_status = _status_name(secondary.status)
    solution = secondary.x if secondary.x is not None else admission_solution
    assignments, periods = _extract_assignments(
        solution,
        sfcs=sfcs,
        candidates=candidates,
        variables=variables,
    )
    timing_policy = "solver_times"
    idle_removed = 0.0
    maximum_idle_removed = 0.0
    if assignments and config.reconstruct_earliest_start:
        (
            assignments,
            periods,
            idle_removed,
            maximum_idle_removed,
        ) = _reconstruct_earliest_start(
            assignments,
            release_s=release_s,
            sfc_by_id=sfc_by_id,
            candidate_by_id=candidate_by_id,
            input_delay=input_delay,
            output_delay=output_delay,
            cumulative_processing=cumulative_processing,
        )
        timing_policy = "earliest_start_reconstruction"

    if (
        admission_status == "limit_reached"
        or secondary_status == "limit_reached"
    ):
        status = "limit_reached"
    elif secondary_status == "optimal":
        status = "optimal"
    else:
        status = secondary_status
    runtime = time.perf_counter() - started
    mip_gap_value = (
        float(secondary.mip_gap)
        if getattr(secondary, "mip_gap", None) is not None
        else (
            float(admission.mip_gap)
            if getattr(admission, "mip_gap", None) is not None
            else None
        )
    )
    node_count_value = (
        int(secondary.mip_node_count)
        if getattr(secondary, "mip_node_count", None) is not None
        else (
            int(admission.mip_node_count)
            if getattr(admission, "mip_node_count", None) is not None
            else None
        )
    )
    return MILPResult(
        status=status,
        objective=(
            float(secondary.fun) if secondary.fun is not None else None
        ),
        assignments=tuple(assignments),
        activated_periods=periods,
        runtime_s=runtime,
        mip_gap=mip_gap_value,
        node_count=node_count_value,
        variable_count=variables.count,
        constraint_count=len(constraints.rows),
        big_m=big_m,
        message=(
            f"admission={admission.message}; secondary={secondary.message}"
        ),
        timing_policy=timing_policy,
        avoidable_idle_removed_s=idle_removed,
        maximum_avoidable_idle_removed_s=maximum_idle_removed,
        bos_size=len(sfcs),
        candidate_count=len(candidates),
        admitted_count=len(assignments),
        admission_status=admission_status,
        secondary_status=secondary_status,
    )
