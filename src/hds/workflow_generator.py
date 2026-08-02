"""Deterministic synthetic scientific-workflow generators.

The generators preserve the broad dependency motifs of the four workflow
families used in the paper. They do not claim to be authoritative Pegasus
traces; users can load real DAX traces with :mod:`hds.workflow_loader`.
"""

from __future__ import annotations

import random

from .models import VNF, Workflow, WorkflowEdge


def _vnf(index: int, rng: random.Random, fastest_mips: float = 16_000) -> VNF:
    runtime_s = rng.uniform(0.1, 5.0)
    workload = runtime_s * fastest_mips
    cpu = rng.uniform(500, min(8000, fastest_mips))
    ram = rng.uniform(64, 480)
    storage = rng.uniform(32, 256)
    return VNF(
        id=f"t{index:04d}",
        workload_mi=workload,
        cpu_mips=cpu,
        ram_mb=ram,
        storage_mb=storage,
        source_data_mb=rng.uniform(0.1, 10.0) if index == 0 else 0.0,
    )


def _add_edge(
    edges: list[WorkflowEdge],
    seen: set[tuple[str, str]],
    parent: str,
    child: str,
    rng: random.Random,
) -> None:
    if parent == child or (parent, child) in seen:
        return
    edges.append(WorkflowEdge(parent, child, rng.uniform(0.1, 10.0)))
    seen.add((parent, child))


def generate_workflow(
    family: str,
    size: int,
    deadline_factor: float = 3.0,
    seed: int = 1,
) -> Workflow:
    """Generate a reproducible DAG with 25-600 VNFs.

    ``deadline_factor`` multiplies a conservative serial execution estimate.
    TASTD later performs the paper's formal feasibility check.
    """

    if size < 2:
        raise ValueError("Synthetic workflows need at least two VNFs")
    family_key = family.strip().lower()
    supported = {"montage", "epigenomics", "inspiral", "cybershake"}
    if family_key not in supported:
        raise ValueError(f"Unsupported workflow family: {family}")
    if deadline_factor <= 0:
        raise ValueError("deadline_factor must be positive")

    rng = random.Random(seed)
    vnfs = {f"t{i:04d}": _vnf(i, rng) for i in range(size)}
    edges: list[WorkflowEdge] = []
    seen: set[tuple[str, str]] = set()

    # Build connected stage DAGs. Alternating width-one and wider stages
    # create the pipeline, fork, and join structures needed by the schedulers.
    # All edges point to a larger index, so acyclicity holds by construction.
    width_cycles = {
        "montage": (4, 1, 4, 1),
        "epigenomics": (1, 1, 2, 1, 1),
        "inspiral": (2, 1, 2, 1),
        "cybershake": (4, 1, 3, 1),
    }
    previous_layer = [0]
    cursor = 1
    cycle = width_cycles[family_key]
    layer_number = 0
    while cursor < size:
        width = min(cycle[layer_number % len(cycle)], size - cursor)
        current_layer = list(range(cursor, cursor + width))
        for parent in previous_layer:
            for child in current_layer:
                _add_edge(
                    edges,
                    seen,
                    f"t{parent:04d}",
                    f"t{child:04d}",
                    rng,
                )
        previous_layer = current_layer
        cursor += width
        layer_number += 1

    # Mark all actual sinks with output data.
    children = {edge.parent for edge in edges}
    for task_id in set(vnfs) - children:
        original = vnfs[task_id]
        vnfs[task_id] = VNF(
            id=original.id,
            workload_mi=original.workload_mi,
            cpu_mips=original.cpu_mips,
            ram_mb=original.ram_mb,
            storage_mb=original.storage_mb,
            source_data_mb=original.source_data_mb,
            sink_data_mb=rng.uniform(0.1, 10.0),
        )

    serial_fastest_s = sum(task.workload_mi / 16_000 for task in vnfs.values())
    deadline = max(1.0, deadline_factor * serial_fastest_s)
    return Workflow(
        id=f"{family_key}-{size}-seed-{seed}",
        family=family_key,
        vnfs=vnfs,
        edges=edges,
        deadline_s=deadline,
    )
