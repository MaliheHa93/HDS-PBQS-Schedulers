"""Cross-module invariant checks."""

from __future__ import annotations

from .models import Assignment, SFCGraph, Workflow


def validate_sfc_partition(workflow: Workflow, graph: SFCGraph) -> None:
    """Validate completeness, uniqueness, order, and dependency preservation."""

    flattened = [
        vnf_id
        for sfc in graph.sfcs.values()
        for vnf_id in sfc.vnf_ids
    ]
    if len(flattened) != len(set(flattened)):
        raise AssertionError("An input VNF occurs in more than one SFC")
    if set(flattened) != set(workflow.vnfs):
        raise AssertionError("SFC partition does not cover all workflow VNFs")

    position: dict[str, int] = {}
    for sfc in graph.sfcs.values():
        for index, vnf_id in enumerate(sfc.vnf_ids):
            position[vnf_id] = index
        for parent, child in zip(sfc.vnf_ids, sfc.vnf_ids[1:]):
            try:
                workflow.edge(parent, child)
            except KeyError as error:
                raise AssertionError(
                    "An ordered SFC contains non-adjacent workflow VNFs"
                ) from error
    expected_inter_sfc_data: dict[tuple[str, str], float] = {}
    for edge in workflow.edges:
        parent_sfc = graph.vnf_to_sfc[edge.parent]
        child_sfc = graph.vnf_to_sfc[edge.child]
        if parent_sfc == child_sfc:
            if position[edge.parent] >= position[edge.child]:
                raise AssertionError("An intra-SFC dependency order was reversed")
        elif (parent_sfc, child_sfc) not in graph.edges_mb:
            raise AssertionError("An inter-SFC dependency was lost")
        else:
            key = (parent_sfc, child_sfc)
            expected_inter_sfc_data[key] = (
                expected_inter_sfc_data.get(key, 0.0) + edge.data_mb
            )
    if set(expected_inter_sfc_data) != set(graph.edges_mb):
        raise AssertionError("The SFC graph contains a spurious dependency")
    for key, expected in expected_inter_sfc_data.items():
        if abs(graph.edges_mb[key] - expected) > 1e-9:
            raise AssertionError("Inter-SFC transfer data was not preserved")
    graph.topological_order()


def validate_assignments(
    graph: SFCGraph,
    assignments: list[Assignment],
) -> None:
    """Validate uniqueness, deadlines, and precedence timing."""

    by_sfc = {assignment.sfc_id: assignment for assignment in assignments}
    if len(by_sfc) != len(assignments):
        raise AssertionError("An SFC was assigned more than once")
    for assignment in assignments:
        if assignment.finish_s > graph.sfcs[assignment.sfc_id].deadline_s + 1e-7:
            raise AssertionError(f"Deadline miss by {assignment.sfc_id}")
        for predecessor in graph.predecessors(assignment.sfc_id):
            if predecessor in by_sfc:
                predecessor_finish = by_sfc[predecessor].finish_s
                if assignment.start_s + 1e-7 < predecessor_finish:
                    raise AssertionError("SFC began before its predecessor completed")
