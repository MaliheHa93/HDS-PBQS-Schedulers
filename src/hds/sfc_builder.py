"""Workflow-DAG to dependency-preserving SFC transformation."""

from __future__ import annotations

from dataclasses import replace

from .models import SFC, SFCGraph, Workflow


class SFCBuilder:
    """Create a complete, non-overlapping SFC partition.

    Adjacent VNFs share one chain only when their edge is a linear segment:
    the parent has one successor and the child has one predecessor. Therefore,
    a fork terminates its incoming chain and every join begins a new chain
    that retains all incoming SFC dependencies (manuscript Equation 6).
    """

    def __init__(
        self,
        fastest_mips: float,
        reference_bandwidth_mbps: float = 1000.0,
        reference_link_latency_s: float = 0.0005,
        transfer_window_s: float = 1.0,
    ) -> None:
        if min(
            fastest_mips,
            reference_bandwidth_mbps,
            transfer_window_s,
        ) <= 0:
            raise ValueError("Reference capacities must be positive")
        self.fastest_mips = fastest_mips
        self.reference_bandwidth_mbps = reference_bandwidth_mbps
        self.reference_link_latency_s = reference_link_latency_s
        self.transfer_window_s = transfer_window_s

    def build(self, workflow: Workflow) -> SFCGraph:
        chains: list[list[str]] = []
        membership: dict[str, int] = {}

        for task_id in workflow.topological_order():
            predecessors = workflow.predecessors(task_id)
            selected_parent: str | None = None
            if len(predecessors) == 1:
                parent = predecessors[0]
                if (
                    len(workflow.successors(parent)) == 1
                    and len(predecessors) == 1
                ):
                    selected_parent = parent

            chain_index = membership.get(selected_parent) if selected_parent else None
            can_append = (
                chain_index is not None
                and chains[chain_index][-1] == selected_parent
            )
            if can_append:
                chains[chain_index].append(task_id)
                membership[task_id] = chain_index
            else:
                membership[task_id] = len(chains)
                chains.append([task_id])

        sfcs: dict[str, SFC] = {}
        vnf_to_sfc: dict[str, str] = {}
        for index, chain in enumerate(chains):
            sfc_id = f"{workflow.id}:sfc-{index:04d}"
            for task_id in chain:
                if task_id in vnf_to_sfc:
                    raise AssertionError(f"Duplicate VNF membership: {task_id}")
                vnf_to_sfc[task_id] = sfc_id
            tasks = [workflow.vnfs[task_id] for task_id in chain]
            sfcs[sfc_id] = SFC(
                id=sfc_id,
                workflow_id=workflow.id,
                vnf_ids=tuple(chain),
                workload_mi=sum(task.workload_mi for task in tasks),
                cpu_mips=max(task.cpu_mips for task in tasks),
                ram_mb=max(task.ram_mb for task in tasks),
                incoming_bandwidth_mbps=0.0,
                outgoing_bandwidth_mbps=0.0,
                bandwidth_mbps=0.0,
                storage_mb=max(task.storage_mb for task in tasks),
                source_data_mb=sum(task.source_data_mb for task in tasks),
                sink_data_mb=sum(task.sink_data_mb for task in tasks),
                vnf_workloads_mi=tuple(
                    task.workload_mi for task in tasks
                ),
            )

        if set(vnf_to_sfc) != set(workflow.vnfs):
            missing = set(workflow.vnfs) - set(vnf_to_sfc)
            raise AssertionError(f"Incomplete SFC partition: {missing}")

        edge_data: dict[tuple[str, str], float] = {}
        for edge in workflow.edges:
            source_sfc = vnf_to_sfc[edge.parent]
            destination_sfc = vnf_to_sfc[edge.child]
            if source_sfc != destination_sfc:
                key = (source_sfc, destination_sfc)
                edge_data[key] = edge_data.get(key, 0.0) + edge.data_mb

        incoming_by_sfc = {sfc_id: 0.0 for sfc_id in sfcs}
        outgoing_by_sfc = {sfc_id: 0.0 for sfc_id in sfcs}
        nonterminal_sfc_ids = {parent for parent, _child in edge_data}
        for (parent, child), size in edge_data.items():
            outgoing_by_sfc[parent] += size
            incoming_by_sfc[child] += size
        for sfc_id, sfc in list(sfcs.items()):
            incoming_mb = sfc.source_data_mb + incoming_by_sfc[sfc_id]
            outgoing_mb = sfc.sink_data_mb + outgoing_by_sfc[sfc_id]
            incoming_mbps = incoming_mb * 8.0 / self.transfer_window_s
            outgoing_mbps = outgoing_mb * 8.0 / self.transfer_window_s
            sfcs[sfc_id] = replace(
                sfc,
                incoming_bandwidth_mbps=incoming_mbps,
                outgoing_bandwidth_mbps=outgoing_mbps,
                bandwidth_mbps=max(incoming_mbps, outgoing_mbps),
                # Temporary VNF storage is released between VNFs, while
                # cross-SFC and terminal outputs persist until delivery.
                storage_mb=sfc.storage_mb + outgoing_mb,
                is_terminal=sfc_id not in nonterminal_sfc_ids,
                workflow_deadline_s=workflow.deadline_s,
            )

        graph = SFCGraph(workflow.id, sfcs, edge_data, vnf_to_sfc)
        graph.topological_order()
        return graph
