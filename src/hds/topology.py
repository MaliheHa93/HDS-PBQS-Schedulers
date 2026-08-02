"""Physical topology, routing, and transfer-time calculations."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

from .models import FogNode, Link, Tier


@dataclass(frozen=True, slots=True)
class PathInfo:
    nodes: tuple[str, ...]
    links: tuple[tuple[str, str], ...]
    propagation_latency_s: float
    bottleneck_bandwidth_mbps: float

    def transfer_time_s(self, data_mb: float) -> float:
        if data_mb <= 0:
            return self.propagation_latency_s
        serialization_s = (data_mb * 8.0) / self.bottleneck_bandwidth_mbps
        return self.propagation_latency_s + serialization_s


class FogTopology:
    """Directed graph with deterministic shortest-latency routing."""

    def __init__(
        self,
        nodes: list[FogNode],
        links: list[Link],
        edge_device_id: str = "edge",
    ) -> None:
        self.nodes = {node.id: node for node in nodes}
        if len(self.nodes) != len(nodes):
            raise ValueError("Duplicate fog-node identifier")
        self.edge_device_id = edge_device_id
        endpoints = set(self.nodes) | {edge_device_id}
        self.links = {(link.source, link.destination): link for link in links}
        for link in links:
            if link.source not in endpoints or link.destination not in endpoints:
                raise ValueError(f"Unknown physical-link endpoint: {link}")
        self._adjacency: dict[str, list[str]] = {node: [] for node in endpoints}
        for source, destination in self.links:
            self._adjacency[source].append(destination)
        for values in self._adjacency.values():
            values.sort()

    def shortest_path(self, source: str, destination: str) -> PathInfo:
        if source == destination:
            return PathInfo((source,), (), 0.0, math.inf)
        if source not in self._adjacency or destination not in self._adjacency:
            raise KeyError(f"Unknown topology endpoint: {source} or {destination}")

        queue: list[tuple[float, tuple[str, ...], str]] = [(0.0, (source,), source)]
        best: dict[str, float] = {source: 0.0}
        while queue:
            latency, path, node = heapq.heappop(queue)
            if latency > best.get(node, math.inf) + 1e-12:
                continue
            if node == destination:
                link_ids = tuple(zip(path, path[1:]))
                bandwidth = min(self.links[key].bandwidth_mbps for key in link_ids)
                return PathInfo(path, link_ids, latency, bandwidth)
            for neighbor in self._adjacency[node]:
                link = self.links[(node, neighbor)]
                new_latency = latency + link.latency_s
                if new_latency <= best.get(neighbor, math.inf) + 1e-12:
                    best[neighbor] = new_latency
                    heapq.heappush(queue, (new_latency, path + (neighbor,), neighbor))
        raise ValueError(f"No path from {source} to {destination}")

    def transfer_time_s(self, source: str, destination: str, data_mb: float) -> float:
        return self.shortest_path(source, destination).transfer_time_s(data_mb)

    def min_propagation_latency_s(self) -> float:
        latencies = [link.latency_s for link in self.links.values()]
        return min(latencies, default=0.0)

    def nodes_in_tier(self, tier: Tier) -> list[FogNode]:
        return sorted(
            (node for node in self.nodes.values() if node.tier == tier),
            key=lambda node: node.id,
        )

    def path_indicator(
        self, source: str, destination: str
    ) -> dict[tuple[str, str], int]:
        path = self.shortest_path(source, destination)
        return {link_id: int(link_id in path.links) for link_id in self.links}


def paper_base_topology() -> FogTopology:
    """Return the five-node, two-tier topology described in Table II."""

    return paper_scaled_topology(3, 2)


def paper_scaled_topology(
    local_count: int,
    global_count: int,
) -> FogTopology:
    """Build the small/medium/large topology families from Table III."""

    if local_count <= 0 or global_count <= 0:
        raise ValueError("Both topology tiers must contain at least one node")
    nodes = [
        FogNode(f"local-{i}", Tier.LOCAL, 8000, 512, 1_000_000, 5)
        for i in range(1, local_count + 1)
    ]
    nodes += [
        FogNode(f"global-{i}", Tier.GLOBAL, 16000, 2048, 5_000_000, 10)
        for i in range(1, global_count + 1)
    ]
    links: list[Link] = []
    for node in nodes:
        if node.tier == Tier.LOCAL:
            links.extend(
                [
                    Link("edge", node.id, 0.0005, 100),
                    Link(node.id, "edge", 0.0005, 100),
                ]
            )
    for local in (n for n in nodes if n.tier == Tier.LOCAL):
        for index, global_node in enumerate(
            n for n in nodes if n.tier == Tier.GLOBAL
        ):
            # Table II specifies a 2-3 ms local-global latency range. Alternate
            # deterministically within that range so scaled topologies do not
            # silently increase the latency when more global nodes are added.
            latency = 0.002 + 0.001 * (index % 2)
            links.extend(
                [
                    Link(local.id, global_node.id, latency, 1000),
                    Link(global_node.id, local.id, latency, 1000),
                ]
            )
    return FogTopology(nodes, links)
