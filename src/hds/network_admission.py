"""Shared fixed-path link-capacity admission for all schedulers.

The simulator does not model packets or sub-second transfer intervals.  It
therefore applies one conservative, deterministic admission rule to every
scheduling path: the aggregate external bandwidth admitted in one scheduler
batch may not exceed any directed physical link's residual capacity.
"""

from __future__ import annotations

from dataclasses import dataclass

from .topology import FogTopology


@dataclass(slots=True)
class LinkCapacityLedger:
    """Track residual directed-link capacity within one scheduler batch."""

    topology: FogTopology
    residual_mbps: dict[tuple[str, str], float]

    @classmethod
    def full_capacity(cls, topology: FogTopology) -> "LinkCapacityLedger":
        return cls(
            topology=topology,
            residual_mbps={
                link_id: link.bandwidth_mbps
                for link_id, link in topology.links.items()
            },
        )

    def path_links(
        self,
        source: str,
        destination: str,
    ) -> tuple[tuple[str, str], ...]:
        return self.topology.shortest_path(source, destination).links

    def feasible(
        self,
        source: str,
        destination: str,
        demand_mbps: float,
    ) -> bool:
        if demand_mbps < 0:
            raise ValueError("Link demand cannot be negative")
        return all(
            demand_mbps <= self.residual_mbps[link_id] + 1e-9
            for link_id in self.path_links(source, destination)
        )

    def reserve(
        self,
        source: str,
        destination: str,
        demand_mbps: float,
    ) -> None:
        if not self.feasible(source, destination, demand_mbps):
            raise ValueError(
                "Physical-link capacity exceeded for "
                f"{source}->{destination} ({demand_mbps} Mbit/s)"
            )
        for link_id in self.path_links(source, destination):
            self.residual_mbps[link_id] -= demand_mbps

    def aggregate_demands(
        self,
        transfers: tuple[tuple[str, float], ...],
        destination: str,
        transfer_window_s: float = 1.0,
    ) -> dict[tuple[str, str], float]:
        """Aggregate independent transfer demands on shared path links."""

        demands: dict[tuple[str, str], float] = {}
        for source, data_mb in transfers:
            demand_mbps = data_mb * 8.0 / transfer_window_s
            for link_id in self.path_links(source, destination):
                demands[link_id] = demands.get(link_id, 0.0) + demand_mbps
        return demands

    def aggregate_routes(
        self,
        routes: tuple[tuple[str, str, float], ...],
        transfer_window_s: float = 1.0,
    ) -> dict[tuple[str, str], float]:
        """Aggregate source, parent, and terminal-output route demands."""

        demands: dict[tuple[str, str], float] = {}
        for source, destination, data_mb in routes:
            if data_mb <= 0 or source == destination:
                continue
            demand_mbps = data_mb * 8.0 / transfer_window_s
            for link_id in self.path_links(source, destination):
                demands[link_id] = demands.get(link_id, 0.0) + demand_mbps
        return demands

    def feasible_many(
        self,
        transfers: tuple[tuple[str, float], ...],
        destination: str,
    ) -> bool:
        demands = self.aggregate_demands(transfers, destination)
        return all(
            demand <= self.residual_mbps[link_id] + 1e-9
            for link_id, demand in demands.items()
        )

    def reserve_many(
        self,
        transfers: tuple[tuple[str, float], ...],
        destination: str,
    ) -> None:
        demands = self.aggregate_demands(transfers, destination)
        if not all(
            demand <= self.residual_mbps[link_id] + 1e-9
            for link_id, demand in demands.items()
        ):
            raise ValueError("Physical-link capacity exceeded by transfer set")
        for link_id, demand in demands.items():
            self.residual_mbps[link_id] -= demand

    def feasible_routes(
        self,
        routes: tuple[tuple[str, str, float], ...],
    ) -> bool:
        demands = self.aggregate_routes(routes)
        return all(
            demand <= self.residual_mbps[link_id] + 1e-9
            for link_id, demand in demands.items()
        )

    def reserve_routes(
        self,
        routes: tuple[tuple[str, str, float], ...],
    ) -> None:
        demands = self.aggregate_routes(routes)
        if not all(
            demand <= self.residual_mbps[link_id] + 1e-9
            for link_id, demand in demands.items()
        ):
            raise ValueError("Physical-link capacity exceeded by route set")
        for link_id, demand in demands.items():
            self.residual_mbps[link_id] -= demand

    def snapshot(self) -> dict[tuple[str, str], float]:
        """Return a copy suitable for the BoS MILP."""

        return dict(self.residual_mbps)
