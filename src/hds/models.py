"""Domain models and unit conventions for HDS.

Units used throughout the project:

* time: seconds
* processing speed and CPU capacity: MIPS
* workload: million instructions (MI)
* RAM and transferred data: MB
* link/access bandwidth: Mbit/s
* money: US dollars
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import heapq
import math
from typing import Iterable, Iterator


class Tier(str, Enum):
    """Fog-node tier."""

    LOCAL = "local"
    GLOBAL = "global"


@dataclass(frozen=True, slots=True)
class VNF:
    """One workflow task represented as a virtual network function."""

    id: str
    workload_mi: float
    cpu_mips: float
    ram_mb: float
    storage_mb: float = 64.0
    source_data_mb: float = 0.0
    sink_data_mb: float = 0.0

    def __post_init__(self) -> None:
        for name in ("workload_mi", "cpu_mips", "ram_mb", "storage_mb"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive for VNF {self.id}")
        if self.source_data_mb < 0 or self.sink_data_mb < 0:
            raise ValueError("External data sizes cannot be negative")


@dataclass(frozen=True, slots=True)
class WorkflowEdge:
    """A precedence edge and its data volume."""

    parent: str
    child: str
    data_mb: float

    def __post_init__(self) -> None:
        if self.parent == self.child:
            raise ValueError("Workflow self-loops are not allowed")
        if self.data_mb < 0:
            raise ValueError("Edge data size cannot be negative")


@dataclass(slots=True)
class Workflow:
    """Scientific workflow DAG."""

    id: str
    vnfs: dict[str, VNF]
    edges: list[WorkflowEdge]
    deadline_s: float
    arrival_s: float = 0.0
    family: str = "custom"
    _edge_index: dict[tuple[str, str], WorkflowEdge] = field(
        init=False, repr=False, default_factory=dict
    )
    _predecessor_index: dict[str, tuple[str, ...]] = field(
        init=False, repr=False, default_factory=dict
    )
    _successor_index: dict[str, tuple[str, ...]] = field(
        init=False, repr=False, default_factory=dict
    )
    _topological_order_cache: tuple[str, ...] = field(
        init=False, repr=False, default=()
    )

    def __post_init__(self) -> None:
        if not self.vnfs:
            raise ValueError("A workflow must contain at least one VNF")
        if self.deadline_s <= self.arrival_s:
            raise ValueError("Workflow deadline must be after its arrival")
        predecessors: dict[str, list[str]] = {vnf_id: [] for vnf_id in self.vnfs}
        successors: dict[str, list[str]] = {vnf_id: [] for vnf_id in self.vnfs}
        for edge in self.edges:
            if edge.parent not in self.vnfs or edge.child not in self.vnfs:
                raise ValueError(f"Unknown endpoint in edge {edge}")
            key = (edge.parent, edge.child)
            if key in self._edge_index:
                raise ValueError(f"Duplicate edge {key}")
            self._edge_index[key] = edge
            predecessors[edge.child].append(edge.parent)
            successors[edge.parent].append(edge.child)
        self._predecessor_index = {
            vnf_id: tuple(sorted(values))
            for vnf_id, values in predecessors.items()
        }
        self._successor_index = {
            vnf_id: tuple(sorted(values))
            for vnf_id, values in successors.items()
        }
        self._topological_order_cache = self._calculate_topological_order()

    def predecessors(self, vnf_id: str) -> list[str]:
        return list(self._predecessor_index[vnf_id])

    def successors(self, vnf_id: str) -> list[str]:
        return list(self._successor_index[vnf_id])

    def edge(self, parent: str, child: str) -> WorkflowEdge:
        return self._edge_index[(parent, child)]

    @property
    def sources(self) -> list[str]:
        return sorted(v for v in self.vnfs if not self._predecessor_index[v])

    @property
    def sinks(self) -> list[str]:
        return sorted(v for v in self.vnfs if not self._successor_index[v])

    def _calculate_topological_order(self) -> tuple[str, ...]:
        indegree = {
            vnf_id: len(self._predecessor_index[vnf_id])
            for vnf_id in self.vnfs
        }
        ready = [vnf_id for vnf_id, degree in indegree.items() if degree == 0]
        heapq.heapify(ready)
        order: list[str] = []
        while ready:
            current = heapq.heappop(ready)
            order.append(current)
            for child in self._successor_index[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(ready, child)
        if len(order) != len(self.vnfs):
            raise ValueError(f"Workflow {self.id} is not acyclic")
        return tuple(order)

    def topological_order(self) -> list[str]:
        """Return the cached deterministic Kahn topological order."""

        return list(self._topological_order_cache)


@dataclass(frozen=True, slots=True)
class FogNode:
    """Physical fog node."""

    id: str
    tier: Tier
    cpu_mips: float
    ram_mb: float
    storage_mb: float
    processing_elements: int

    def __post_init__(self) -> None:
        if min(
            self.cpu_mips, self.ram_mb, self.storage_mb, self.processing_elements
        ) <= 0:
            raise ValueError(f"Fog node {self.id} has an invalid capacity")


@dataclass(frozen=True, slots=True)
class Link:
    """Directed physical network link."""

    source: str
    destination: str
    latency_s: float
    bandwidth_mbps: float

    def __post_init__(self) -> None:
        if self.source == self.destination:
            raise ValueError("A physical link cannot be a self-loop")
        if self.latency_s < 0 or self.bandwidth_mbps <= 0:
            raise ValueError("Invalid link latency or bandwidth")


@dataclass(frozen=True, slots=True)
class VMType:
    """VM flavor available for provisioning."""

    id: str
    tier: Tier
    speed_mips: float
    cpu_capacity_mips: float
    ram_mb: float
    access_bandwidth_mbps: float
    cost_per_period: float
    billing_period_s: float = 60.0
    storage_mb: float = 1_000_000.0
    startup_time_s: float = 0.0

    def __post_init__(self) -> None:
        if min(
            self.speed_mips,
            self.cpu_capacity_mips,
            self.ram_mb,
            self.access_bandwidth_mbps,
            self.billing_period_s,
            self.storage_mb,
        ) <= 0:
            raise ValueError(f"VM type {self.id} has an invalid capacity")
        if self.cost_per_period < 0:
            raise ValueError("VM cost cannot be negative")
        if self.startup_time_s < 0:
            raise ValueError("VM startup time cannot be negative")


@dataclass(frozen=True, slots=True)
class Reservation:
    """One SFC execution interval on a VM."""

    sfc_id: str
    start_s: float
    finish_s: float

    def __post_init__(self) -> None:
        if self.start_s < 0 or self.finish_s < self.start_s:
            raise ValueError("Invalid VM reservation interval")


@dataclass(slots=True)
class VMInstance:
    """Provisioned or candidate VM instance."""

    id: str
    vm_type: VMType
    node_id: str
    provisioned_at_s: float | None = None
    paid_until_s: float | None = None
    reservations: list[Reservation] = field(default_factory=list)

    @property
    def is_provisioned(self) -> bool:
        return self.provisioned_at_s is not None

    @property
    def available_at_s(self) -> float:
        if not self.reservations:
            return self.provisioned_at_s or 0.0
        return max(r.finish_s for r in self.reservations)

    @property
    def busy_time_s(self) -> float:
        return sum(r.finish_s - r.start_s for r in self.reservations)

    @property
    def purchased_periods(self) -> int:
        if self.provisioned_at_s is None or self.paid_until_s is None:
            return 0
        duration = self.paid_until_s - self.provisioned_at_s
        return int(math.ceil(max(0.0, duration) / self.vm_type.billing_period_s))

    @property
    def provisioning_cost(self) -> float:
        return self.purchased_periods * self.vm_type.cost_per_period

    def is_idle_at(self, time_s: float) -> bool:
        return self.available_at_s <= time_s + 1e-9

    def has_paid_time_at(self, time_s: float) -> bool:
        return self.paid_until_s is not None and self.paid_until_s > time_s + 1e-9

    def reserve(self, sfc_id: str, start_s: float, finish_s: float) -> None:
        if start_s + 1e-9 < self.available_at_s:
            raise ValueError(f"SpaceShared overlap on VM {self.id}")
        self.reservations.append(Reservation(sfc_id, start_s, finish_s))


@dataclass(slots=True)
class SFC:
    """Ordered, dependency-preserving workflow segment."""

    id: str
    workflow_id: str
    vnf_ids: tuple[str, ...]
    workload_mi: float
    cpu_mips: float
    ram_mb: float
    incoming_bandwidth_mbps: float
    outgoing_bandwidth_mbps: float
    bandwidth_mbps: float
    deadline_s: float = math.inf
    storage_mb: float = 0.0
    source_data_mb: float = 0.0
    sink_data_mb: float = 0.0
    vnf_workloads_mi: tuple[float, ...] = ()
    vnf_deadlines_s: tuple[float, ...] = ()
    is_terminal: bool = False
    workflow_deadline_s: float = math.inf

    def __post_init__(self) -> None:
        if not self.vnf_ids:
            raise ValueError("An SFC cannot be empty")
        if min(self.workload_mi, self.cpu_mips, self.ram_mb) <= 0:
            raise ValueError(f"SFC {self.id} has invalid resource values")
        if self.storage_mb < 0:
            raise ValueError(f"SFC {self.id} has invalid storage")
        if not self.vnf_workloads_mi:
            share = self.workload_mi / len(self.vnf_ids)
            self.vnf_workloads_mi = tuple(share for _ in self.vnf_ids)
        if len(self.vnf_workloads_mi) != len(self.vnf_ids):
            raise ValueError("VNF workload vector must match SFC membership")
        if any(value <= 0 for value in self.vnf_workloads_mi):
            raise ValueError("Every VNF workload must be positive")
        if not self.vnf_deadlines_s:
            self.vnf_deadlines_s = tuple(
                self.deadline_s for _ in self.vnf_ids
            )
        if len(self.vnf_deadlines_s) != len(self.vnf_ids):
            raise ValueError("VNF deadline vector must match SFC membership")


@dataclass(slots=True)
class SFCGraph:
    """DAG of SFC scheduling units derived from one workflow."""

    workflow_id: str
    sfcs: dict[str, SFC]
    edges_mb: dict[tuple[str, str], float]
    vnf_to_sfc: dict[str, str]
    _predecessor_index: dict[str, tuple[str, ...]] = field(
        init=False, repr=False, default_factory=dict
    )
    _successor_index: dict[str, tuple[str, ...]] = field(
        init=False, repr=False, default_factory=dict
    )
    _topological_order_cache: tuple[str, ...] = field(
        init=False, repr=False, default=()
    )

    def __post_init__(self) -> None:
        predecessors: dict[str, list[str]] = {sfc_id: [] for sfc_id in self.sfcs}
        successors: dict[str, list[str]] = {sfc_id: [] for sfc_id in self.sfcs}
        for parent, child in self.edges_mb:
            if parent not in self.sfcs or child not in self.sfcs:
                raise ValueError(f"Unknown SFC edge endpoint: {(parent, child)}")
            predecessors[child].append(parent)
            successors[parent].append(child)
        self._predecessor_index = {
            sfc_id: tuple(sorted(values))
            for sfc_id, values in predecessors.items()
        }
        self._successor_index = {
            sfc_id: tuple(sorted(values))
            for sfc_id, values in successors.items()
        }
        self._topological_order_cache = self._calculate_topological_order()

    def predecessors(self, sfc_id: str) -> list[str]:
        return list(self._predecessor_index[sfc_id])

    def successors(self, sfc_id: str) -> list[str]:
        return list(self._successor_index[sfc_id])

    def _calculate_topological_order(self) -> tuple[str, ...]:
        indegree = {
            sfc_id: len(self._predecessor_index[sfc_id])
            for sfc_id in self.sfcs
        }
        ready = [sfc_id for sfc_id, degree in indegree.items() if degree == 0]
        heapq.heapify(ready)
        result: list[str] = []
        while ready:
            current = heapq.heappop(ready)
            result.append(current)
            for child in self._successor_index[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(ready, child)
        if len(result) != len(self.sfcs):
            raise ValueError("Derived SFC graph is not acyclic")
        return tuple(result)

    def topological_order(self) -> list[str]:
        return list(self._topological_order_cache)

    def ready(
        self,
        completed: set[str],
        scheduled: set[str] | None = None,
    ) -> list[str]:
        scheduled = scheduled or set()
        return [
            sfc_id
            for sfc_id in self._topological_order_cache
            if sfc_id not in completed
            and sfc_id not in scheduled
            and all(
                parent in completed
                for parent in self._predecessor_index[sfc_id]
            )
        ]


@dataclass(frozen=True, slots=True)
class Assignment:
    """Observed SFC placement and execution decision."""

    sfc_id: str
    vm_id: str
    node_id: str
    start_s: float
    finish_s: float
    communication_latency_s: float
    transferred_data_mb: float
    reused_vm: bool
    scheduler: str
    solver_status: str = "not_applicable"
    source_node_id: str = "edge"
    propagation_latency_s: float = 0.0
    serialization_delay_s: float = 0.0
    queueing_delay_s: float = 0.0
    vnf_completion_times_s: tuple[float, ...] = ()
    terminal_delivery_s: float | None = None

    def __post_init__(self) -> None:
        if self.finish_s + 1e-9 < self.start_s:
            raise ValueError("Assignment finish time must not precede start time")
        if self.transferred_data_mb < 0:
            raise ValueError("Transferred data cannot be negative")
        for name in (
            "communication_latency_s",
            "propagation_latency_s",
            "serialization_delay_s",
            "queueing_delay_s",
        ):
            if getattr(self, name) < -1e-9:
                raise ValueError(f"{name} cannot be negative")
        if self.vnf_completion_times_s:
            if any(
                self.vnf_completion_times_s[index] + 1e-9
                < self.vnf_completion_times_s[index - 1]
                for index in range(1, len(self.vnf_completion_times_s))
            ):
                raise ValueError("VNF completion times must be monotone")
            if (
                self.vnf_completion_times_s[-1] + 1e-7
                < self.finish_s
                or self.finish_s + 1e-7
                < self.vnf_completion_times_s[-1]
            ):
                raise ValueError(
                    "Final VNF completion must equal SFC finish time"
                )
        if (
            self.terminal_delivery_s is not None
            and self.terminal_delivery_s + 1e-9 < self.finish_s
        ):
            raise ValueError("Terminal delivery cannot precede SFC completion")

    @property
    def execution_time_s(self) -> float:
        return self.finish_s - self.start_s


def pairwise(items: Iterable[str]) -> Iterator[tuple[str, str]]:
    """Yield adjacent pairs."""

    iterator = iter(items)
    try:
        previous = next(iterator)
    except StopIteration:
        return
    for item in iterator:
        yield previous, item
        previous = item
