"""VM candidate generation, provisioning, reuse, and host capacities."""

from __future__ import annotations

from collections import defaultdict
import math

from .models import SFC, Tier, VMInstance, VMType
from .topology import FogTopology


REUSE_POLICIES = {"none", "idle_only", "queue_aware"}


def validate_reuse_policy(policy: str) -> str:
    """Return a normalized, supported VM-reuse policy."""

    normalized = policy.strip().lower().replace("-", "_")
    if normalized not in REUSE_POLICIES:
        raise ValueError(
            "reuse_policy must be one of: none, idle_only, queue_aware"
        )
    return normalized


def paper_vm_types() -> list[VMType]:
    """Return the two VM flavors listed in manuscript Table II."""

    return [
        VMType(
            id="local-standard",
            tier=Tier.LOCAL,
            speed_mips=8000,
            cpu_capacity_mips=8000,
            ram_mb=256,
            access_bandwidth_mbps=100,
            cost_per_period=0.005,
            storage_mb=100_000,
            startup_time_s=0.25,
        ),
        VMType(
            id="global-standard",
            tier=Tier.GLOBAL,
            speed_mips=16000,
            cpu_capacity_mips=16000,
            ram_mb=512,
            access_bandwidth_mbps=1000,
            cost_per_period=0.01,
            storage_mb=500_000,
            startup_time_s=0.50,
        ),
    ]


class VMCapacityError(ValueError):
    """Raised when a VM cannot be activated on its selected fog node."""


class VMPool:
    """Stateful set of provisioned VM instances."""

    def __init__(self, topology: FogTopology, vm_types: list[VMType]) -> None:
        self.topology = topology
        self.vm_types = {vm_type.id: vm_type for vm_type in vm_types}
        self.instances: dict[str, VMInstance] = {}
        self._serial = 0

    def active_instances(self, time_s: float) -> list[VMInstance]:
        return [
            instance
            for instance in self.instances.values()
            if instance.paid_until_s is not None
            and instance.paid_until_s > time_s + 1e-9
        ]

    def reusable_idle(self, time_s: float) -> list[VMInstance]:
        """Backward-compatible idle-only candidate query."""

        return self.reusable_instances(time_s, "idle_only")

    def reusable_instances(
        self,
        time_s: float,
        policy: str,
    ) -> list[VMInstance]:
        """Return the legal paid VM candidates under a common policy."""

        normalized = validate_reuse_policy(policy)
        if normalized == "none":
            return []
        return sorted(
            (
                instance
                for instance in self.active_instances(time_s)
                if normalized == "queue_aware"
                or instance.is_idle_at(time_s)
            ),
            key=lambda item: (
                item.vm_type.cost_per_period,
                item.paid_until_s or 0.0,
                item.id,
            ),
        )

    def residual_node_capacities(
        self, time_s: float
    ) -> dict[str, tuple[float, float, float]]:
        used_cpu: dict[str, float] = defaultdict(float)
        used_ram: dict[str, float] = defaultdict(float)
        used_storage: dict[str, float] = defaultdict(float)
        for instance in self.active_instances(time_s):
            used_cpu[instance.node_id] += instance.vm_type.cpu_capacity_mips
            used_ram[instance.node_id] += instance.vm_type.ram_mb
            used_storage[instance.node_id] += instance.vm_type.storage_mb
        return {
            node_id: (
                max(0.0, node.cpu_mips - used_cpu[node_id]),
                max(0.0, node.ram_mb - used_ram[node_id]),
                max(0.0, node.storage_mb - used_storage[node_id]),
            )
            for node_id, node in self.topology.nodes.items()
        }

    def candidate_instances(
        self,
        time_s: float,
        maximum: int = 20,
    ) -> list[VMInstance]:
        """Generate deterministic candidate instances for one MILP round."""

        if maximum <= 0:
            return []
        candidates: list[VMInstance] = []
        sequence = 0
        nodes = sorted(
            self.topology.nodes.values(),
            key=lambda node: (node.tier != Tier.LOCAL, node.id),
        )
        types = sorted(self.vm_types.values(), key=lambda item: item.id)
        residual = self.residual_node_capacities(time_s)
        slots: list[tuple[str, VMType, int]] = []
        for node in nodes:
            for vm_type in types:
                if vm_type.tier != node.tier:
                    continue
                cpu, ram, storage = residual[node.id]
                slot_count = int(
                    min(
                        cpu // vm_type.cpu_capacity_mips,
                        ram // vm_type.ram_mb,
                        storage // vm_type.storage_mb,
                    )
                )
                slots.extend(
                    (node.id, vm_type, slot_index)
                    for slot_index in range(slot_count)
                )
        for node_id, vm_type, slot_index in slots[:maximum]:
            candidate_id = (
                f"candidate-{self._serial}-{sequence:04d}-"
                f"{node_id}-{vm_type.id}-slot{slot_index:03d}"
            )
            candidates.append(VMInstance(candidate_id, vm_type, node_id))
            sequence += 1
        return candidates

    def can_host_new(
        self,
        node_id: str,
        vm_type: VMType,
        time_s: float,
    ) -> bool:
        cpu, ram, storage = self.residual_node_capacities(time_s)[node_id]
        return (
            vm_type.cpu_capacity_mips <= cpu + 1e-9
            and vm_type.ram_mb <= ram + 1e-9
            and vm_type.storage_mb <= storage + 1e-9
        )

    def activate(
        self,
        candidate: VMInstance,
        time_s: float,
        periods: int,
    ) -> VMInstance:
        if periods < 1:
            raise ValueError("An activated VM needs at least one billing period")
        if not self.can_host_new(candidate.node_id, candidate.vm_type, time_s):
            raise VMCapacityError(
                f"Node {candidate.node_id} cannot host {candidate.vm_type.id}"
            )
        self._serial += 1
        instance_id = f"vm-{self._serial:06d}"
        instance = VMInstance(
            id=instance_id,
            vm_type=candidate.vm_type,
            node_id=candidate.node_id,
            provisioned_at_s=time_s,
            paid_until_s=(
                time_s + periods * candidate.vm_type.billing_period_s
            ),
        )
        self.instances[instance_id] = instance
        return instance

    @staticmethod
    def execution_time_s(sfc: SFC, vm: VMInstance) -> float:
        return sfc.workload_mi / vm.vm_type.speed_mips

    @staticmethod
    def resource_feasible(sfc: SFC, vm: VMInstance) -> bool:
        return (
            sfc.cpu_mips <= vm.vm_type.cpu_capacity_mips + 1e-9
            and sfc.ram_mb <= vm.vm_type.ram_mb + 1e-9
            and sfc.storage_mb <= vm.vm_type.storage_mb + 1e-9
            and sfc.bandwidth_mbps
            <= vm.vm_type.access_bandwidth_mbps + 1e-9
        )

    @staticmethod
    def periods_for_duration(vm: VMInstance, duration_s: float) -> int:
        return max(1, math.ceil(duration_s / vm.vm_type.billing_period_s))
