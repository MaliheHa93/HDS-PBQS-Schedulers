"""Incremental-cost-first heuristic under the common scheduling semantics."""

from __future__ import annotations

from dataclasses import replace

from .models import SFC
from .pbqs_scheduler import PBQSConfig, PBQSScheduler
from .topology import FogTopology
from .vm_pool import VMPool


class CostFirstScheduler(PBQSScheduler):
    """Prefer paid reuse, then the lowest incremental provisioning cost.

    The baseline changes only ready-SFC ordering and placement preference.
    VM eligibility, deadline checks, billing, network admission, and
    SpaceShared execution remain identical to PBQS.
    """

    name = "COST"

    def __init__(
        self,
        topology: FogTopology,
        vm_pool: VMPool,
        config: PBQSConfig | None = None,
    ) -> None:
        base = config or PBQSConfig()
        super().__init__(
            topology,
            vm_pool,
            replace(base, placement_policy="cost_first"),
        )

    def _priority(self, sfcs: list[SFC], current_s: float) -> list[SFC]:
        del current_s
        return sorted(
            sfcs,
            key=lambda sfc: (sfc.deadline_s, -sfc.workload_mi, sfc.id),
        )
