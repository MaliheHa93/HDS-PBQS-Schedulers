"""Earliest-Deadline-First baseline with the common placement semantics."""

from __future__ import annotations

from .models import SFC
from .pbqs_scheduler import PBQSConfig, PBQSScheduler


class EDFScheduler(PBQSScheduler):
    """Recognized deadline baseline using the same VM and network model.

    EDF changes only the ready-SFC priority rule. VM reuse, local-first
    placement, billing, resource feasibility, and sharable/non-sharable
    behavior remain identical to PBQS so comparisons are controlled.
    """

    name = "EDF"

    def __init__(self, *args, config: PBQSConfig | None = None, **kwargs) -> None:
        super().__init__(*args, config=config or PBQSConfig(), **kwargs)

    def _priority(self, sfcs: list[SFC], current_s: float) -> list[SFC]:
        del current_s
        return sorted(sfcs, key=lambda sfc: (sfc.deadline_s, sfc.id))
