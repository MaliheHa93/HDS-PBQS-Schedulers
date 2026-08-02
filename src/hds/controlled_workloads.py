"""Controlled workloads that isolate BoS-level scheduling behavior."""

from __future__ import annotations

import random

from .models import VNF, Workflow


def independent_bos_workflow(
    bos_size: int,
    seed: int,
    *,
    provisional_deadline_s: float = 1.0,
) -> Workflow:
    """Create one fixed-size bag of mutually independent one-VNF SFCs."""

    if bos_size <= 0:
        raise ValueError("bos_size must be positive")
    rng = random.Random(seed)
    vnfs = {}
    for index in range(bos_size):
        runtime_s = rng.uniform(0.5, 4.0)
        identifier = f"bos-{index:03d}"
        vnfs[identifier] = VNF(
            id=identifier,
            workload_mi=runtime_s * 16_000,
            cpu_mips=rng.uniform(500, 4_000),
            ram_mb=rng.uniform(64, 240),
            storage_mb=rng.uniform(32, 192),
            source_data_mb=rng.uniform(1.0, 8.0),
            sink_data_mb=rng.uniform(0.1, 2.0),
        )
    return Workflow(
        id=f"controlled-bos-{bos_size}-seed-{seed}",
        family="controlled_bos",
        vnfs=vnfs,
        edges=[],
        deadline_s=provisional_deadline_s,
    )
