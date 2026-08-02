#!/usr/bin/env python3
"""Run paper-aligned deployment profiles derived from Table III.

The profiles increase fog capacity and workflow size together.  They answer
the end-to-end question "what happens as the deployment grows?"  Controlled
BoS scaling is run separately so that batch width is not conflated with this
deployment-profile axis.

Rows are flushed after every case.  ``--resume`` safely reuses completed
profiles from an earlier paper-aligned run and executes only missing profiles.
"""

from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from hds.experiment import result_record, run_case  # noqa: E402
from hds.topology import paper_scaled_topology  # noqa: E402


SCALES = {
    # Upper boundaries of the original Table III profiles.
    "small": (3, 2, 100),
    "small_plus": (5, 3, 200),
    "medium": (8, 4, 300),
    "medium_plus": (11, 7, 450),
    "large": (15, 10, 600),
    # New measured profiles beyond the original Table III boundary.
    "large_plus": (18, 12, 700),
    "xlarge_minus": (21, 14, 750),
    "xlarge": (24, 16, 800),
    "xlarge_plus": (30, 20, 900),
    "xxlarge": (36, 24, 1000),
}

KEY_FIELDS = ("topology", "family", "seed", "configuration")


def configuration_name(scheduler: str, sharable: bool) -> str:
    return (
        f"{scheduler.upper()}-"
        f"{'Sharable' if sharable else 'NonSharable'}"
    )


def csv_names(value: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(item.strip() for item in value.split(",") if item.strip())
    )


def completed_keys(path: Path) -> set[tuple[str, ...]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise SystemExit(f"{path} has no CSV header")
        missing = set(KEY_FIELDS) - set(reader.fieldnames)
        if missing:
            raise SystemExit(
                f"{path} cannot be resumed; missing columns: "
                f"{sorted(missing)}"
            )
        return {
            tuple(str(row[field]) for field in KEY_FIELDS)
            for row in reader
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--kappa", type=float, default=3.0)
    parser.add_argument("--solver-time-limit", type=float, default=15.0)
    parser.add_argument(
        "--profiles",
        default=",".join(SCALES),
        help="Comma-separated deployment profiles; defaults to D1-D10.",
    )
    parser.add_argument(
        "--families",
        default="montage,epigenomics,inspiral,cybershake",
        help="Comma-separated workflow families.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "results/raw/deployment_scaling_paper_aligned.csv",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=20)
    args = parser.parse_args()
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite cannot be used together")
    if args.output.exists() and not (args.resume or args.overwrite):
        raise SystemExit(
            f"{args.output} exists; pass --resume or --overwrite"
        )
    if args.seeds <= 0:
        raise SystemExit("--seeds must be positive")
    if args.kappa <= 0:
        raise SystemExit("--kappa must be positive")
    if args.solver_time_limit <= 0:
        raise SystemExit("--solver-time-limit must be positive")

    profiles = csv_names(args.profiles)
    unknown_profiles = set(profiles) - set(SCALES)
    if unknown_profiles:
        raise SystemExit(
            f"Unknown deployment profiles: {sorted(unknown_profiles)}"
        )
    families = csv_names(args.families)
    supported_families = {
        "montage",
        "epigenomics",
        "inspiral",
        "cybershake",
    }
    unknown_families = set(families) - supported_families
    if unknown_families:
        raise SystemExit(
            f"Unknown workflow families: {sorted(unknown_families)}"
        )
    if not profiles or not families:
        raise SystemExit("At least one profile and workflow family are required")
    variants = (("hds", True), ("hds", False), ("pbqs", True), ("pbqs", False))
    cases = list(product(profiles, families, range(1, args.seeds + 1), variants))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    already_done = completed_keys(args.output) if args.resume else set()
    mode = "a" if args.resume and args.output.exists() else "w"
    started = time.perf_counter()
    executed = 0
    skipped = 0
    writer = None
    fieldnames = None
    if mode == "a":
        with args.output.open(newline="", encoding="utf-8") as existing:
            reader = csv.DictReader(existing)
            fieldnames = reader.fieldnames
        if fieldnames is None:
            raise SystemExit(f"{args.output} has no CSV header")
    with args.output.open(mode, newline="", encoding="utf-8") as stream:
        if fieldnames is not None:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
        for scale, family, seed, (scheduler, sharable) in cases:
            key = (
                scale,
                family,
                str(seed),
                configuration_name(scheduler, sharable),
            )
            if key in already_done:
                skipped += 1
                continue
            local_count, global_count, size = SCALES[scale]
            topology = paper_scaled_topology(local_count, global_count)
            reuse_policy = (
                "idle_only" if scheduler == "hds" else "queue_aware"
            )
            case_started = time.perf_counter()
            result = run_case(
                family,
                size,
                args.kappa,
                seed,
                scheduler,
                sharable,
                topology=topology,
                candidate_count=len(topology.nodes),
                solver_time_limit_s=args.solver_time_limit,
                deadline_mode="reference",
                reuse_policy=reuse_policy,
            )
            row = result_record(
                result,
                family=family,
                size=size,
                deadline_factor=args.kappa,
                seed=seed,
                scheduler=scheduler,
                sharable=sharable,
                reuse_policy=reuse_policy,
                deadline_mode="reference",
            )
            row.update(
                {
                    "study": "deployment_scaling",
                    "topology": scale,
                    "node_count": len(topology.nodes),
                    "local_node_count": local_count,
                    "global_node_count": global_count,
                    "candidate_count": len(topology.nodes),
                    "effective_candidate_count": (
                        result.initial_effective_candidate_count
                    ),
                    "candidate_node_coverage": (
                        result.initial_effective_candidate_count
                        / len(topology.nodes)
                    ),
                    "solver_time_limit_s": args.solver_time_limit,
                    "eta": 0.7,
                    "alpha": 0.7,
                    "beta": 0.2,
                    "gamma": 0.1,
                    "omega_u": 0.7,
                    "omega_w": 0.3,
                    "case_wall_runtime_s": time.perf_counter() - case_started,
                }
            )
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            executed += 1
            if args.progress_every > 0 and executed % args.progress_every == 0:
                print(
                    f"Completed {executed} new cases "
                    f"({skipped} reused; {len(cases)} total profiles)"
                )
    print(
        f"Deployment scaling complete: {executed} new, {skipped} reused, "
        f"{len(cases)} total rows in "
        f"{time.perf_counter() - started:.2f}s"
    )


if __name__ == "__main__":
    main()
