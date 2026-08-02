#!/usr/bin/env python3
"""Run the deadline-sensitivity matrix and preserve every raw observation.

Rows are flushed immediately. A stopped run can be resumed without discarding
or duplicating completed cases.
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

from hds.csv_output import DetailCsvWriter  # noqa: E402
from hds.experiment import (  # noqa: E402
    assignment_records,
    link_records,
    node_records,
    result_record,
    run_case,
)


def parse_csv_list(value: str, cast):
    result = [cast(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("At least one value is required")
    return list(dict.fromkeys(result))


def parse_integer_spec(value: str) -> list[int]:
    result: list[int] = []
    for item in parse_csv_list(value, str):
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending range: {item}")
            result.extend(range(start, end + 1))
        else:
            result.append(int(item))
    return list(dict.fromkeys(result))


CASE_FIELDS = (
    "family",
    "workflow_size",
    "deadline_factor",
    "deadline_mode",
    "seed",
    "configuration",
    "candidate_count",
    "eta",
    "alpha",
    "beta",
    "gamma",
    "omega_u",
    "omega_w",
    "tastd_mode",
    "enable_vm_reuse",
    "reuse_policy",
    "joint_bos_optimization",
    "adaptive_bos_fallback",
    "reconstruct_earliest_start",
)


def case_key(record: dict[str, object]) -> tuple[str, ...]:
    return tuple(str(record[field]) for field in CASE_FIELDS)


def completed_keys(path: Path) -> set[tuple[str, ...]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise SystemExit(f"{path} has no CSV header")
        missing = set(CASE_FIELDS) - set(reader.fieldnames)
        if missing:
            raise SystemExit(
                f"{path} cannot be resumed; missing columns: {sorted(missing)}"
            )
        return {case_key(row) for row in reader}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--families",
        default="montage,epigenomics,inspiral,cybershake",
    )
    result.add_argument("--sizes", default="100")
    result.add_argument(
        "--deadline-factors",
        default=(
            "0.8,0.9,1.0,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,"
            "1.45,1.5,1.6,1.7,1.8,1.9,2.0,2.25,2.5,2.75,3.0,"
            "3.25,3.5,3.75,4.0,4.25,4.5,4.75,5.0"
        ),
        help="Kappa values relative to the critical-path reference by default",
    )
    result.add_argument(
        "--deadline-mode",
        choices=["reference", "serial"],
        default="reference",
    )
    result.add_argument(
        "--seeds",
        default="1-50",
        help="Comma-separated seeds and/or inclusive ranges, e.g. 1-30",
    )
    result.add_argument("--candidate-count", type=int, default=20)
    result.add_argument("--solver-time-limit", type=float, default=15)
    result.add_argument("--eta", type=float, default=0.7)
    result.add_argument("--alpha", type=float, default=0.7)
    result.add_argument("--beta", type=float, default=0.2)
    result.add_argument("--gamma", type=float, default=0.1)
    result.add_argument("--omega-u", type=float, default=0.7)
    result.add_argument("--omega-w", type=float, default=0.3)
    result.add_argument(
        "--reuse-policy",
        choices=["auto", "idle_only", "queue_aware", "none"],
        default="auto",
        help=(
            "Reuse policy override. 'auto' uses idle-only reuse for HDS and "
            "queue-aware reuse for PBQS, matching the manuscript."
        ),
    )
    result.add_argument(
        "--disable-earliest-start",
        action="store_true",
        help="Diagnostic only: retain raw MILP timing variables",
    )
    result.add_argument(
        "--tastd-mode",
        choices=["downstream_reserved", "topology"],
        default="downstream_reserved",
    )
    result.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT
            / "results/raw/deadline_curves_paper_aligned_50seeds.csv"
        ),
    )
    result.add_argument("--node-output", type=Path)
    result.add_argument("--link-output", type=Path)
    result.add_argument("--trace-output", type=Path)
    result.add_argument(
        "--smoke",
        action="store_true",
        help="Use a small four-configuration validation matrix",
    )
    result.add_argument(
        "--resume",
        action="store_true",
        help="Append only cases not already present in the output",
    )
    result.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output instead of refusing to start",
    )
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and print the case count without executing",
    )
    result.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress after this many new rows; use 0 to disable.",
    )
    result.add_argument("--quiet", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite cannot be used together")
    if args.solver_time_limit <= 0:
        raise SystemExit("--solver-time-limit must be positive")
    if args.candidate_count <= 0:
        raise SystemExit("--candidate-count must be positive")
    if not 0 < args.eta <= 1:
        raise SystemExit("--eta must be within (0, 1]")
    if abs(args.alpha + args.beta + args.gamma - 1.0) > 1e-9:
        raise SystemExit("--alpha + --beta + --gamma must equal 1")
    if abs(args.omega_u + args.omega_w - 1.0) > 1e-9:
        raise SystemExit("--omega-u + --omega-w must equal 1")

    if args.smoke:
        families = ["montage"]
        sizes = [25]
        deadlines = [6.0]
        seeds = [1, 2]
    else:
        families = parse_csv_list(args.families, str)
        sizes = parse_csv_list(args.sizes, int)
        deadlines = parse_csv_list(args.deadline_factors, float)
        seeds = parse_integer_spec(args.seeds)
    if any(size <= 0 for size in sizes):
        raise SystemExit("--sizes values must be positive")
    if any(deadline <= 0 for deadline in deadlines):
        raise SystemExit("--deadline-factors values must be positive")

    variants = [
        ("hds", True),
        ("hds", False),
        ("pbqs", True),
        ("pbqs", False),
    ]
    cases = list(product(families, sizes, deadlines, seeds, variants))
    if args.dry_run:
        print(
            f"Validated {len(cases)} cases across {len(families)} families, "
            f"{len(sizes)} sizes, {len(deadlines)} deadline factors, and "
            f"{len(seeds)} seeds."
        )
        return
    if args.output.exists() and not (args.resume or args.overwrite):
        raise SystemExit(
            f"{args.output} already exists; use --resume or --overwrite"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    already_done = completed_keys(args.output) if args.resume else set()
    mode = "a" if args.resume and args.output.exists() else "w"
    stream = args.output.open(mode, newline="", encoding="utf-8")
    writer: csv.DictWriter | None = None
    if mode == "a":
        with args.output.open(newline="", encoding="utf-8") as existing:
            reader = csv.DictReader(existing)
            if reader.fieldnames is None:
                raise SystemExit(f"{args.output} has no CSV header")
            writer = csv.DictWriter(stream, fieldnames=reader.fieldnames)

    started = time.perf_counter()
    executed = 0
    skipped = 0
    detail_append = mode == "a"
    node_writer = DetailCsvWriter(
        args.node_output,
        append=detail_append,
    )
    link_writer = DetailCsvWriter(
        args.link_output,
        append=detail_append,
    )
    trace_writer = DetailCsvWriter(
        args.trace_output,
        append=detail_append,
    )
    try:
        for index, (family, size, deadline, seed, variant) in enumerate(cases, 1):
            scheduler, sharable = variant
            reuse_policy = (
                ("idle_only" if scheduler == "hds" else "queue_aware")
                if args.reuse_policy == "auto"
                else args.reuse_policy
            )
            identity = {
                "family": family,
                "workflow_size": str(size),
                "deadline_factor": str(float(deadline)),
                "deadline_mode": args.deadline_mode,
                "seed": str(seed),
                "configuration": (
                    f"{scheduler.upper()}-"
                    f"{'Sharable' if sharable else 'NonSharable'}"
                ),
                "candidate_count": str(args.candidate_count),
                "eta": str(float(args.eta)),
                "alpha": str(float(args.alpha)),
                "beta": str(float(args.beta)),
                "gamma": str(float(args.gamma)),
                "omega_u": str(float(args.omega_u)),
                "omega_w": str(float(args.omega_w)),
                "tastd_mode": args.tastd_mode,
                "enable_vm_reuse": "1",
                "reuse_policy": reuse_policy,
                "joint_bos_optimization": "1",
                "adaptive_bos_fallback": "1",
                "reconstruct_earliest_start": str(
                    int(not args.disable_earliest_start)
                ),
            }
            if case_key(identity) in already_done:
                skipped += 1
                continue

            case_started = time.perf_counter()
            result = run_case(
                family,
                size,
                deadline,
                seed,
                scheduler,
                sharable,
                candidate_count=args.candidate_count,
                solver_time_limit_s=args.solver_time_limit,
                eta=args.eta,
                alpha=args.alpha,
                beta=args.beta,
                hds_gamma=args.gamma,
                omega_u=args.omega_u,
                omega_w=args.omega_w,
                tastd_mode=args.tastd_mode,
                reuse_policy=reuse_policy,
                adaptive_bos_fallback=True,
                deadline_mode=args.deadline_mode,
                reconstruct_earliest_start=(
                    not args.disable_earliest_start
                ),
            )
            record = result_record(
                result,
                family=family,
                size=size,
                deadline_factor=deadline,
                seed=seed,
                scheduler=scheduler,
                sharable=sharable,
                reuse_policy=reuse_policy,
                adaptive_bos_fallback=True,
                deadline_mode=args.deadline_mode,
                reconstruct_earliest_start=(
                    not args.disable_earliest_start
                ),
            )
            record.update(
                {
                    "topology": "small",
                    "node_count": 5,
                    "local_node_count": 3,
                    "global_node_count": 2,
                    "candidate_count": args.candidate_count,
                    "solver_time_limit_s": args.solver_time_limit,
                    "eta": args.eta,
                    "alpha": args.alpha,
                    "beta": args.beta,
                    "gamma": args.gamma,
                    "omega_u": args.omega_u,
                    "omega_w": args.omega_w,
                    "deadline_mode": args.deadline_mode,
                    "reuse_policy": reuse_policy,
                    "reconstruct_earliest_start": int(
                        not args.disable_earliest_start
                    ),
                    "case_wall_runtime_s": time.perf_counter() - case_started,
                }
            )
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(record))
                writer.writeheader()
            writer.writerow(record)
            stream.flush()
            details = {
                "family": family,
                "size": size,
                "deadline_factor": deadline,
                "seed": seed,
                "scheduler": scheduler,
                "sharable": sharable,
            }
            detail_extra = {
                "topology": "small",
                "node_count": 5,
                "candidate_count": args.candidate_count,
            }
            node_writer.write(
                node_records(result, **details),
                extra=detail_extra,
            )
            link_writer.write(
                link_records(result, **details),
                extra=detail_extra,
            )
            trace_writer.write(
                assignment_records(result, **details),
                extra=detail_extra,
            )
            executed += 1
            if (
                not args.quiet
                and args.progress_every > 0
                and executed % args.progress_every == 0
            ):
                print(
                    f"[{index}/{len(cases)}] completed {executed} new "
                    f"rows ({skipped} reused); latest={family} n={size} "
                    f"k={deadline} {record['configuration']}"
                )
    finally:
        stream.close()
        node_writer.close()
        link_writer.close()
        trace_writer.close()

    print(
        f"Wrote {executed} new unfiltered runs to {args.output}; "
        f"skipped {skipped}; elapsed {time.perf_counter() - started:.2f}s"
    )


if __name__ == "__main__":
    main()
