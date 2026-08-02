#!/usr/bin/env python3
"""Run the paper-aligned controlled BoS-width scalability study."""

from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from hds.controlled_workloads import independent_bos_workflow  # noqa: E402
from hds.experiment import (  # noqa: E402
    apply_deadline_policy,
    result_record,
    run_workflow,
)
from hds.models import Tier  # noqa: E402
from hds.topology import paper_base_topology  # noqa: E402


VARIANTS = {
    "hds_full": ("hds", True, True),
    "hds_no_joint": ("hds", False, True),
    "pbqs": ("pbqs", True, True),
    "edf": ("edf", True, True),
    "cost": ("cost", True, True),
}


def integer_list(value: str) -> list[int]:
    values: list[int] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, finish_text = item.split("-", 1)
            start, finish = int(start_text), int(finish_text)
            if finish < start:
                raise ValueError(f"Invalid descending range: {item}")
            values.extend(range(start, finish + 1))
        else:
            values.append(int(item))
    return list(dict.fromkeys(values))


def float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def seed_list(value: str) -> list[int]:
    if "-" in value and "," not in value:
        start, finish = (int(item) for item in value.split("-", 1))
        return list(range(start, finish + 1))
    return integer_list(value)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--bos-sizes",
        default="2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20",
    )
    result.add_argument("--kappas", default="5.0")
    result.add_argument("--seeds", default="1-5")
    result.add_argument(
        "--variants",
        default="hds_full,pbqs",
    )
    result.add_argument(
        "--sharing",
        choices=["sharable", "nonsharable", "both"],
        default="both",
    )
    result.add_argument(
        "--candidate-counts",
        default="20",
        help="Comma-separated candidate counts, e.g. 5,10,20,30",
    )
    result.add_argument("--solver-time-limit", type=float, default=15.0)
    result.add_argument(
        "--reuse-policy",
        choices=["auto", "idle_only", "queue_aware", "none"],
        default="auto",
    )
    result.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "results/raw/bos_scaling_paper_aligned.csv",
    )
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--quiet", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    bos_sizes = integer_list(args.bos_sizes)
    kappas = float_list(args.kappas)
    seeds = seed_list(args.seeds)
    candidate_counts = integer_list(args.candidate_counts)
    if not bos_sizes or any(value <= 0 for value in bos_sizes):
        raise SystemExit("--bos-sizes must contain positive integers")
    if not candidate_counts or any(value <= 0 for value in candidate_counts):
        raise SystemExit("--candidate-counts must contain positive integers")
    if not kappas or any(value <= 0 for value in kappas):
        raise SystemExit("--kappas must contain positive values")
    if not seeds or any(value <= 0 for value in seeds):
        raise SystemExit("--seeds must contain positive integers")
    variant_names = [
        item.strip() for item in args.variants.split(",") if item.strip()
    ]
    unknown = set(variant_names) - set(VARIANTS)
    if unknown:
        raise SystemExit(f"Unknown variants: {sorted(unknown)}")
    sharing_modes = (
        [True, False]
        if args.sharing == "both"
        else [args.sharing == "sharable"]
    )
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists; pass --overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    topology = paper_base_topology()
    cases = list(
        product(
            bos_sizes,
            kappas,
            seeds,
            variant_names,
            sharing_modes,
            candidate_counts,
        )
    )
    started = time.perf_counter()
    writer = None
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        for index, (
            bos_size,
            kappa,
            seed,
            variant,
            sharable,
            candidate_count,
        ) in enumerate(
            cases,
            1,
        ):
            scheduler, joint, adaptive = VARIANTS[variant]
            reuse_policy = (
                ("idle_only" if scheduler == "hds" else "queue_aware")
                if args.reuse_policy == "auto"
                else args.reuse_policy
            )
            workflow = independent_bos_workflow(bos_size, seed)
            workflow = apply_deadline_policy(
                workflow,
                deadline_factor=kappa,
                topology=topology,
                deadline_mode="reference",
            )
            case_started = time.perf_counter()
            result = run_workflow(
                workflow,
                scheduler_name=scheduler,
                sharable=sharable,
                topology=topology,
                candidate_count=candidate_count,
                solver_time_limit_s=args.solver_time_limit,
                reuse_policy=reuse_policy,
                joint_bos_optimization=joint,
                adaptive_bos_fallback=adaptive,
            )
            record = result_record(
                result,
                family="controlled_bos",
                size=bos_size,
                deadline_factor=kappa,
                seed=seed,
                scheduler=scheduler,
                sharable=sharable,
                reuse_policy=reuse_policy,
                joint_bos_optimization=joint,
                adaptive_bos_fallback=adaptive,
                deadline_mode="reference",
            )
            record.update(
                {
                    "study": "bos_scaling",
                    "variant": variant,
                    "controlled_bos_size": bos_size,
                    "topology": "small",
                    "node_count": len(topology.nodes),
                    "local_node_count": sum(
                        node.tier == Tier.LOCAL
                        for node in topology.nodes.values()
                    ),
                    "global_node_count": sum(
                        node.tier == Tier.GLOBAL
                        for node in topology.nodes.values()
                    ),
                    "candidate_count": candidate_count,
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
                    "case_wall_runtime_s": (
                        time.perf_counter() - case_started
                    ),
                }
            )
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(record))
                writer.writeheader()
            writer.writerow(record)
            stream.flush()
            if not args.quiet:
                print(
                    f"[{index}/{len(cases)}] BoS={bos_size} k={kappa} "
                    f"{variant} {'S' if sharable else 'NS'} "
                    f"c={candidate_count} "
                    "first-round accepted="
                    f"{record['first_round_accepted_sfc_ratio']:.2f}"
                )
    print(
        f"Wrote {len(cases)} paired BoS-study runs to {args.output} "
        f"in {time.perf_counter() - started:.2f}s"
    )


if __name__ == "__main__":
    main()
