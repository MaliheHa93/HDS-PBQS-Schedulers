"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import result_record, run_case, run_workflow
from .workflow_loader import load_dax, load_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one HDS/PBQS paper simulation"
    )
    parser.add_argument(
        "--family",
        choices=["montage", "epigenomics", "inspiral", "cybershake"],
        default="montage",
    )
    parser.add_argument("--size", type=int, default=25)
    parser.add_argument("--deadline-factor", type=float, default=1.25)
    parser.add_argument(
        "--deadline-mode",
        choices=["reference", "serial"],
        default="reference",
    )
    parser.add_argument(
        "--workflow-file",
        type=Path,
        help="Run a project JSON or Pegasus DAX/XML workflow instead",
    )
    parser.add_argument(
        "--deadline-s",
        type=float,
        help="Absolute deadline required for a DAX/XML workflow",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--scheduler",
        choices=["hds", "pbqs", "edf", "cost"],
        default="hds",
    )
    parser.add_argument(
        "--sharing",
        choices=["sharable", "nonsharable"],
        default="sharable",
    )
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--solver-time-limit", type=float, default=15.0)
    parser.add_argument("--eta", type=float, default=0.7)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--omega-u", type=float, default=0.7)
    parser.add_argument("--omega-w", type=float, default=0.3)
    parser.add_argument(
        "--tastd-mode",
        choices=["downstream_reserved", "topology"],
        default="downstream_reserved",
    )
    parser.add_argument("--disable-vm-reuse", action="store_true")
    parser.add_argument(
        "--reuse-policy",
        choices=["auto", "idle_only", "queue_aware", "none"],
        default="auto",
    )
    parser.add_argument("--disable-joint-bos", action="store_true")
    parser.add_argument("--disable-earliest-start", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sharable = args.sharing == "sharable"
    reuse_policy = (
        ("idle_only" if args.scheduler == "hds" else "queue_aware")
        if args.reuse_policy == "auto"
        else args.reuse_policy
    )
    common = {
        "scheduler_name": args.scheduler,
        "sharable": sharable,
        "candidate_count": args.candidate_count,
        "solver_time_limit_s": args.solver_time_limit,
        "eta": args.eta,
        "alpha": args.alpha,
        "beta": args.beta,
        "hds_gamma": args.gamma,
        "omega_u": args.omega_u,
        "omega_w": args.omega_w,
        "tastd_mode": args.tastd_mode,
        "enable_vm_reuse": not args.disable_vm_reuse,
        "reuse_policy": reuse_policy,
        "joint_bos_optimization": not args.disable_joint_bos,
        "reconstruct_earliest_start": not args.disable_earliest_start,
    }
    family = args.family
    size = args.size
    deadline_factor = args.deadline_factor
    if args.workflow_file:
        suffix = args.workflow_file.suffix.lower()
        if suffix == ".json":
            workflow = load_json(args.workflow_file)
        elif suffix in {".xml", ".dax"}:
            if args.deadline_s is None or args.deadline_s <= 0:
                raise SystemExit(
                    "--deadline-s must be positive for DAX/XML workflows"
                )
            workflow = load_dax(
                args.workflow_file,
                deadline_s=args.deadline_s,
            )
        else:
            raise SystemExit("--workflow-file must be JSON, XML, or DAX")
        result = run_workflow(workflow=workflow, **common)
        family = workflow.family
        size = len(workflow.vnfs)
        deadline_factor = (
            (workflow.deadline_s - workflow.arrival_s)
            / result.tastd.minimum_makespan_s
        )
    else:
        result = run_case(
            family=args.family,
            size=args.size,
            deadline_factor=args.deadline_factor,
            seed=args.seed,
            deadline_mode=args.deadline_mode,
            **common,
        )
    record = result_record(
        result,
        family=family,
        size=size,
        deadline_factor=deadline_factor,
        seed=args.seed,
        scheduler=args.scheduler,
        sharable=sharable,
        enable_vm_reuse=not args.disable_vm_reuse,
        reuse_policy=reuse_policy,
        joint_bos_optimization=not args.disable_joint_bos,
        deadline_mode=args.deadline_mode,
        reconstruct_earliest_start=not args.disable_earliest_start,
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
