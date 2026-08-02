#!/usr/bin/env python3
"""Derive manuscript-safe claims directly from the validated v0.7.0 rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from .plot_four_publication_figures_paper_aligned import (
        CONFIGS,
        FAMILIES,
        common_global_success,
    )
except ImportError:
    from plot_four_publication_figures_paper_aligned import (
        CONFIGS,
        FAMILIES,
        common_global_success,
    )


PROJECT = Path(__file__).resolve().parents[1]


def sharing_effects(
    deadline: pd.DataFrame,
    *,
    deadline_factor: float,
) -> dict[str, object]:
    selected = common_global_success(
        deadline[deadline.deadline_factor.eq(deadline_factor)]
    )
    metrics = (
        "provisioning_cost",
        "network_data_mb",
        "end_to_end_delay_s",
        "cpu_utilization",
        "ram_utilization",
    )
    result: dict[str, object] = {}
    for scheduler in ("HDS", "PBQS"):
        part = selected[selected.scheduler.eq(scheduler)]
        scheduler_result: dict[str, object] = {
            "paired_seeds_by_workflow": {
                family: int(part[part.family.eq(family)].seed.nunique())
                for family in FAMILIES
            }
        }
        for metric in metrics:
            levels: dict[str, dict[str, float]] = {}
            for family in FAMILIES:
                family_rows = part[part.family.eq(family)]
                levels[family] = {
                    sharing: float(
                        family_rows[
                            family_rows.sharing.eq(sharing)
                        ][metric].mean()
                    )
                    for sharing in ("Sharable", "NonSharable")
                }
            sharable_macro = sum(
                values["Sharable"] for values in levels.values()
            ) / len(FAMILIES)
            nonsharable_macro = sum(
                values["NonSharable"] for values in levels.values()
            ) / len(FAMILIES)
            relative_change = (
                100.0
                * (sharable_macro - nonsharable_macro)
                / nonsharable_macro
            )
            scheduler_result[metric] = {
                "sharable_macro": sharable_macro,
                "nonsharable_macro": nonsharable_macro,
                "sharable_relative_change_pct": relative_change,
                "sharable_absolute_change": (
                    sharable_macro - nonsharable_macro
                ),
                "workflow_levels": levels,
            }
        result[scheduler] = scheduler_result
    return {
        "deadline_factor": deadline_factor,
        "filter": (
            "Within each workflow and seed, retain only cases where all four "
            "configurations succeed globally; average within workflow, then "
            "average the four workflow means equally."
        ),
        "schedulers": result,
    }


def success_differences(deadline: pd.DataFrame) -> dict[str, object]:
    rates = (
        deadline.groupby(
            ["family", "sharing", "deadline_factor", "scheduler"],
            sort=True,
        )
        .global_deadline_success.mean()
        .unstack("scheduler")
        .reset_index()
    )
    rates["hds_minus_pbqs_pp"] = 100.0 * (
        rates["HDS"] - rates["PBQS"]
    )
    original = rates[rates.deadline_factor.le(3.0)]
    best = original.loc[original.hds_minus_pbqs_pp.idxmax()]
    worst = original.loc[original.hds_minus_pbqs_pp.idxmin()]
    return {
        "original_range_kappa_max": 3.0,
        "maximum_hds_advantage": {
            "family": str(best.family),
            "sharing": str(best.sharing),
            "deadline_factor": float(best.deadline_factor),
            "percentage_points": float(best.hds_minus_pbqs_pp),
        },
        "maximum_hds_disadvantage": {
            "family": str(worst.family),
            "sharing": str(worst.sharing),
            "deadline_factor": float(worst.deadline_factor),
            "percentage_points": float(worst.hds_minus_pbqs_pp),
        },
    }


def dataset_audit(
    deadline: pd.DataFrame,
    deployment: pd.DataFrame,
    bos: pd.DataFrame,
) -> dict[str, object]:
    identities = {
        "deadline": [
            "family",
            "workflow_size",
            "deadline_factor",
            "seed",
            "configuration",
        ],
        "deployment": [
            "topology",
            "family",
            "seed",
            "configuration",
        ],
        "bos": [
            "controlled_bos_size",
            "seed",
            "configuration",
        ],
    }
    frames = {
        "deadline": deadline,
        "deployment": deployment,
        "bos": bos,
    }
    audits: dict[str, object] = {}
    for name, frame in frames.items():
        audits[name] = {
            "rows": len(frame),
            "duplicate_case_rows": int(
                frame.duplicated(identities[name], keep=False).sum()
            ),
            "solver_limit_rows": int(
                frame.milp_limit_reached_count.gt(0).sum()
            ),
            "stage1_limit_rows": int(
                frame.milp_admission_limit_reached_count.gt(0).sum()
            ),
            "stage2_limit_rows": int(
                frame.milp_secondary_limit_reached_count.gt(0).sum()
            ),
            "unfinished_rows": int(frame.unfinished_sfc_count.gt(0).sum()),
        }
    denominator_mismatches = int(
        bos.first_round_submitted_sfc_count.ne(
            bos.controlled_bos_size
        ).sum()
    )
    ratio_error = float(
        (
            bos.first_round_accepted_sfc_ratio
            - (
                bos.first_round_admitted_sfc_count
                / bos.first_round_submitted_sfc_count
            )
        ).abs().max()
    )
    return {
        "datasets": audits,
        "deployment_maximum": {
            "fog_nodes": int(deployment.node_count.max()),
            "workflow_vnfs": int(deployment.workflow_size.max()),
        },
        "bos_maximum_width": int(bos.controlled_bos_size.max()),
        "bos_first_round_denominator_mismatches": denominator_mismatches,
        "bos_first_round_ratio_max_error": ratio_error,
        "deadline_global_subdeadline_mismatch_rows": int(
            deadline.global_deadline_success.ne(
                deadline.sfc_subdeadline_success
            ).sum()
        ),
    }


def manuscript_markdown(payload: dict[str, object]) -> str:
    effects = payload["sharing_effects"]["schedulers"]

    def directional_change(metric: str, value: float) -> str:
        verb = "reduces" if value < 0 else "increases"
        return f"{verb} {metric} by {abs(value):.1f}\\%"

    scheduler_sentences: list[str] = []
    for scheduler in ("HDS", "PBQS"):
        values = effects[scheduler]
        cost = values["provisioning_cost"]["sharable_relative_change_pct"]
        transfer = values["network_data_mb"][
            "sharable_relative_change_pct"
        ]
        delay = values["end_to_end_delay_s"][
            "sharable_relative_change_pct"
        ]
        cpu_points = (
            100.0
            * values["cpu_utilization"]["sharable_absolute_change"]
        )
        ram_points = (
            100.0
            * values["ram_utilization"]["sharable_absolute_change"]
        )
        scheduler_sentences.append(
            f"For {scheduler}, sharing "
            f"{directional_change('provisioning cost', cost)}, "
            f"{directional_change('transferred-data volume', transfer)}, "
            f"{directional_change('end-to-end delay', delay)}, and changes "
            "CPU/RAM utilization by "
            f"{cpu_points:+.2f}/{ram_points:+.2f} percentage points."
        )
    audit = payload["audit"]
    deployment_limits = audit["datasets"]["deployment"][
        "solver_limit_rows"
    ]
    bos_limits = audit["datasets"]["bos"]["solver_limit_rows"]
    return (
        "# v0.7.0 Manuscript Result Claims\n\n"
        "All values below are derived from the v0.7.0 raw CSV files.\n\n"
        "## Abstract-safe quantitative sentence\n\n"
        "At $\\kappa=3$, after retaining within-workflow seeds where all four "
        "configurations meet the global deadline and averaging the four "
        "workflow means equally:\n\n"
        f"- {scheduler_sentences[0]}\n"
        f"- {scheduler_sentences[1]}\n\n"
        "## Scalability wording\n\n"
        f"The evaluation produced complete records through "
        f"{audit['deployment_maximum']['fog_nodes']} fog nodes, "
        f"{audit['deployment_maximum']['workflow_vnfs']:,} VNFs, and a BoS "
        f"width of {audit['bos_maximum_width']}. It recorded "
        f"{deployment_limits} deployment and {bos_limits} controlled-BoS "
        "rows with at least one MILP solver-limit status; therefore the paper "
        "must not claim that no solver limit was reached.\n\n"
        "## Deadline-comparison caution\n\n"
        "Use the full paired curve-level analysis rather than a selected "
        "point to claim HDS superiority. The JSON report records both the "
        "largest HDS advantage and disadvantage over the original "
        "$\\kappa\\leq3$ range.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deadline",
        type=Path,
        default=(
            PROJECT
            / "results/raw/deadline_curves_paper_aligned_50seeds.csv"
        ),
    )
    parser.add_argument(
        "--deployment",
        type=Path,
        default=PROJECT / "results/raw/deployment_scaling_paper_aligned.csv",
    )
    parser.add_argument(
        "--bos",
        type=Path,
        default=PROJECT / "results/raw/bos_scaling_paper_aligned.csv",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=PROJECT / "results/PAPER_CLAIMS_V0_7.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=PROJECT / "results/PAPER_CLAIMS_V0_7.md",
    )
    parser.add_argument("--deadline-factor", type=float, default=3.0)
    args = parser.parse_args()

    deadline = pd.read_csv(args.deadline)
    deployment = pd.read_csv(args.deployment)
    bos = pd.read_csv(args.bos)
    if set(deadline.configuration) != set(CONFIGS):
        raise SystemExit("Deadline configuration matrix is incomplete")
    payload = {
        "release": "v0.7.0",
        "sharing_effects": sharing_effects(
            deadline,
            deadline_factor=args.deadline_factor,
        ),
        "deadline_success_differences": success_differences(deadline),
        "audit": dataset_audit(deadline, deployment, bos),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.write_text(
        manuscript_markdown(payload),
        encoding="utf-8",
    )
    print(f"Wrote {args.json_output} and {args.markdown_output}")


if __name__ == "__main__":
    main()
