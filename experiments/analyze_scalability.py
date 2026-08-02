#!/usr/bin/env python3
"""Aggregate deployment-scaling results and paired HDS-vs-PBQS changes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from hds.metrics import confidence_interval_95  # noqa: E402
from hds.normalization import add_normalized_columns  # noqa: E402

METRICS = [
    "minimum_makespan_s",
    "workflow_slack_s",
    "workflow_deadline_s",
    "workflow_deadline_duration_s",
    "deadline_to_minimum_makespan_ratio",
    "unfinished_sfc_count",
    "completed_sfc_count",
    "total_sfc_count",
    "first_round_submitted_sfc_count",
    "first_round_admitted_sfc_count",
    "first_round_accepted_sfc_ratio",
    "first_round_scheduler_runtime_s",
    "milp_invocations",
    "provisioning_cost",
    "end_to_end_delay_s",
    "network_data_mb",
    "network_hop_data_mb",
    "resource_utilization",
    "cpu_utilization",
    "ram_utilization",
    "vm_time_utilization",
    "mean_node_cpu_utilization",
    "max_node_cpu_utilization",
    "mean_node_ram_utilization",
    "max_node_ram_utilization",
    "mean_link_bandwidth_utilization",
    "max_link_bandwidth_utilization",
    "workflow_completed",
    "global_deadline_success",
    "sfc_subdeadline_success",
    "workflow_deadline_success",
    "sfc_deadline_miss_rate",
    "makespan_s",
    "vm_reuse_rate",
    "provisioned_vm_count",
    "scheduler_runtime_s",
    "scheduler_overhead_ratio",
    "case_wall_runtime_s",
    "milp_runtime_s",
    "milp_max_single_solve_s",
    "milp_max_bos_size",
    "requested_candidate_count",
    "initial_effective_candidate_count",
    "milp_min_effective_candidate_count",
    "milp_max_effective_candidate_count",
    "milp_max_variables",
    "milp_max_constraints",
    "milp_max_gap",
    "milp_limit_reached_count",
    "communication_delay_s",
    "propagation_delay_s",
    "serialization_delay_s",
    "queueing_delay_s",
    "execution_time_s",
    "measurement_duration_s",
    "used_link_count",
    "provisioning_cost_normalized",
    "end_to_end_delay_s_normalized",
    "network_data_mb_normalized",
    "deadline_ratio",
    "deadline_normalized_0_1",
]
GROUPS = [
    "topology",
    "node_count",
    "local_node_count",
    "global_node_count",
    "family",
    "workflow_size",
    "deadline_factor",
    "candidate_count",
    "configuration",
]


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for key, frame in data.groupby(GROUPS, sort=True, dropna=False):
        row = dict(zip(GROUPS, key))
        row["sample_size"] = len(frame)
        for metric in METRICS:
            values = pd.to_numeric(frame[metric], errors="coerce")
            if metric in {
                "provisioning_cost",
                "end_to_end_delay_s",
                "network_data_mb",
                "network_hop_data_mb",
                "resource_utilization",
                "cpu_utilization",
                "ram_utilization",
                "vm_time_utilization",
                "mean_link_bandwidth_utilization",
                "max_link_bandwidth_utilization",
                "makespan_s",
            }:
                values = values[
                    frame["global_deadline_success"].eq(1)
                ]
            values = values.dropna()
            mean = values.mean()
            std = values.std(ddof=1) if len(values) > 1 else 0.0
            low, high = confidence_interval_95(values.astype(float).tolist())
            row[f"{metric}_mean"] = mean
            row[f"{metric}_median"] = values.median()
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def paired_changes(data: pd.DataFrame) -> pd.DataFrame:
    """Compare HDS and PBQS only within the same seed and sharing mode."""

    identity = [
        "topology",
        "node_count",
        "family",
        "workflow_size",
        "deadline_factor",
        "candidate_count",
        "seed",
        "sharing",
    ]
    comparison_metrics = [
        "provisioning_cost",
        "end_to_end_delay_s",
        "network_data_mb",
        "resource_utilization",
        "cpu_utilization",
        "ram_utilization",
        "max_link_bandwidth_utilization",
        "workflow_deadline_success",
        "workflow_completed",
        "global_deadline_success",
        "sfc_subdeadline_success",
    ]
    schedulers = set(data["scheduler"].astype(str))
    if not {"HDS", "PBQS"}.issubset(schedulers):
        return pd.DataFrame()
    hds = data[data["scheduler"] == "HDS"][identity + comparison_metrics].rename(
        columns={metric: f"{metric}_hds" for metric in comparison_metrics}
    )
    pbqs = data[data["scheduler"] == "PBQS"][
        identity + comparison_metrics
    ].rename(
        columns={metric: f"{metric}_pbqs" for metric in comparison_metrics}
    )
    records = hds.merge(pbqs, on=identity, how="inner", validate="one_to_one")
    both_successful = (
        records["global_deadline_success_hds"].eq(1)
        & records["global_deadline_success_pbqs"].eq(1)
    )
    for metric in comparison_metrics:
        hds_values = pd.to_numeric(
            records[f"{metric}_hds"], errors="coerce"
        )
        pbqs_values = pd.to_numeric(
            records[f"{metric}_pbqs"], errors="coerce"
        )
        if metric in {
            "provisioning_cost",
            "end_to_end_delay_s",
            "network_data_mb",
            "max_link_bandwidth_utilization",
        }:
            change = (
                100.0
                * (pbqs_values - hds_values)
                / pbqs_values.replace(0, np.nan)
            )
        elif metric in {
            "resource_utilization",
            "cpu_utilization",
            "ram_utilization",
        }:
            change = (
                100.0
                * (hds_values - pbqs_values)
                / pbqs_values.replace(0, np.nan)
            )
        else:
            change = 100.0 * (hds_values - pbqs_values)
        if metric not in {
            "workflow_deadline_success",
            "workflow_completed",
            "global_deadline_success",
            "sfc_subdeadline_success",
        }:
            change = change.where(both_successful)
        records[f"{metric}_improvement_pct"] = change.to_numpy()

    group_columns = [item for item in identity if item != "seed"]
    rows: list[dict] = []
    change_columns = [
        item for item in records if item.endswith("_improvement_pct")
    ]
    for key, frame in records.groupby(group_columns, sort=True, dropna=False):
        row = dict(zip(group_columns, key))
        row["paired_sample_size"] = len(frame)
        for column in change_columns:
            values_column = pd.to_numeric(
                frame[column], errors="coerce"
            ).dropna()
            mean = values_column.mean()
            std = (
                values_column.std(ddof=1) if len(values_column) > 1 else 0.0
            )
            low, high = confidence_interval_95(
                values_column.astype(float).tolist()
            )
            row[f"{column}_mean"] = mean
            row[f"{column}_ci95_low"] = low
            row[f"{column}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT / "results/raw/deployment_scaling_paper_aligned.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT
            / "results/processed/deployment_scaling_paper_aligned_summary.csv"
        ),
    )
    parser.add_argument(
        "--comparisons-output",
        type=Path,
        default=(
            PROJECT
            / "results/processed/deployment_scaling_paper_aligned_paired.csv"
        ),
    )
    parser.add_argument(
        "--normalized-output",
        type=Path,
        default=(
            PROJECT
            / "results/processed/deployment_scaling_paper_aligned_normalized.csv"
        ),
    )
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    data = add_normalized_columns(
        data,
        group_columns=["family", "topology"],
        deadline_group_columns=[
            "family",
            "topology",
            "workflow_size",
            "seed",
        ],
    )
    required = set(GROUPS + METRICS + ["seed", "scheduler", "sharing"])
    missing = required - set(data)
    if missing:
        raise SystemExit(f"Input is missing columns: {sorted(missing)}")

    summary = summarize(data)
    comparisons = paired_changes(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.normalized_output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.normalized_output, index=False)
    summary.to_csv(args.output, index=False)
    comparisons.to_csv(args.comparisons_output, index=False)
    print(f"Wrote {len(summary)} aggregate rows to {args.output}")
    print(f"Wrote normalized raw rows to {args.normalized_output}")
    print(
        f"Wrote {len(comparisons)} paired comparison rows to "
        f"{args.comparisons_output}"
    )


if __name__ == "__main__":
    main()
