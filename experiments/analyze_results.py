#!/usr/bin/env python3
"""Aggregate raw results with standard deviations and 95% confidence intervals."""

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
    "reference_deadline_feasible",
    "deadline_kappa",
    "unfinished_sfc_count",
    "completed_sfc_count",
    "total_sfc_count",
    "accepted_sfc_ratio",
    "first_round_submitted_sfc_count",
    "first_round_admitted_sfc_count",
    "first_round_accepted_sfc_ratio",
    "first_round_scheduler_runtime_s",
    "realized_workflow_slack_s",
    "workflow_deadline_consumption",
    "scheduling_round_count",
    "bos_max_size",
    "bos_mean_size",
    "jointly_scheduled_sfc_count",
    "jointly_scheduled_sfc_ratio",
    "single_sfc_fallback_count",
    "reuse_path_count",
    "new_vm_path_count",
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
    "milp_avoidable_idle_removed_s",
    "milp_max_avoidable_idle_removed_s",
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

SUCCESS_CONDITIONAL_METRICS = {
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
    "makespan_s",
    "vm_reuse_rate",
    "provisioned_vm_count",
    "communication_delay_s",
    "propagation_delay_s",
    "serialization_delay_s",
    "queueing_delay_s",
    "execution_time_s",
    "measurement_duration_s",
    "used_link_count",
    "realized_workflow_slack_s",
    "workflow_deadline_consumption",
}


def paired_changes(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate paired HDS-vs-PBQS changes for matched seeds."""

    identity = [
        "family",
        "workflow_size",
        "deadline_factor",
        "seed",
        "sharing",
    ]
    identity += [
        column
        for column in (
            "candidate_count",
            "eta",
            "alpha",
            "beta",
            "gamma",
            "omega_u",
            "omega_w",
            "deadline_mode",
            "reconstruct_earliest_start",
            "study",
            "topology",
            "node_count",
        )
        if column in data
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
    schedulers = set(data["scheduler"].astype(str).str.upper())
    if not {"HDS", "PBQS"}.issubset(schedulers):
        return pd.DataFrame()
    hds = data[data["scheduler"].str.upper() == "HDS"][
        identity + comparison_metrics
    ].rename(columns={metric: f"{metric}_hds" for metric in comparison_metrics})
    pbqs = data[data["scheduler"].str.upper() == "PBQS"][
        identity + comparison_metrics
    ].rename(columns={metric: f"{metric}_pbqs" for metric in comparison_metrics})
    records = hds.merge(pbqs, on=identity, how="inner", validate="one_to_one")

    for metric in comparison_metrics:
        hds_values = pd.to_numeric(records[f"{metric}_hds"], errors="coerce")
        pbqs_values = pd.to_numeric(records[f"{metric}_pbqs"], errors="coerce")
        both_successful = (
            pd.to_numeric(
                records["global_deadline_success_hds"],
                errors="coerce",
            )
            == 1
        ) & (
            pd.to_numeric(
                records["global_deadline_success_pbqs"],
                errors="coerce",
            )
            == 1
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

    group_columns = [column for column in identity if column != "seed"]
    change_columns = [
        column for column in records if column.endswith("_improvement_pct")
    ]
    rows: list[dict] = []
    for key, frame in records.groupby(group_columns, sort=True, dropna=False):
        row = dict(zip(group_columns, key))
        row["paired_sample_size"] = len(frame)
        for column in change_columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            low, high = confidence_interval_95(values.astype(float).tolist())
            row[f"{column}_mean"] = values.mean()
            row[f"{column}_ci95_low"] = low
            row[f"{column}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            PROJECT / "results/raw/deadline_curves_paper_aligned_50seeds.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT
            / "results/processed/deadline_curves_paper_aligned_summary.csv"
        ),
    )
    parser.add_argument(
        "--comparisons-output",
        type=Path,
        default=(
            PROJECT
            / "results/processed/deadline_curves_paper_aligned_paired.csv"
        ),
    )
    parser.add_argument(
        "--normalized-output",
        type=Path,
        default=(
            PROJECT
            / "results/processed/deadline_curves_paper_aligned_normalized.csv"
        ),
    )
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    data = add_normalized_columns(
        data,
        group_columns=["family", "workflow_size"],
        deadline_group_columns=["family", "workflow_size", "seed"],
    )
    required = {
        "family",
        "workflow_size",
        "deadline_factor",
        "configuration",
        "scheduler",
        "sharing",
        "seed",
    } | set(METRICS)
    missing = required - set(data)
    if missing:
        raise SystemExit(f"Input is missing columns: {sorted(missing)}")

    groups = [
        "family",
        "workflow_size",
        "deadline_factor",
        "configuration",
    ]
    groups += [
        column
        for column in (
            "deadline_mode",
            "reuse_policy",
            "study",
            "topology",
            "node_count",
            "candidate_count",
        )
        if column in data
    ]
    rows: list[dict] = []
    for key, frame in data.groupby(groups, sort=True):
        row = dict(zip(groups, key))
        row["sample_size"] = len(frame)
        for metric in METRICS:
            raw_values = pd.to_numeric(
                frame[metric],
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan)
            if metric in SUCCESS_CONDITIONAL_METRICS:
                finite_unconditional = raw_values.dropna()
                row[f"{metric}_unconditional_mean"] = (
                    finite_unconditional.mean()
                )
                row[f"{metric}_completed_sample_size"] = int(
                    (
                        frame["workflow_completed"].eq(1)
                        & raw_values.notna()
                    ).sum()
                )
                row[f"{metric}_successful_sample_size"] = int(
                    (
                        frame["global_deadline_success"].eq(1)
                        & raw_values.notna()
                    ).sum()
                )
                values = raw_values[
                    frame["global_deadline_success"].eq(1)
                ].dropna()
            else:
                values = raw_values.dropna()
            mean = values.mean()
            standard_deviation = values.std(ddof=1) if len(values) > 1 else 0.0
            low, high = confidence_interval_95(values.astype(float).tolist())
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = standard_deviation
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.normalized_output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.normalized_output, index=False)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    comparisons = paired_changes(data)
    comparisons.to_csv(args.comparisons_output, index=False)
    print(f"Wrote {len(rows)} aggregate rows to {args.output}")
    print(f"Wrote normalized raw rows to {args.normalized_output}")
    print(
        f"Wrote {len(comparisons)} paired comparison rows to "
        f"{args.comparisons_output}"
    )


if __name__ == "__main__":
    main()
