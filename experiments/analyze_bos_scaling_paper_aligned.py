#!/usr/bin/env python3
"""Aggregate and pair the controlled BoS stress experiment."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t, ttest_rel

PROJECT = Path(__file__).resolve().parents[1]


METRICS = (
    "workflow_deadline_success",
    "global_deadline_success",
    "first_round_accepted_sfc_ratio",
    "provisioning_cost",
    "makespan_s",
    "network_data_mb",
    "scheduler_runtime_s",
    "first_round_scheduler_runtime_s",
    "milp_runtime_s",
)

CONDITIONAL_ON_GLOBAL_SUCCESS = {
    "provisioning_cost",
    "makespan_s",
    "network_data_mb",
}

HIGHER_IS_BETTER = {
    "workflow_deadline_success",
    "global_deadline_success",
    "first_round_accepted_sfc_ratio",
}


def mean_interval(values: pd.Series) -> tuple[float, float, float]:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if clean.empty:
        return math.nan, math.nan, math.nan
    mean = float(clean.mean())
    if len(clean) < 2:
        return mean, mean, mean
    margin = float(t.ppf(0.975, len(clean) - 1)) * float(
        clean.std(ddof=1) / math.sqrt(len(clean))
    )
    return mean, mean - margin, mean + margin


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "controlled_bos_size",
        "deadline_factor",
        "variant",
        "sharing",
        "candidate_count",
    ]
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(group_columns, dropna=False, sort=True):
        row = dict(zip(group_columns, key))
        row["n"] = len(group)
        for metric in METRICS:
            values = group[metric]
            if metric in CONDITIONAL_ON_GLOBAL_SUCCESS:
                values = values[group["global_deadline_success"].eq(1)]
            mean, low, high = mean_interval(values)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
            row[f"{metric}_median"] = float(
                values.replace([np.inf, -np.inf], np.nan).median()
            )
        for metric in (
            "milp_invocations",
            "milp_max_variables",
            "milp_max_constraints",
            "milp_max_single_solve_s",
            "milp_limit_reached_count",
            "jointly_scheduled_sfc_ratio",
        ):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_max"] = float(group[metric].max())
        rows.append(row)
    return pd.DataFrame(rows)


def paired(frame: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "controlled_bos_size",
        "deadline_factor",
        "sharing",
        "candidate_count",
    ]
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(group_columns, dropna=False, sort=True):
        identity = dict(zip(group_columns, key))
        for comparator in sorted(set(group["variant"]) - {"hds_full"}):
            pair_group = group[
                group["variant"].isin({"hds_full", comparator})
            ]
            success = pair_group.pivot(
                index="seed",
                columns="variant",
                values="global_deadline_success",
            )
            for metric in METRICS:
                values = pair_group.pivot(
                    index="seed",
                    columns="variant",
                    values=metric,
                )
                if not {"hds_full", comparator}.issubset(values.columns):
                    continue
                pairs = values[["hds_full", comparator]].replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                if metric in CONDITIONAL_ON_GLOBAL_SUCCESS:
                    successful = success[
                        success["hds_full"].eq(1)
                        & success[comparator].eq(1)
                    ].index
                    pairs = pairs.loc[pairs.index.intersection(successful)]
                pairs = pairs.dropna()
                differences = (
                    pairs["hds_full"] - pairs[comparator]
                ).to_numpy(dtype=float)
                if len(differences):
                    mean = float(np.mean(differences))
                    median = float(np.median(differences))
                else:
                    mean = median = math.nan
                if len(differences) >= 2:
                    standard_deviation = float(
                        np.std(differences, ddof=1)
                    )
                    standard_error = (
                        standard_deviation / math.sqrt(len(differences))
                    )
                    margin = float(t.ppf(0.975, len(differences) - 1))
                    low = mean - margin * standard_error
                    high = mean + margin * standard_error
                    if standard_deviation <= 1e-12:
                        p_value = 1.0 if abs(mean) <= 1e-12 else 0.0
                        effect = math.nan
                    else:
                        p_value = float(
                            ttest_rel(
                                pairs["hds_full"],
                                pairs[comparator],
                            ).pvalue
                        )
                        effect = mean / standard_deviation
                else:
                    low = high = mean
                    p_value = math.nan
                    effect = 0.0
                comparator_mean = float(pairs[comparator].mean())
                hds_mean = float(pairs["hds_full"].mean())
                if abs(comparator_mean) > 1e-12:
                    if metric in HIGHER_IS_BETTER:
                        improvement = (
                            100.0
                            * (hds_mean - comparator_mean)
                            / abs(comparator_mean)
                        )
                    else:
                        improvement = (
                            100.0
                            * (comparator_mean - hds_mean)
                            / abs(comparator_mean)
                        )
                else:
                    improvement = math.nan
                rows.append(
                    {
                        **identity,
                        "comparison": f"hds_full-{comparator}",
                        "metric": metric,
                        "paired_seeds": len(differences),
                        "hds_mean": hds_mean,
                        "comparator_mean": comparator_mean,
                        "mean_paired_difference": mean,
                        "median_paired_difference": median,
                        "paired_ci95_low": low,
                        "paired_ci95_high": high,
                        "cohen_dz": effect,
                        "paired_t_p_value": p_value,
                        "hds_improvement_pct": improvement,
                        "conditional_on_both_globally_successful": int(
                            metric in CONDITIONAL_ON_GLOBAL_SUCCESS
                        ),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT / "results/raw/bos_scaling_paper_aligned.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=(
            PROJECT / "results/processed/bos_scaling_paper_aligned_summary.csv"
        ),
    )
    parser.add_argument(
        "--paired-output",
        type=Path,
        default=(
            PROJECT / "results/processed/bos_scaling_paper_aligned_paired.csv"
        ),
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    summary = summarize(frame)
    comparisons = paired(frame)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    comparisons.to_csv(args.paired_output, index=False)
    print(
        f"Wrote {len(summary)} summary and {len(comparisons)} paired rows"
    )


if __name__ == "__main__":
    main()
