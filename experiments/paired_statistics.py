#!/usr/bin/env python3
"""Paired HDS-versus-PBQS inference with uncertainty and effect sizes."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, t, ttest_rel, wilcoxon


DEFAULT_METRICS = (
    "global_deadline_success",
    "sfc_subdeadline_success",
    "accepted_sfc_ratio",
    "sfc_deadline_miss_rate",
    "provisioning_cost",
    "makespan_s",
    "network_data_mb",
    "cpu_utilization",
    "ram_utilization",
    "scheduler_runtime_s",
)

CONDITIONAL_ON_GLOBAL_SUCCESS = {
    "provisioning_cost",
    "makespan_s",
    "network_data_mb",
    "cpu_utilization",
    "ram_utilization",
}

HIGHER_IS_BETTER = {
    "global_deadline_success",
    "sfc_subdeadline_success",
    "accepted_sfc_ratio",
    "cpu_utilization",
    "ram_utilization",
}


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm family-wise error correction, preserving NaNs."""

    adjusted = [math.nan] * len(p_values)
    finite = [
        (index, value)
        for index, value in enumerate(p_values)
        if math.isfinite(value)
    ]
    ordered = sorted(finite, key=lambda item: item[1])
    running = 0.0
    count = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * value)
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_interval(differences: np.ndarray) -> tuple[float, float]:
    if len(differences) < 2:
        value = float(differences[0]) if len(differences) else math.nan
        return value, value
    mean = float(np.mean(differences))
    standard_error = float(np.std(differences, ddof=1) / math.sqrt(len(differences)))
    margin = float(t.ppf(0.975, len(differences) - 1)) * standard_error
    return mean - margin, mean + margin


def analyze(
    frame: pd.DataFrame,
    *,
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    required = {
        "family",
        "workflow_size",
        "deadline_factor",
        "seed",
        "scheduler",
        "sharing",
        "global_deadline_success",
        *metrics,
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    selected = frame[
        frame["scheduler"].astype(str).str.upper().isin({"HDS", "PBQS"})
    ].copy()
    group_columns = [
        "family",
        "workflow_size",
        "deadline_factor",
        "sharing",
    ]
    for optional in (
        "deadline_mode",
        "candidate_count",
        "topology",
    ):
        if optional in selected:
            group_columns.append(optional)

    rows = []
    for group_key, group in selected.groupby(
        group_columns,
        dropna=False,
        sort=True,
    ):
        identity = dict(zip(group_columns, group_key))
        for metric in metrics:
            columns = list(
                dict.fromkeys(
                    [
                        "seed",
                        "scheduler",
                        metric,
                        "global_deadline_success",
                    ]
                )
            )
            subset = group[columns].copy()
            values = subset.pivot(index="seed", columns="scheduler", values=metric)
            success = subset.pivot(
                index="seed",
                columns="scheduler",
                values="global_deadline_success",
            )
            if not {"HDS", "PBQS"}.issubset(values.columns):
                continue
            pairs = values[["HDS", "PBQS"]].dropna()
            total_pairs = len(pairs)
            if metric in CONDITIONAL_ON_GLOBAL_SUCCESS:
                common = success[
                    (success.get("HDS") == 1)
                    & (success.get("PBQS") == 1)
                ].index
                pairs = pairs.loc[pairs.index.intersection(common)]
            hds = pairs["HDS"].to_numpy(dtype=float)
            pbqs = pairs["PBQS"].to_numpy(dtype=float)
            finite = np.isfinite(hds) & np.isfinite(pbqs)
            hds, pbqs = hds[finite], pbqs[finite]
            differences = hds - pbqs
            low, high = paired_interval(differences)
            if metric in {
                "global_deadline_success",
                "sfc_subdeadline_success",
                "workflow_deadline_success",
            }:
                hds_wins = int(np.sum((hds == 1) & (pbqs == 0)))
                pbqs_wins = int(np.sum((hds == 0) & (pbqs == 1)))
                discordant = hds_wins + pbqs_wins
                p_value = (
                    float(
                        binomtest(
                            min(hds_wins, pbqs_wins),
                            discordant,
                            0.5,
                            alternative="two-sided",
                        ).pvalue
                    )
                    if discordant
                    else 1.0
                )
                test_name = "exact_mcnemar"
            else:
                p_value = (
                    float(ttest_rel(hds, pbqs).pvalue)
                    if len(differences) >= 2
                    else math.nan
                )
                test_name = "paired_t"
            nonzero = differences[np.abs(differences) > 1e-12]
            wilcoxon_p = (
                float(wilcoxon(nonzero).pvalue)
                if len(nonzero) >= 2
                else math.nan
            )
            standard_deviation = (
                float(np.std(differences, ddof=1))
                if len(differences) >= 2
                else math.nan
            )
            effect = (
                float(np.mean(differences) / standard_deviation)
                if standard_deviation > 0
                else 0.0
            )
            pbqs_mean = float(np.mean(pbqs)) if len(pbqs) else math.nan
            hds_mean = float(np.mean(hds)) if len(hds) else math.nan
            if len(hds) and abs(pbqs_mean) > 1e-12:
                if metric in HIGHER_IS_BETTER:
                    improvement = 100.0 * (hds_mean - pbqs_mean) / abs(
                        pbqs_mean
                    )
                else:
                    improvement = 100.0 * (pbqs_mean - hds_mean) / abs(
                        pbqs_mean
                    )
            else:
                improvement = math.nan
            rows.append(
                {
                    **identity,
                    "metric": metric,
                    "comparison": "HDS-PBQS",
                    "difference_direction": (
                        "negative_favors_HDS"
                        if metric not in HIGHER_IS_BETTER
                        else "positive_favors_HDS"
                    ),
                    "total_paired_seeds": total_pairs,
                    "analyzed_paired_seeds": len(differences),
                    "hds_mean": hds_mean,
                    "pbqs_mean": pbqs_mean,
                    "mean_paired_difference": (
                        float(np.mean(differences))
                        if len(differences)
                        else math.nan
                    ),
                    "median_paired_difference": (
                        float(np.median(differences))
                        if len(differences)
                        else math.nan
                    ),
                    "paired_ci95_low": low,
                    "paired_ci95_high": high,
                    "cohen_dz": effect,
                    "hds_improvement_pct": improvement,
                    "primary_test": test_name,
                    "p_value": p_value,
                    "wilcoxon_p_value": wilcoxon_p,
                    "conditional_on_both_globally_successful": int(
                        metric in CONDITIONAL_ON_GLOBAL_SUCCESS
                    ),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["holm_adjusted_p_value_global"] = holm_adjust(
            result["p_value"].astype(float).tolist()
        )
        result["holm_adjusted_p_value_within_metric"] = result.groupby(
            "metric",
            dropna=False,
        )["p_value"].transform(
            lambda series: holm_adjust(series.astype(float).tolist())
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
    )
    args = parser.parse_args()
    metrics = tuple(
        item.strip() for item in args.metrics.split(",") if item.strip()
    )
    result = analyze(pd.read_csv(args.input), metrics=metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} paired inference rows to {args.output}")


if __name__ == "__main__":
    main()
