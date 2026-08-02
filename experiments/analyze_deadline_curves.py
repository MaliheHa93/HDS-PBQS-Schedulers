#!/usr/bin/env python3
"""Curve-level paired inference for the original paper deadline range."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

try:
    from .paired_statistics import holm_adjust
except ImportError:  # Direct execution: python experiments/<script>.py
    from paired_statistics import holm_adjust


PROJECT = Path(__file__).resolve().parents[1]
ORIGINAL_FACTORS = (
    0.8,
    0.9,
    1.0,
    1.05,
    1.1,
    1.15,
    1.2,
    1.25,
    1.3,
    1.35,
    1.4,
    1.45,
    1.5,
    1.6,
    1.7,
    1.8,
    1.9,
    2.0,
    2.25,
    2.5,
    2.75,
    3.0,
)


def normalized_auc(values: pd.Series) -> float:
    """Return trapezoidal AUC divided by the evaluated kappa width."""

    x = np.asarray(ORIGINAL_FACTORS, dtype=float)
    y = values.reindex(ORIGINAL_FACTORS).to_numpy(dtype=float)
    if not np.isfinite(y).all():
        return math.nan
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def bootstrap_interval(
    differences: np.ndarray,
    *,
    repetitions: int = 10_000,
    seed: int = 20260729,
) -> tuple[float, float]:
    if not len(differences):
        return math.nan, math.nan
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        len(differences),
        size=(repetitions, len(differences)),
    )
    means = differences[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def curve_inference(data: pd.DataFrame) -> pd.DataFrame:
    original = data[data.deadline_factor.isin(ORIGINAL_FACTORS)].copy()
    observed = set(original.deadline_factor.unique())
    if observed != set(ORIGINAL_FACTORS):
        raise ValueError(
            "Original deadline grid mismatch: "
            f"observed {sorted(observed)}"
        )
    rates = (
        original.groupby(
            ["family", "sharing", "scheduler", "seed", "deadline_factor"],
            sort=True,
        )
        .global_deadline_success.mean()
        .unstack("deadline_factor")
    )
    rows: list[dict[str, float | int | str]] = []
    for (family, sharing), group in rates.groupby(
        level=["family", "sharing"],
        sort=True,
    ):
        auc_by_scheduler: dict[str, pd.Series] = {}
        for scheduler in ("HDS", "PBQS"):
            scheduler_rows = group.xs(
                scheduler,
                level="scheduler",
            )
            auc_by_scheduler[scheduler] = scheduler_rows.apply(
                normalized_auc,
                axis=1,
            )
        paired = pd.concat(auc_by_scheduler, axis=1).dropna()
        differences = (
            paired["HDS"] - paired["PBQS"]
        ).to_numpy(dtype=float)
        low, high = bootstrap_interval(differences)
        nonzero = differences[np.abs(differences) > 1e-12]
        rows.append(
            {
                "family": family,
                "sharing": sharing,
                "kappa_min": ORIGINAL_FACTORS[0],
                "kappa_max": ORIGINAL_FACTORS[-1],
                "paired_seeds": len(paired),
                "hds_normalized_auc": paired["HDS"].mean(),
                "pbqs_normalized_auc": paired["PBQS"].mean(),
                "auc_difference_percentage_points": (
                    100.0 * differences.mean()
                ),
                "auc_difference_ci95_low_pp": 100.0 * low,
                "auc_difference_ci95_high_pp": 100.0 * high,
                "paired_t_p_value": (
                    1.0
                    if len(differences)
                    and np.all(np.abs(differences) <= 1e-12)
                    else (
                        float(
                            ttest_rel(
                                paired["HDS"],
                                paired["PBQS"],
                            ).pvalue
                        )
                        if len(paired) >= 2
                        else math.nan
                    )
                ),
                "wilcoxon_p_value": (
                    float(wilcoxon(nonzero).pvalue)
                    if len(nonzero) >= 2
                    else (1.0 if len(differences) else math.nan)
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["holm_adjusted_p_value"] = holm_adjust(
        result.paired_t_p_value.astype(float).tolist()
    )
    return result


def sustained_kappa_90(data: pd.DataFrame) -> pd.DataFrame:
    original = data[data.deadline_factor.isin(ORIGINAL_FACTORS)]
    rates = (
        original.groupby(
            ["family", "configuration", "deadline_factor"],
            sort=True,
        )
        .global_deadline_success.mean()
        .reset_index()
    )
    rows: list[dict[str, float | str]] = []
    for (family, configuration), group in rates.groupby(
        ["family", "configuration"],
        sort=True,
    ):
        ordered = group.set_index("deadline_factor").reindex(
            ORIGINAL_FACTORS
        )
        values = ordered.global_deadline_success.to_numpy(dtype=float)
        threshold = math.nan
        for index, factor in enumerate(ORIGINAL_FACTORS):
            if np.all(values[index:] >= 0.9):
                threshold = factor
                break
        rows.append(
            {
                "family": family,
                "configuration": configuration,
                "sustained_kappa_90": threshold,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            PROJECT
            / "results/raw/deadline_curves_paper_aligned_50seeds.csv"
        ),
    )
    parser.add_argument(
        "--auc-output",
        type=Path,
        default=(
            PROJECT
            / "results/processed/deadline_curve_level_auc.csv"
        ),
    )
    parser.add_argument(
        "--kappa90-output",
        type=Path,
        default=PROJECT / "results/processed/deadline_kappa90.csv",
    )
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    required = {
        "family",
        "sharing",
        "scheduler",
        "configuration",
        "seed",
        "deadline_factor",
        "global_deadline_success",
    }
    missing = required - set(data)
    if missing:
        raise SystemExit(f"Input is missing columns: {sorted(missing)}")
    if data.seed.nunique() != 50:
        raise SystemExit("Curve-level analysis requires 50 paired seeds")
    auc = curve_inference(data)
    kappa90 = sustained_kappa_90(data)
    args.auc_output.parent.mkdir(parents=True, exist_ok=True)
    auc.to_csv(args.auc_output, index=False)
    kappa90.to_csv(args.kappa90_output, index=False)
    print(f"Wrote {len(auc)} curve-level comparisons to {args.auc_output}")
    print(f"Wrote {len(kappa90)} sustained-kappa rows to {args.kappa90_output}")


if __name__ == "__main__":
    main()
