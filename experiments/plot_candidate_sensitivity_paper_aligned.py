#!/usr/bin/env python3
"""Plot the measured candidate-cap sensitivity supplement."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

try:
    from .plot_four_publication_figures_paper_aligned import (
        CONFIGS,
        STYLES,
        mean_ci95,
        plot_curve,
        style,
        wilson_interval,
    )
except ImportError:  # Direct execution: python experiments/<script>.py
    from plot_four_publication_figures_paper_aligned import (
        CONFIGS,
        STYLES,
        mean_ci95,
        plot_curve,
        style,
        wilson_interval,
    )


PROJECT = Path(__file__).resolve().parents[1]
INPUT = PROJECT / "results/raw/candidate_sensitivity_paper_aligned.csv"
OUTPUT = (
    PROJECT
    / "results/figures_supplementary/candidate_sensitivity_paper_aligned.png"
)


def metric_curve(
    data: pd.DataFrame,
    configuration: str,
    metric: str,
    *,
    binary: bool = False,
) -> pd.DataFrame:
    rows: list[tuple[int, float, float, float]] = []
    selected = data[data.configuration.eq(configuration)]
    for requested, group in selected.groupby("candidate_count", sort=True):
        if binary:
            mean = float(group[metric].mean())
            low, high = wilson_interval(group[metric])
        else:
            mean, low, high, _ = mean_ci95(group[metric])
        rows.append((int(requested), mean, low, high))
    return pd.DataFrame(
        rows,
        columns=("requested", "mean", "low", "high"),
    )


def plot_metric(
    ax: plt.Axes,
    data: pd.DataFrame,
    metric: str,
    ylabel: str,
    *,
    percent: bool = False,
    binary: bool = False,
    logarithmic: bool = False,
) -> None:
    scale = 100.0 if percent else 1.0
    for configuration in CONFIGS:
        curve = metric_curve(
            data,
            configuration,
            metric,
            binary=binary,
        )
        x = curve.requested.to_numpy(dtype=float)
        plot_curve(
            ax,
            x,
            scale * curve["mean"].to_numpy(dtype=float),
            configuration,
        )
        visual = STYLES[configuration]
        ax.fill_between(
            x,
            scale * curve.low.to_numpy(dtype=float),
            scale * curve.high.to_numpy(dtype=float),
            color=visual["color"],
            alpha=0.05,
            linewidth=0,
        )
    ax.axvline(5, color="#666666", linestyle=":", linewidth=1.0)
    ax.set_xticks(np.arange(1, 21, 2))
    ax.set_xlabel("Requested candidate cap")
    ax.set_ylabel(ylabel)
    if percent:
        ax.set_ylim(-3, 103)
    if logarithmic:
        ax.set_yscale("log")


def main() -> None:
    style()
    data = pd.read_csv(INPUT)
    required = {
        "candidate_count",
        "initial_effective_candidate_count",
        "configuration",
        "accepted_sfc_ratio",
        "global_deadline_success",
        "scheduler_runtime_s",
    }
    missing = required - set(data.columns)
    if missing:
        raise SystemExit(f"Candidate input missing columns: {sorted(missing)}")
    if set(data.candidate_count) != set(range(1, 21)):
        raise SystemExit("Candidate study must contain requested caps 1-20")
    if set(data.configuration) != set(CONFIGS):
        raise SystemExit("Candidate study must contain all four configurations")

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.8))

    effective = (
        data.groupby("candidate_count", sort=True)[
            "initial_effective_candidate_count"
        ]
        .median()
        .reset_index()
    )
    axes[0, 0].plot(
        effective.candidate_count,
        effective.initial_effective_candidate_count,
        color="#333333",
        marker="o",
        label="Effective unique slots",
    )
    axes[0, 0].plot(
        range(1, 21),
        range(1, 21),
        color="#999999",
        linestyle="--",
        label="Requested = effective",
    )
    axes[0, 0].axvline(5, color="#666666", linestyle=":", linewidth=1.0)
    axes[0, 0].set_xlabel("Requested candidate cap")
    axes[0, 0].set_ylabel("Initial unique feasible slots")
    axes[0, 0].set_xticks(np.arange(1, 21, 2))
    axes[0, 0].set_yticks(np.arange(1, 21, 2))
    axes[0, 0].legend(frameon=True)

    plot_metric(
        axes[0, 1],
        data,
        "accepted_sfc_ratio",
        "Accepted SFCs (%)",
        percent=True,
    )
    plot_metric(
        axes[1, 0],
        data,
        "global_deadline_success",
        "Globally successful workflows (%)",
        percent=True,
        binary=True,
    )
    plot_metric(
        axes[1, 1],
        data,
        "scheduler_runtime_s",
        "Scheduling runtime (s)",
        logarithmic=True,
    )

    for ax, title in zip(
        axes.flat,
        (
            "(a) Requested versus effective candidates",
            "(b) Accepted-SFC ratio",
            "(c) Global success",
            "(d) Scheduling runtime",
        ),
    ):
        ax.set_title(title)

    handles = [
        Line2D(
            [0],
            [0],
            color=STYLES[item]["color"],
            marker=STYLES[item]["marker"],
            linestyle=STYLES[item]["linestyle"],
            label=item,
        )
        for item in CONFIGS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=True)
    fig.suptitle(
        "Candidate-cap sensitivity at BoS width 5 and "
        r"$\kappa=5$",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.045,
        "Markers are measured values over ten paired seeds. The base "
        "topology exposes five initial unique node-VM slots.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp.png")
    fig.savefig(temporary, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    plt.imread(temporary)
    temporary.replace(OUTPUT)
    print(f"Wrote measured candidate-sensitivity figure to {OUTPUT}")


if __name__ == "__main__":
    main()
