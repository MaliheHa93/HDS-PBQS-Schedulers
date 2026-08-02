#!/usr/bin/env python3
"""Generate four line-based, objective-aligned publication figures.

Every marker is a measured experiment point.  Lines connect ordered
measurements only; the script deliberately avoids spline interpolation and
other visual smoothing that could imply unmeasured results.
"""

from __future__ import annotations

import math
from pathlib import Path
import shutil
import sys

import matplotlib

matplotlib.use("Agg")

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results/figures_paper_aligned"
PDF_OUT = PROJECT / "results/figures_paper_aligned_pdf"
MANUSCRIPT_OUT = PROJECT / "results/figures_for_manuscript"
FAMILIES = ("montage", "epigenomics", "inspiral", "cybershake")
DEADLINE_FAMILIES = ("epigenomics", "cybershake")
FAMILY_LABELS = {
    "montage": "Montage",
    "epigenomics": "Epigenomics",
    "inspiral": "Inspiral",
    "cybershake": "CyberShake",
}
CONFIGS = (
    "HDS-Sharable",
    "PBQS-Sharable",
    "HDS-NonSharable",
    "PBQS-NonSharable",
)
STYLES = {
    "HDS-Sharable": {
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
    },
    "PBQS-Sharable": {
        "color": "#D55E00",
        "marker": "s",
        "linestyle": "-",
    },
    "HDS-NonSharable": {
        "color": "#56B4E9",
        "marker": "^",
        "linestyle": "--",
    },
    "PBQS-NonSharable": {
        "color": "#E69F00",
        "marker": "D",
        "linestyle": "--",
    },
}
DEADLINE_TICKS = (
    0.8,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
)
EXPECTED_DEADLINE_FACTORS = {
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
    3.25,
    3.5,
    3.75,
    4.0,
    4.25,
    4.5,
    4.75,
    5.0,
}
MIN_COMMON_SUCCESSES = 5


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "figure.dpi": 120,
            "legend.fontsize": 8,
            "lines.linewidth": 1.55,
            "lines.markersize": 4.4,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PDF_OUT.mkdir(parents=True, exist_ok=True)
    final_path = OUT / f"{stem}.png"
    final_pdf = PDF_OUT / f"{stem}.pdf"
    temporary_path = OUT / f".{stem}.tmp.png"
    temporary_pdf = PDF_OUT / f".{stem}.tmp.pdf"
    fig.savefig(
        temporary_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        temporary_pdf,
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Creator": "HDS paper-aligned v0.7.0 renderer",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)
    # Decode every pixel before atomically publishing the image.  This keeps a
    # reader from observing a partially written PNG and catches truncated files.
    plt.imread(temporary_path)
    temporary_path.replace(final_path)
    temporary_pdf.replace(final_pdf)


def pair_identity(data: pd.DataFrame) -> list[str]:
    candidates = (
        "family",
        "workflow_size",
        "deadline_factor",
        "seed",
        "topology",
        "node_count",
        "controlled_bos_size",
        "candidate_count",
    )
    return [column for column in candidates if column in data.columns]


def common_global_success(data: pd.DataFrame) -> pd.DataFrame:
    """Keep identical cases where all four configurations succeed globally."""

    subset = data.copy()
    identity = pair_identity(subset)
    duplicates = subset.duplicated(identity + ["configuration"], keep=False)
    if bool(duplicates.any()):
        raise ValueError("Duplicate configuration rows prevent paired filtering")
    successful = subset[subset.global_deadline_success.eq(1)]
    counts = successful.groupby(identity, dropna=False)[
        "configuration"
    ].nunique()
    keys = counts[counts.eq(len(CONFIGS))].reset_index()[identity]
    retained = subset.merge(
        keys,
        on=identity,
        how="inner",
        validate="many_to_one",
    )
    return retained[retained.global_deadline_success.eq(1)].copy()


def mean_ci95(values: pd.Series) -> tuple[float, float, float, int]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    count = len(clean)
    if count == 0:
        return math.nan, math.nan, math.nan, 0
    mean = float(clean.mean())
    if count < 2:
        return mean, mean, mean, count
    margin = float(t.ppf(0.975, count - 1)) * float(
        clean.std(ddof=1) / math.sqrt(count)
    )
    return mean, mean - margin, mean + margin, count


def wilson_interval(successes: pd.Series) -> tuple[float, float]:
    """Return a two-sided 95% Wilson interval for binary outcomes."""

    clean = pd.to_numeric(successes, errors="coerce").dropna().astype(float)
    count = len(clean)
    if count == 0:
        return math.nan, math.nan
    proportion = float(clean.mean())
    z = 1.959963984540054
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / count
            + z * z / (4.0 * count * count)
        )
        / denominator
    )
    return center - margin, center + margin


def plot_curve(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    configuration: str,
    *,
    label: str | None = None,
    line_style: str | None = None,
    marker: str | None = None,
    marker_face: str | None = None,
    alpha: float = 1.0,
) -> None:
    visual = STYLES[configuration]
    ax.plot(
        x,
        y,
        color=visual["color"],
        marker=visual["marker"] if marker is None else marker,
        markerfacecolor=(
            visual["color"] if marker_face is None else marker_face
        ),
        markeredgecolor=visual["color"],
        linestyle=visual["linestyle"] if line_style is None else line_style,
        label=configuration if label is None else label,
        alpha=alpha,
        markeredgewidth=0.7,
    )


def configure_deadline_axis(
    ax: plt.Axes,
    *,
    lower: float = 0.8,
    upper: float = 5.0,
) -> None:
    ax.set_xlim(lower, upper)
    ticks = [value for value in DEADLINE_TICKS if lower <= value <= upper]
    if lower not in ticks:
        ticks.insert(0, lower)
    ax.set_xticks(ticks)
    ax.set_xlabel(r"Deadline factor $\kappa$")
    if upper > 3.0:
        ax.axvline(
            3.0,
            color="#6B6B6B",
            linestyle=(0, (4, 3)),
            linewidth=0.85,
            alpha=0.75,
            zorder=0,
        )


def readable_deadline_lower(first_point: float) -> float:
    if first_point >= 2.7:
        return 2.25
    if first_point >= 2.2:
        return 2.0
    if first_point >= 1.35:
        return 1.2
    if first_point >= 1.1:
        return 1.0
    return 0.8


def figure_deadlines(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.2, 3.55),
        sharex=True,
        sharey=True,
    )
    panel_titles = ("(a) Epigenomics", "(b) CyberShake")
    for column, (ax, family, panel_title) in enumerate(
        zip(axes.flat, DEADLINE_FAMILIES, panel_titles)
    ):
        part = data[
            data.family.eq(family)
            & data.deadline_factor.between(0.8, 3.0)
        ]
        for index, configuration in enumerate(CONFIGS):
            values = (
                part[part.configuration.eq(configuration)]
                .groupby("deadline_factor", sort=True)
                .agg(
                    global_rate=("global_deadline_success", "mean"),
                    sub_rate=("sfc_subdeadline_success", "mean"),
                    global_low=(
                        "global_deadline_success",
                        lambda item: wilson_interval(item)[0],
                    ),
                    global_high=(
                        "global_deadline_success",
                        lambda item: wilson_interval(item)[1],
                    ),
                )
                .reset_index()
            )
            x = values.deadline_factor.to_numpy(dtype=float)
            plot_curve(
                ax,
                x,
                100 * values.sub_rate.to_numpy(dtype=float),
                configuration,
                label="_nolegend_",
                line_style=":",
                marker="",
                alpha=0.48,
            )
            plot_curve(
                ax,
                x,
                100 * values.global_rate.to_numpy(dtype=float),
                configuration,
                line_style="-",
            )
            visual = STYLES[configuration]
            ax.fill_between(
                x,
                100 * values.global_low.to_numpy(dtype=float),
                100 * values.global_high.to_numpy(dtype=float),
                color=visual["color"],
                alpha=0.035,
                linewidth=0,
            )
        ax.set_title(panel_title)
        ax.set_ylim(-3, 103)
        if column == 0:
            ax.set_ylabel("Success rate (%)")
        ax.set_xlim(0.8, 3.0)
        ax.set_xticks((0.8, 1.0, 1.5, 2.0, 2.5, 3.0))
        ax.set_xlabel(r"Deadline factor $\kappa$")

    config_handles = [
        Line2D(
            [0],
            [0],
            color=STYLES[item]["color"],
            marker=STYLES[item]["marker"],
            linestyle="-",
            label=item,
        )
        for item in CONFIGS
    ]
    meaning_handles = [
        Line2D([0], [0], color="#333333", linestyle="-", label="Global deadline"),
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle=":",
            label="All SFC subdeadlines",
        ),
    ]
    axes[0].legend(
        config_handles + meaning_handles,
        [item.get_label() for item in config_handles + meaning_handles],
        loc="lower right",
        ncol=2,
        fontsize=7.2,
        columnspacing=1.0,
        handlelength=2.4,
        frameon=True,
    )
    fig.tight_layout()
    save(fig, "Figure3_deadline_success")


def paired_metric_curve(
    data: pd.DataFrame,
    family: str,
    configuration: str,
    metric: str,
) -> pd.DataFrame:
    part = data[data.family.eq(family)]
    paired = common_global_success(part)
    paired = paired[paired.configuration.eq(configuration)]
    rows: list[dict[str, float | int]] = []
    for deadline_factor, group in paired.groupby(
        "deadline_factor",
        sort=True,
    ):
        mean, low, high, count = mean_ci95(group[metric])
        if count < MIN_COMMON_SUCCESSES:
            continue
        rows.append(
            {
                "deadline_factor": float(deadline_factor),
                "mean": mean,
                "low": low,
                "high": high,
                "count": count,
            }
        )
    return pd.DataFrame(rows)


def figure_cost_delay(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 3.65),
        sharex=True,
    )
    metrics = (
        ("provisioning_cost", "Provisioning cost ($)"),
        ("end_to_end_delay_s", "End-to-end delay (s)"),
    )
    titles = (
        "(a) Macro-average provisioning cost",
        "(b) Macro-average end-to-end delay",
    )
    curves = {
        (metric, configuration): macro_metric_curve(
            data, configuration, metric
        )
        for metric, _ in metrics
        for configuration in CONFIGS
    }
    first_valid = min(
        float(curve.deadline_factor.min())
        for curve in curves.values()
        if not curve.empty
    )
    axis_lower = 2.0
    for ax, (metric, ylabel), title in zip(axes.flat, metrics, titles):
        for configuration in CONFIGS:
            values = curves[(metric, configuration)]
            if values.empty:
                continue
            x = values.deadline_factor.to_numpy(dtype=float)
            plot_curve(
                ax,
                x,
                values["mean"].to_numpy(dtype=float),
                configuration,
            )
            visual = STYLES[configuration]
            ax.fill_between(
                x,
                values.low.to_numpy(dtype=float),
                values.high.to_numpy(dtype=float),
                color=visual["color"],
                alpha=0.045,
                linewidth=0,
            )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        configure_deadline_axis(
            ax,
            lower=axis_lower,
            upper=float(data.deadline_factor.max()),
        )
        if first_valid > axis_lower + 1e-9:
            ax.axvspan(
                axis_lower,
                first_valid,
                color="#BDBDBD",
                alpha=0.12,
                linewidth=0,
            )

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
    axes[1].legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.02, 0.56),
        ncol=2,
        frameon=True,
    )
    fig.tight_layout()
    save(fig, "Figure4_macro_cost_delay")


def macro_metric_curve(
    data: pd.DataFrame,
    configuration: str,
    metric: str,
) -> pd.DataFrame:
    """Equal-family macro-mean over common global-success seed cases."""

    paired = common_global_success(data)
    paired = paired[paired.configuration.eq(configuration)]
    rows: list[dict[str, float]] = []
    for deadline_factor, at_deadline in paired.groupby(
        "deadline_factor",
        sort=True,
    ):
        family_means: list[float] = []
        mean_variances: list[float] = []
        counts: list[int] = []
        for family in FAMILIES:
            values = pd.to_numeric(
                at_deadline[at_deadline.family.eq(family)][metric],
                errors="coerce",
            ).dropna()
            count = len(values)
            if count < MIN_COMMON_SUCCESSES:
                break
            family_means.append(float(values.mean()))
            mean_variances.append(
                float(values.var(ddof=1)) / count if count > 1 else 0.0
            )
            counts.append(count)
        if len(family_means) != len(FAMILIES):
            continue
        mean = float(np.mean(family_means))
        margin = (
            1.959963984540054
            * math.sqrt(sum(mean_variances))
            / len(FAMILIES)
        )
        rows.append(
            {
                "deadline_factor": float(deadline_factor),
                "mean": mean,
                "low": mean - margin,
                "high": mean + margin,
                "count": min(counts),
            }
        )
    return pd.DataFrame(rows)


def figure_resources(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 3.65),
        sharex=True,
    )
    metrics = ("network_data_mb", "cpu_utilization", "ram_utilization")
    first_valid_points: list[float] = []
    for configuration in CONFIGS:
        values = macro_metric_curve(data, configuration, metrics[0])
        if not values.empty:
            first_valid_points.append(float(values.deadline_factor.min()))
    first_valid = min(first_valid_points)
    axis_lower = 2.0
    for configuration in CONFIGS:
        values = macro_metric_curve(data, configuration, "network_data_mb")
        plot_curve(
            axes[0],
            values.deadline_factor.to_numpy(dtype=float),
            values["mean"].to_numpy(dtype=float),
            configuration,
        )
        visual = STYLES[configuration]
        axes[0].fill_between(
            values.deadline_factor.to_numpy(dtype=float),
            values.low.to_numpy(dtype=float),
            values.high.to_numpy(dtype=float),
            color=visual["color"],
            alpha=0.045,
            linewidth=0,
        )
    axes[0].set_ylabel("Transferred data (MB)")
    axes[0].set_title("(a) Transferred data")

    for metric, marker_face in (
        ("cpu_utilization", None),
        ("ram_utilization", "white"),
    ):
        for configuration in CONFIGS:
            values = macro_metric_curve(data, configuration, metric)
            if values.empty:
                continue
            plot_curve(
                axes[1],
                values.deadline_factor.to_numpy(dtype=float),
                100.0 * values["mean"].to_numpy(dtype=float),
                configuration,
                label="_nolegend_",
                marker_face=marker_face,
                alpha=0.9 if metric == "ram_utilization" else 1.0,
            )
            visual = STYLES[configuration]
            axes[1].fill_between(
                values.deadline_factor.to_numpy(dtype=float),
                100.0 * values.low.to_numpy(dtype=float),
                100.0 * values.high.to_numpy(dtype=float),
                color=visual["color"],
                alpha=0.025,
                linewidth=0,
            )
    axes[1].set_ylabel("Purchased-capacity utilization (%)")
    axes[1].set_title("(b) CPU and RAM utilization")
    for ax in axes.flat:
        configure_deadline_axis(
            ax,
            lower=axis_lower,
            upper=float(data.deadline_factor.max()),
        )
        if first_valid > axis_lower + 1e-9:
            ax.axvspan(
                axis_lower,
                first_valid,
                color="#BDBDBD",
                alpha=0.12,
                linewidth=0,
            )

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
    metric_handles = [
        Line2D(
            [0], [0], color="#333333", marker="o",
            markerfacecolor="#333333", linestyle="", label="CPU (filled)"
        ),
        Line2D(
            [0], [0], color="#333333", marker="o",
            markerfacecolor="white", linestyle="", label="RAM (open)"
        ),
    ]
    axes[0].legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.55, 0.48),
        ncol=2,
        frameon=True,
    )
    axes[1].legend(
        handles=metric_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=2,
        frameon=True,
    )
    fig.tight_layout()
    save(fig, "Figure5_transfer_cpu_ram")


def median_iqr(
    values: pd.Series,
) -> tuple[float, float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if clean.empty:
        return math.nan, math.nan, math.nan
    return (
        float(clean.median()),
        float(clean.quantile(0.25)),
        float(clean.quantile(0.75)),
    )


def deployment_metric_curve(
    data: pd.DataFrame,
    configuration: str,
    metric: str,
) -> pd.DataFrame:
    selected = data[data.configuration.eq(configuration)]
    rows: list[dict[str, float]] = []
    for node_count, group in selected.groupby("node_count", sort=True):
        median, low, high = median_iqr(group[metric])
        workflow_size = int(group.workflow_size.iloc[0])
        rows.append(
            {
                "node_count": int(node_count),
                "workflow_size": workflow_size,
                "median": median,
                "low": low,
                "high": high,
            }
        )
    return pd.DataFrame(rows)


def deployment_ticks(data: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    profiles = (
        data[["node_count", "workflow_size"]]
        .drop_duplicates()
        .sort_values("node_count")
    )
    nodes = profiles.node_count.to_numpy(dtype=float)
    labels = [
        f"{int(node_count)}/{int(workflow_size)}"
        for node_count, workflow_size in profiles.itertuples(index=False)
    ]
    return nodes, labels


def plot_deployment_metric(
    ax: plt.Axes,
    data: pd.DataFrame,
    metric: str,
    ylabel: str,
    *,
    percent: bool = False,
    logarithmic: bool = False,
) -> None:
    scale = 100.0 if percent else 1.0
    for configuration in CONFIGS:
        values = deployment_metric_curve(data, configuration, metric)
        x = values.node_count.to_numpy(dtype=float)
        median = scale * values["median"].to_numpy(dtype=float)
        plot_curve(ax, x, median, configuration)
        visual = STYLES[configuration]
        ax.fill_between(
            x,
            scale * values.low.to_numpy(dtype=float),
            scale * values.high.to_numpy(dtype=float),
            color=visual["color"],
            alpha=0.055,
            linewidth=0,
        )
    nodes, labels = deployment_ticks(data)
    ax.set_xticks(nodes, labels, rotation=52, ha="right")
    ax.tick_params(axis="x", labelsize=6.8)
    ax.set_xlabel("Fog nodes / workflow VNFs")
    ax.set_ylabel(ylabel)
    if logarithmic:
        ax.set_yscale("log")


def plot_bos_runtime(ax: plt.Axes, data: pd.DataFrame) -> None:
    for configuration in CONFIGS:
        values = data[data.configuration.eq(configuration)]
        rows = []
        for bos_size, group in values.groupby(
            "controlled_bos_size",
            sort=True,
        ):
            median, low, high = median_iqr(
                group.first_round_scheduler_runtime_s
            )
            rows.append((int(bos_size), median, low, high))
        curve = pd.DataFrame(
            rows,
            columns=("bos_size", "median", "low", "high"),
        )
        x = curve.bos_size.to_numpy(dtype=float)
        plot_curve(
            ax,
            x,
            curve["median"].to_numpy(dtype=float),
            configuration,
        )
        visual = STYLES[configuration]
        ax.fill_between(
            x,
            curve.low.to_numpy(dtype=float),
            curve.high.to_numpy(dtype=float),
            color=visual["color"],
            alpha=0.055,
            linewidth=0,
        )
    ax.set_yscale("log")
    ax.set_xticks(np.arange(2, 21, 2))
    ax.set_xlabel("Ready SFCs per BoS")
    ax.set_ylabel("Median scheduling runtime (s)")
    annotate_bos_capacity(ax, data)


def annotate_bos_capacity(ax: plt.Axes, data: pd.DataFrame) -> None:
    candidate_column = (
        "initial_effective_candidate_count"
        if "initial_effective_candidate_count" in data
        else "effective_candidate_count"
    )
    if candidate_column not in data:
        return
    values = pd.to_numeric(data[candidate_column], errors="coerce").dropna()
    if values.empty:
        return
    capacity = float(values.median())
    ax.axvline(
        capacity,
        color="#666666",
        linestyle=":",
        linewidth=1.0,
        alpha=0.8,
    )


def plot_bos_quality(
    ax: plt.Axes,
    data: pd.DataFrame,
    metric: str,
    ylabel: str,
    *,
    binary: bool = False,
) -> None:
    for configuration in CONFIGS:
        values = data[data.configuration.eq(configuration)]
        rows: list[tuple[int, float, float, float]] = []
        for bos_size, group in values.groupby(
            "controlled_bos_size",
            sort=True,
        ):
            if binary:
                mean = float(group[metric].mean())
                low, high = wilson_interval(group[metric])
            else:
                mean, low, high, _ = mean_ci95(group[metric])
            rows.append((int(bos_size), mean, low, high))
        curve = pd.DataFrame(
            rows,
            columns=("bos_size", "mean", "low", "high"),
        )
        x = curve.bos_size.to_numpy(dtype=float)
        plot_curve(
            ax,
            x,
            100.0 * curve["mean"].to_numpy(dtype=float),
            configuration,
        )
        visual = STYLES[configuration]
        ax.fill_between(
            x,
            100.0 * curve.low.to_numpy(dtype=float),
            100.0 * curve.high.to_numpy(dtype=float),
            color=visual["color"],
            alpha=0.055,
            linewidth=0,
        )
    ax.set_ylim(-3, 103)
    ax.set_xticks(np.arange(2, 21, 2))
    ax.set_xlabel("Ready SFCs per BoS")
    ax.set_ylabel(ylabel)
    annotate_bos_capacity(ax, data)


def figure_scalability(bos: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.55))
    plot_bos_runtime(axes[0], bos)
    axes[0].set_title("(a) BoS-width runtime")
    plot_bos_quality(
        axes[1],
        bos,
        "first_round_accepted_sfc_ratio",
        "Accepted-SFC ratio (%)",
    )
    axes[1].set_title("(b) BoS accepted-SFC ratio")

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
    axes[1].legend(
        handles=handles,
        loc="lower left",
        ncol=2,
        frameon=True,
    )
    fig.tight_layout()
    save(fig, "Figure6_runtime_bos_acceptance")


def require_values(
    data: pd.DataFrame,
    column: str,
    expected: set[float | int],
    label: str,
) -> None:
    observed = set(pd.to_numeric(data[column], errors="raise").unique())
    if observed != expected:
        raise SystemExit(
            f"{label} grid mismatch: expected {sorted(expected)}, "
            f"observed {sorted(observed)}"
        )


def validate_inputs(
    deadline: pd.DataFrame,
    deployment: pd.DataFrame,
    bos: pd.DataFrame,
) -> None:
    required = {
        "configuration",
        "scheduler",
        "sharing",
        "workflow_completed",
        "global_deadline_success",
        "sfc_subdeadline_success",
        "accepted_sfc_ratio",
        "provisioning_cost",
        "end_to_end_delay_s",
        "network_data_mb",
        "resource_utilization",
        "cpu_utilization",
        "ram_utilization",
        "max_link_bandwidth_utilization",
        "scheduler_runtime_s",
        "first_round_submitted_sfc_count",
        "first_round_admitted_sfc_count",
        "first_round_accepted_sfc_ratio",
        "first_round_scheduler_runtime_s",
    }
    for label, frame in (
        ("deadline", deadline),
        ("deployment", deployment),
        ("BoS", bos),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise SystemExit(f"{label} input missing columns: {sorted(missing)}")
        observed_configs = set(frame.configuration)
        if observed_configs != set(CONFIGS):
            raise SystemExit(
                f"{label} configurations mismatch: {sorted(observed_configs)}"
            )
    require_values(
        deployment,
        "node_count",
        {5, 8, 12, 18, 25, 30, 35, 40, 50, 60},
        "deployment",
    )
    require_values(bos, "controlled_bos_size", set(range(2, 21)), "BoS")
    require_values(
        deadline,
        "deadline_factor",
        EXPECTED_DEADLINE_FACTORS,
        "deadline",
    )
    if deadline.seed.nunique() != 50:
        raise SystemExit("Deadline input must contain exactly 50 paired seeds")
    expected_rows = {
        "deadline": 24_000,
        "deployment": 800,
        "BoS": 380,
    }
    observed_rows = {
        "deadline": len(deadline),
        "deployment": len(deployment),
        "BoS": len(bos),
    }
    if observed_rows != expected_rows:
        raise SystemExit(
            f"Row-count mismatch: expected {expected_rows}, "
            f"observed {observed_rows}"
        )
    if bool(
        bos.first_round_submitted_sfc_count
        .ne(bos.controlled_bos_size)
        .any()
    ):
        raise SystemExit("BoS first-round denominator is not the submitted width")
    calculated_acceptance = (
        bos.first_round_admitted_sfc_count
        / bos.first_round_submitted_sfc_count
    )
    if float(
        (
            calculated_acceptance
            - bos.first_round_accepted_sfc_ratio
        ).abs().max()
    ) > 1e-12:
        raise SystemExit("BoS first-round accepted ratio is inconsistent")


def main() -> None:
    style()
    deadline = pd.read_csv(
        PROJECT / "results/raw/deadline_curves_paper_aligned_50seeds.csv"
    )
    deployment = pd.read_csv(
        PROJECT / "results/raw/deployment_scaling_paper_aligned.csv"
    )
    bos = pd.read_csv(
        PROJECT / "results/raw/bos_scaling_paper_aligned.csv"
    )
    validate_inputs(deadline, deployment, bos)
    figure_deadlines(deadline)
    figure_cost_delay(deadline)
    figure_resources(deadline)
    figure_scalability(bos)
    required_figures = {
        "Figure3_deadline_success.png",
        "Figure4_macro_cost_delay.png",
        "Figure5_transfer_cpu_ram.png",
        "Figure6_runtime_bos_acceptance.png",
    }
    figures = {
        path.name
        for path in OUT.glob("*.png")
        if not path.name.startswith(".")
    }
    if figures != required_figures:
        raise SystemExit(
            "Canonical figure set mismatch: "
            f"expected {sorted(required_figures)}, observed {sorted(figures)}"
        )
    MANUSCRIPT_OUT.mkdir(parents=True, exist_ok=True)
    manuscript_names = {
        "Figure3_deadline_success.png": "deadline-robustness.png",
        "Figure4_macro_cost_delay.png": "AVGCost.png",
        "Figure5_transfer_cpu_ram.png": (
            "fig3_resource_network_curves.png"
        ),
        "Figure6_runtime_bos_acceptance.png": (
            "fig4_deployment_bos_scaling.png"
        ),
    }
    for source_name, manuscript_name in manuscript_names.items():
        shutil.copy2(OUT / source_name, MANUSCRIPT_OUT / manuscript_name)
    print(f"Wrote exactly four measured-point PNG figures to {OUT}")
    print(f"Wrote manuscript-filename copies to {MANUSCRIPT_OUT}")


if __name__ == "__main__":
    sys.exit(main())
