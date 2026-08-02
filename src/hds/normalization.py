"""Transparent normalization helpers for publication tables and plots."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


NORMALIZED_METRICS = (
    "provisioning_cost",
    "end_to_end_delay_s",
    "network_data_mb",
)


def add_normalized_columns(
    data: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    deadline_group_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Add documented deadline and max-observed metric normalizations.

    Cost, delay, and inter-node data volume are divided by the maximum
    observed value inside ``group_columns``. Deadline ratio and min-max
    normalization use ``deadline_group_columns`` (or the same groups).
    Raw observations are copied and never overwritten.
    """

    required = {
        *group_columns,
        "workflow_deadline_duration_s",
        *NORMALIZED_METRICS,
    }
    deadline_groups = list(deadline_group_columns or group_columns)
    required.update(deadline_groups)
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"Cannot normalize data; missing columns: {sorted(missing)}"
        )

    result = data.copy()
    for metric in NORMALIZED_METRICS:
        values = pd.to_numeric(result[metric], errors="coerce")
        maxima = values.groupby(
            [result[column] for column in group_columns],
            dropna=False,
        ).transform("max")
        result[f"{metric}_normalized"] = np.where(
            maxima > 0,
            values / maxima,
            0.0,
        )

    deadlines = pd.to_numeric(
        result["workflow_deadline_duration_s"],
        errors="coerce",
    )
    deadline_keys = [result[column] for column in deadline_groups]
    minimum = deadlines.groupby(deadline_keys, dropna=False).transform("min")
    maximum = deadlines.groupby(deadline_keys, dropna=False).transform("max")
    result["deadline_ratio"] = np.where(
        minimum > 0,
        deadlines / minimum,
        np.nan,
    )
    span = maximum - minimum
    result["deadline_normalized_0_1"] = np.where(
        span > 0,
        (deadlines - minimum) / span,
        0.0,
    )
    return result
