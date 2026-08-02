"""Structural and scientific-integrity checks for raw experiment results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Mapping

import pandas as pd


CASE_COLUMNS = (
    "topology",
    "family",
    "workflow_size",
    "deadline_factor",
    "seed",
    "configuration",
    "candidate_count",
    "eta",
    "alpha",
    "beta",
    "gamma",
    "omega_u",
    "omega_w",
    "tastd_mode",
    "enable_vm_reuse",
    "joint_bos_optimization",
)

REQUIRED_COLUMNS = set(CASE_COLUMNS) | {
    "scheduler",
    "sharing",
    "node_count",
    "local_node_count",
    "global_node_count",
    "unfinished_sfc_count",
    "completed_sfc_count",
    "total_sfc_count",
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
    "workflow_completed",
    "global_deadline_success",
    "sfc_subdeadline_success",
    "workflow_deadline_success",
    "accepted_sfc_ratio",
    "first_round_submitted_sfc_count",
    "first_round_admitted_sfc_count",
    "first_round_accepted_sfc_ratio",
    "first_round_scheduler_runtime_s",
    "reference_deadline_feasible",
    "deadline_kappa",
    "sfc_deadline_miss_rate",
    "makespan_s",
    "vm_reuse_rate",
    "provisioned_vm_count",
    "scheduler_runtime_s",
    "scheduler_overhead_ratio",
    "case_wall_runtime_s",
    "milp_runtime_s",
    "milp_max_gap",
    "milp_limit_reached_count",
    "workflow_deadline_s",
    "workflow_deadline_duration_s",
    "deadline_to_minimum_makespan_ratio",
    "requested_candidate_count",
    "initial_effective_candidate_count",
    "milp_min_effective_candidate_count",
    "milp_max_effective_candidate_count",
}

CONDITIONAL_ON_COMPLETION_COLUMNS = {
    "end_to_end_delay_s",
    "makespan_s",
}

NONNEGATIVE_COLUMNS = (
    "unfinished_sfc_count",
    "completed_sfc_count",
    "total_sfc_count",
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
    "accepted_sfc_ratio",
    "first_round_submitted_sfc_count",
    "first_round_admitted_sfc_count",
    "first_round_accepted_sfc_ratio",
    "first_round_scheduler_runtime_s",
    "sfc_deadline_miss_rate",
    "makespan_s",
    "vm_reuse_rate",
    "provisioned_vm_count",
    "scheduler_runtime_s",
    "scheduler_overhead_ratio",
    "case_wall_runtime_s",
    "milp_runtime_s",
    "milp_max_gap",
    "milp_limit_reached_count",
    "workflow_deadline_s",
    "workflow_deadline_duration_s",
    "deadline_to_minimum_makespan_ratio",
    "requested_candidate_count",
    "initial_effective_candidate_count",
    "milp_min_effective_candidate_count",
    "milp_max_effective_candidate_count",
)


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One validation finding."""

    severity: str
    code: str
    message: str
    affected_rows: int = 0


@dataclass(frozen=True, slots=True)
class ResultValidationReport:
    """Machine-readable validation result."""

    input_rows: int
    expected_rows: int | None
    duplicate_rows: int
    error_count: int
    warning_count: int
    issues: tuple[ValidationIssue, ...]

    @property
    def passed(self) -> bool:
        return self.error_count == 0

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "input_rows": self.input_rows,
            "expected_rows": self.expected_rows,
            "duplicate_rows": self.duplicate_rows,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [asdict(issue) for issue in self.issues],
        }


def _normal_case_key(values: Iterable[object]) -> tuple[object, ...]:
    normalized: list[object] = []
    for column, value in zip(CASE_COLUMNS, values):
        if column in {
            "workflow_size",
            "seed",
            "candidate_count",
            "enable_vm_reuse",
            "joint_bos_optimization",
        }:
            normalized.append(int(value))
        elif column in {
            "deadline_factor",
            "eta",
            "alpha",
            "beta",
            "gamma",
            "omega_u",
            "omega_w",
        }:
            normalized.append(round(float(value), 12))
        else:
            normalized.append(str(value))
    return tuple(normalized)


def case_key(record: Mapping[str, object]) -> tuple[object, ...]:
    """Return a stable case identity across CSV numeric representations."""

    return _normal_case_key(record[column] for column in CASE_COLUMNS)


def _numeric_mask(
    data: pd.DataFrame,
    column: str,
    predicate,
) -> pd.Series:
    values = pd.to_numeric(data[column], errors="coerce")
    return values.isna() | ~values.map(math.isfinite) | predicate(values)


def validate_scalability_results(
    data: pd.DataFrame,
    *,
    expected_case_keys: set[tuple[object, ...]] | None = None,
    topology_shapes: Mapping[str, tuple[int, int]] | None = None,
    minimum_seeds_for_inference: int = 5,
) -> ResultValidationReport:
    """Validate a raw Table III result frame without altering observations."""

    issues: list[ValidationIssue] = []
    missing_columns = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing_columns:
        issues.append(
            ValidationIssue(
                "error",
                "missing_columns",
                f"Missing required columns: {missing_columns}",
            )
        )
        return ResultValidationReport(
            input_rows=len(data),
            expected_rows=(
                len(expected_case_keys) if expected_case_keys is not None else None
            ),
            duplicate_rows=0,
            error_count=1,
            warning_count=0,
            issues=tuple(issues),
        )

    duplicates = int(data.duplicated(list(CASE_COLUMNS), keep=False).sum())
    if duplicates:
        issues.append(
            ValidationIssue(
                "error",
                "duplicate_cases",
                "Duplicate experimental case identities were found.",
                duplicates,
            )
        )

    completed_mask = pd.to_numeric(
        data["workflow_completed"],
        errors="coerce",
    ).eq(1)
    for column in NONNEGATIVE_COLUMNS:
        values = pd.to_numeric(data[column], errors="coerce")
        if column in CONDITIONAL_ON_COMPLETION_COLUMNS:
            raw_missing = data[column].isna()
            nonfinite_present = (
                ~raw_missing
                & (~values.map(math.isfinite) | (values < 0))
            )
            missing_for_completed = raw_missing & completed_mask
            invalid = nonfinite_present | missing_for_completed
        else:
            invalid = values.isna() | ~values.map(math.isfinite) | (values < 0)
        count = int(invalid.sum())
        if count:
            issues.append(
                ValidationIssue(
                    "error",
                    f"invalid_{column}",
                    f"{column} contains non-finite or negative values.",
                    count,
                )
            )

    for column in (
        "resource_utilization",
        "cpu_utilization",
        "ram_utilization",
        "vm_time_utilization",
        "mean_link_bandwidth_utilization",
        "max_link_bandwidth_utilization",
        "accepted_sfc_ratio",
        "first_round_accepted_sfc_ratio",
        "workflow_completed",
        "global_deadline_success",
        "sfc_subdeadline_success",
        "workflow_deadline_success",
        "reference_deadline_feasible",
        "sfc_deadline_miss_rate",
        "vm_reuse_rate",
    ):
        invalid = _numeric_mask(
            data,
            column,
            lambda values: (values < 0) | (values > 1),
        )
        count = int(invalid.sum())
        if count:
            issues.append(
                ValidationIssue(
                    "error",
                    f"out_of_range_{column}",
                    f"{column} must be within [0, 1].",
                    count,
                )
            )

    completed = pd.to_numeric(data["completed_sfc_count"], errors="coerce")
    total = pd.to_numeric(data["total_sfc_count"], errors="coerce")
    unfinished = pd.to_numeric(data["unfinished_sfc_count"], errors="coerce")
    inconsistent = (completed + unfinished != total).fillna(True)
    if int(inconsistent.sum()):
        issues.append(
            ValidationIssue(
                "error",
                "inconsistent_completion_counts",
                "completed + unfinished must equal total SFC count.",
                int(inconsistent.sum()),
            )
        )

    workflow_completed = pd.to_numeric(
        data["workflow_completed"], errors="coerce"
    )
    global_success = pd.to_numeric(
        data["global_deadline_success"], errors="coerce"
    )
    subdeadline_success = pd.to_numeric(
        data["sfc_subdeadline_success"], errors="coerce"
    )
    joint_success = pd.to_numeric(
        data["workflow_deadline_success"], errors="coerce"
    )
    accepted_ratio = pd.to_numeric(
        data["accepted_sfc_ratio"], errors="coerce"
    )
    first_round_submitted = pd.to_numeric(
        data["first_round_submitted_sfc_count"], errors="coerce"
    )
    first_round_admitted = pd.to_numeric(
        data["first_round_admitted_sfc_count"], errors="coerce"
    )
    first_round_ratio = pd.to_numeric(
        data["first_round_accepted_sfc_ratio"], errors="coerce"
    )
    expected_completed = (completed == total).astype(int)
    completion_mismatch = workflow_completed != expected_completed
    if int(completion_mismatch.sum()):
        issues.append(
            ValidationIssue(
                "error",
                "workflow_completion_mismatch",
                "workflow_completed must equal completed_sfc_count == total_sfc_count.",
                int(completion_mismatch.sum()),
            )
        )

    success_without_completion = (
        ((global_success == 1) | (subdeadline_success == 1))
        & (workflow_completed != 1)
    )
    if int(success_without_completion.sum()):
        issues.append(
            ValidationIssue(
                "error",
                "success_without_completion",
                "A deadline outcome cannot be successful for an unfinished workflow.",
                int(success_without_completion.sum()),
            )
        )

    expected_joint = (
        (global_success == 1) & (subdeadline_success == 1)
    ).astype(int)
    joint_mismatch = joint_success != expected_joint
    if int(joint_mismatch.sum()):
        issues.append(
            ValidationIssue(
                "error",
                "joint_success_mismatch",
                "workflow_deadline_success must equal global AND subdeadline success.",
                int(joint_mismatch.sum()),
            )
        )

    expected_accepted = completed / total.replace(0, math.nan)
    accepted_mismatch = (
        (accepted_ratio - expected_accepted).abs() > 1e-9
    ).fillna(True)
    if int(accepted_mismatch.sum()):
        issues.append(
            ValidationIssue(
                "error",
                "accepted_ratio_mismatch",
                "accepted_sfc_ratio must equal completed_sfc_count / total_sfc_count.",
                int(accepted_mismatch.sum()),
            )
        )

    invalid_first_round_counts = (
        (first_round_submitted <= 0)
        | (first_round_admitted > first_round_submitted)
    )
    if int(invalid_first_round_counts.sum()):
        issues.append(
            ValidationIssue(
                "error",
                "invalid_first_round_counts",
                (
                    "First-round admitted SFCs cannot exceed the positive "
                    "submitted ready-set count."
                ),
                int(invalid_first_round_counts.sum()),
            )
        )
    expected_first_round_ratio = (
        first_round_admitted / first_round_submitted.replace(0, math.nan)
    )
    first_round_ratio_mismatch = (
        (first_round_ratio - expected_first_round_ratio).abs() > 1e-9
    ).fillna(True)
    if int(first_round_ratio_mismatch.sum()):
        issues.append(
            ValidationIssue(
                "error",
                "first_round_ratio_mismatch",
                (
                    "first_round_accepted_sfc_ratio must equal first-round "
                    "admitted / submitted."
                ),
                int(first_round_ratio_mismatch.sum()),
            )
        )

    requested_candidates = pd.to_numeric(
        data["requested_candidate_count"], errors="coerce"
    )
    initial_candidates = pd.to_numeric(
        data["initial_effective_candidate_count"], errors="coerce"
    )
    minimum_candidates = pd.to_numeric(
        data["milp_min_effective_candidate_count"], errors="coerce"
    )
    maximum_candidates = pd.to_numeric(
        data["milp_max_effective_candidate_count"], errors="coerce"
    )
    invalid_candidates = (
        (requested_candidates <= 0)
        | (initial_candidates > requested_candidates)
        | (minimum_candidates > requested_candidates)
        | (maximum_candidates > requested_candidates)
        | (minimum_candidates > maximum_candidates)
    )
    # Runs without a MILP invocation record both extrema as zero.
    no_milp = pd.to_numeric(data["milp_runtime_s"], errors="coerce").eq(0)
    invalid_candidates &= ~(
        no_milp & minimum_candidates.eq(0) & maximum_candidates.eq(0)
    )
    if int(invalid_candidates.sum()):
        issues.append(
            ValidationIssue(
                "error",
                "invalid_effective_candidate_counts",
                "Effective candidate counts must be ordered and cannot exceed the requested cap.",
                int(invalid_candidates.sum()),
            )
        )

    reference_mode = (
        data["deadline_mode"].astype(str).str.lower().eq("reference")
        if "deadline_mode" in data
        else pd.Series(False, index=data.index)
    )
    deadline_factor = pd.to_numeric(
        data["deadline_factor"], errors="coerce"
    )
    deadline_kappa = pd.to_numeric(data["deadline_kappa"], errors="coerce")
    kappa_mismatch = (
        reference_mode
        & ((deadline_factor - deadline_kappa).abs() > 1e-9)
    )
    if int(kappa_mismatch.sum()):
        issues.append(
            ValidationIssue(
                "error",
                "deadline_factor_mismatch",
                "Reference-mode deadline_factor must equal the implemented deadline_kappa.",
                int(kappa_mismatch.sum()),
            )
        )

    unfinished_count = int((unfinished > 0).sum())
    if unfinished_count:
        issues.append(
            ValidationIssue(
                "warning",
                "unfinished_workflows",
                "Some runs contain unfinished SFCs; retain and explain them.",
                unfinished_count,
            )
        )

    solver_limits = int(
        (pd.to_numeric(data["milp_limit_reached_count"], errors="coerce") > 0).sum()
    )
    if solver_limits:
        issues.append(
            ValidationIssue(
                "warning",
                "solver_limits",
                "At least one MILP solve reached its configured limit.",
                solver_limits,
            )
        )

    for columns, code in (
        (("alpha", "beta", "gamma"), "invalid_hds_weights"),
        (("omega_u", "omega_w"), "invalid_pbqs_weights"),
    ):
        total_weight = sum(
            (
                pd.to_numeric(data[column], errors="coerce")
                for column in columns
            ),
            start=pd.Series(0.0, index=data.index),
        )
        invalid = ~total_weight.map(math.isfinite) | ((total_weight - 1.0).abs() > 1e-9)
        count = int(invalid.sum())
        if count:
            issues.append(
                ValidationIssue(
                    "error",
                    code,
                    f"{' + '.join(columns)} must equal 1.",
                    count,
                )
            )

    if topology_shapes:
        invalid_shape = 0
        for row in data[
            [
                "topology",
                "node_count",
                "local_node_count",
                "global_node_count",
            ]
        ].itertuples(index=False):
            shape = topology_shapes.get(str(row.topology))
            if shape is None:
                invalid_shape += 1
                continue
            local, global_ = shape
            if (
                int(row.local_node_count) != local
                or int(row.global_node_count) != global_
                or int(row.node_count) != local + global_
            ):
                invalid_shape += 1
        if invalid_shape:
            issues.append(
                ValidationIssue(
                    "error",
                    "topology_shape_mismatch",
                    "Rows do not match the declared Table III node composition.",
                    invalid_shape,
                )
            )

    observed_keys = {
        _normal_case_key(row)
        for row in data.loc[:, CASE_COLUMNS].itertuples(index=False, name=None)
    }
    if expected_case_keys is not None:
        missing = expected_case_keys - observed_keys
        unexpected = observed_keys - expected_case_keys
        if missing:
            issues.append(
                ValidationIssue(
                    "error",
                    "missing_cases",
                    "The raw CSV does not contain every expected case.",
                    len(missing),
                )
            )
        if unexpected:
            issues.append(
                ValidationIssue(
                    "warning",
                    "unexpected_cases",
                    "The raw CSV contains cases outside the requested matrix.",
                    len(unexpected),
                )
            )

    pair_groups = [
        column
        for column in CASE_COLUMNS
        if column not in {"configuration"}
    ] + ["sharing"]
    incomplete_pairs = 0
    for _, frame in data.groupby(pair_groups, dropna=False, sort=False):
        schedulers = set(frame["scheduler"].astype(str).str.upper())
        if schedulers != {"HDS", "PBQS"}:
            incomplete_pairs += 1
    if incomplete_pairs:
        issues.append(
            ValidationIssue(
                "error",
                "incomplete_scheduler_pairs",
                "HDS/PBQS pairs are incomplete for matched cases.",
                incomplete_pairs,
            )
        )

    seed_groups = [
        "topology",
        "family",
        "workflow_size",
        "deadline_factor",
        "candidate_count",
        "configuration",
    ]
    smallest_seed_count = int(
        data.groupby(seed_groups, dropna=False)["seed"].nunique().min()
    )
    if smallest_seed_count < minimum_seeds_for_inference:
        issues.append(
            ValidationIssue(
                "warning",
                "small_sample",
                (
                    "At least one result group has only "
                    f"{smallest_seed_count} independent seed(s); use pilot wording."
                ),
            )
        )

    return ResultValidationReport(
        input_rows=len(data),
        expected_rows=(
            len(expected_case_keys) if expected_case_keys is not None else None
        ),
        duplicate_rows=duplicates,
        error_count=sum(issue.severity == "error" for issue in issues),
        warning_count=sum(issue.severity == "warning" for issue in issues),
        issues=tuple(issues),
    )
