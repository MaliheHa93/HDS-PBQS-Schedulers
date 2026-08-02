#!/usr/bin/env python3
"""Validate the complete paper-aligned v0.7.0 release without changing data."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from hds.result_validation import validate_scalability_results


PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "results/raw"
FIGURES = PROJECT / "results/figures_paper_aligned"
MANUSCRIPT_FIGURES = PROJECT / "results/figures_for_manuscript"
CONFIGURATIONS = (
    "HDS-Sharable",
    "PBQS-Sharable",
    "HDS-NonSharable",
    "PBQS-NonSharable",
)
FAMILIES = ("montage", "epigenomics", "inspiral", "cybershake")
DEADLINE_FACTORS = (
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
)
DEPLOYMENT_PROFILES = (
    ("small", 5, 100),
    ("small_plus", 8, 200),
    ("medium", 12, 300),
    ("medium_plus", 18, 450),
    ("large", 25, 600),
    ("large_plus", 30, 700),
    ("xlarge_minus", 35, 750),
    ("xlarge", 40, 800),
    ("xlarge_plus", 50, 900),
    ("xxlarge", 60, 1000),
)
FIGURE_NAMES = {
    "Figure3_deadline_success.png": "deadline-robustness.png",
    "Figure4_macro_cost_delay.png": "AVGCost.png",
    "Figure5_transfer_cpu_ram.png": (
        "fig3_resource_network_curves.png"
    ),
    "Figure6_runtime_bos_acceptance.png": (
        "fig4_deployment_bos_scaling.png"
    ),
}
PROCESSED_FILES = (
    "deadline_curves_paper_aligned_summary.csv",
    "deadline_curves_paper_aligned_normalized.csv",
    "deadline_curves_paper_aligned_paired.csv",
    "deadline_curves_paper_aligned_inference.csv",
    "deadline_curve_level_auc.csv",
    "deadline_kappa90.csv",
    "deployment_scaling_paper_aligned_summary.csv",
    "deployment_scaling_paper_aligned_normalized.csv",
    "deployment_scaling_paper_aligned_paired.csv",
    "bos_scaling_paper_aligned_summary.csv",
    "bos_scaling_paper_aligned_paired.csv",
)


@dataclass(frozen=True, slots=True)
class DatasetResult:
    name: str
    path: str
    rows: int
    expected_rows: int
    duplicate_rows: int
    validation_errors: int
    validation_warnings: int
    exact_grid: bool
    solver_limit_rows: int
    unfinished_rows: int

    @property
    def passed(self) -> bool:
        return (
            self.rows == self.expected_rows
            and self.duplicate_rows == 0
            and self.validation_errors == 0
            and self.exact_grid
        )


def normalized_tuple(row: tuple[object, ...]) -> tuple[object, ...]:
    result: list[object] = []
    for value in row:
        if isinstance(value, float):
            result.append(round(value, 12))
        else:
            result.append(value)
    return tuple(result)


def observed_grid(
    data: pd.DataFrame,
    columns: tuple[str, ...],
) -> set[tuple[object, ...]]:
    return {
        normalized_tuple(tuple(row))
        for row in data.loc[:, columns].itertuples(index=False, name=None)
    }


def validate_dataset(
    name: str,
    filename: str,
    columns: tuple[str, ...],
    expected: set[tuple[object, ...]],
) -> DatasetResult:
    path = RAW / filename
    data = pd.read_csv(path)
    report = validate_scalability_results(
        data,
        minimum_seeds_for_inference=5,
    )
    limits = int(
        pd.to_numeric(
            data["milp_limit_reached_count"],
            errors="raise",
        ).gt(0).sum()
    )
    unfinished = int(
        pd.to_numeric(
            data["unfinished_sfc_count"],
            errors="raise",
        ).gt(0).sum()
    )
    return DatasetResult(
        name=name,
        path=str(path.relative_to(PROJECT)),
        rows=len(data),
        expected_rows=len(expected),
        duplicate_rows=report.duplicate_rows,
        validation_errors=report.error_count,
        validation_warnings=report.warning_count,
        exact_grid=observed_grid(data, columns) == expected,
        solver_limit_rows=limits,
        unfinished_rows=unfinished,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_figures() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for canonical_name, manuscript_name in FIGURE_NAMES.items():
        canonical = FIGURES / canonical_name
        manuscript = MANUSCRIPT_FIGURES / manuscript_name
        with Image.open(canonical) as image:
            image.load()
            width, height = image.size
            dpi = image.info.get("dpi", (0.0, 0.0))
        canonical_hash = sha256(canonical)
        manuscript_hash = sha256(manuscript)
        rows.append(
            {
                "canonical": str(canonical.relative_to(PROJECT)),
                "manuscript_copy": str(manuscript.relative_to(PROJECT)),
                "width_px": width,
                "height_px": height,
                "dpi_x": float(dpi[0]),
                "dpi_y": float(dpi[1]),
                "sha256": canonical_hash,
                "copy_is_identical": canonical_hash == manuscript_hash,
                "passed": (
                    width > 0
                    and height > 0
                    and float(dpi[0]) >= 299.0
                    and float(dpi[1]) >= 299.0
                    and canonical_hash == manuscript_hash
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "results/VALIDATION_REPORT.json",
    )
    args = parser.parse_args()

    deadline_columns = (
        "family",
        "workflow_size",
        "deadline_factor",
        "seed",
        "configuration",
    )
    deadline_expected = {
        normalized_tuple(item)
        for item in itertools.product(
            FAMILIES,
            (100,),
            DEADLINE_FACTORS,
            range(1, 51),
            CONFIGURATIONS,
        )
    }

    deployment_columns = (
        "topology",
        "node_count",
        "workflow_size",
        "family",
        "seed",
        "configuration",
    )
    deployment_expected = {
        normalized_tuple(
            (topology, nodes, size, family, seed, configuration)
        )
        for topology, nodes, size in DEPLOYMENT_PROFILES
        for family in FAMILIES
        for seed in range(1, 6)
        for configuration in CONFIGURATIONS
    }

    bos_columns = (
        "controlled_bos_size",
        "deadline_factor",
        "seed",
        "configuration",
        "candidate_count",
    )
    bos_expected = {
        normalized_tuple(item)
        for item in itertools.product(
            range(2, 21),
            (5.0,),
            range(1, 6),
            CONFIGURATIONS,
            (20,),
        )
    }

    datasets = [
        validate_dataset(
            "deadline",
            "deadline_curves_paper_aligned_50seeds.csv",
            deadline_columns,
            deadline_expected,
        ),
        validate_dataset(
            "deployment",
            "deployment_scaling_paper_aligned.csv",
            deployment_columns,
            deployment_expected,
        ),
        validate_dataset(
            "bos",
            "bos_scaling_paper_aligned.csv",
            bos_columns,
            bos_expected,
        ),
    ]
    figure_rows = validate_figures()
    processed_missing = [
        filename
        for filename in PROCESSED_FILES
        if not (PROJECT / "results/processed" / filename).is_file()
    ]
    global_sub_mismatch = int(
        (
            pd.read_csv(
                RAW / "deadline_curves_paper_aligned_50seeds.csv",
                usecols=(
                    "global_deadline_success",
                    "sfc_subdeadline_success",
                ),
            )
            .pipe(
                lambda data: (
                    data["global_deadline_success"]
                    != data["sfc_subdeadline_success"]
                )
            )
            .sum()
        )
    )
    payload = {
        "release": "v0.7.0-paper-aligned",
        "passed": (
            all(item.passed for item in datasets)
            and all(bool(item["passed"]) for item in figure_rows)
            and not processed_missing
        ),
        "datasets": [
            {**asdict(item), "passed": item.passed}
            for item in datasets
        ],
        "figures": figure_rows,
        "processed_files_missing": processed_missing,
        "deadline_global_subdeadline_mismatch_rows": (
            global_sub_mismatch
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
