#!/usr/bin/env python3
"""Recover a D10 shard after an explicitly identified writer overlap.

This utility is deliberately narrow.  It preserves the verified unique prefix
and suffix of one quarantined D10 CSV, removes the known concurrently measured
ordinal interval, and writes an audit manifest.  The normal deployment driver
must then be run with ``--resume`` to regenerate the removed cases.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile


FAMILIES = ("montage", "epigenomics", "inspiral", "cybershake")
CONFIGURATIONS = (
    "HDS-Sharable",
    "HDS-NonSharable",
    "PBQS-Sharable",
    "PBQS-NonSharable",
)
IDENTITY = ("topology", "family", "seed", "configuration")


def expected_keys() -> list[tuple[str, str, str, str]]:
    return [
        ("xxlarge", family, str(seed), configuration)
        for family in FAMILIES
        for seed in range(1, 6)
        for configuration in CONFIGURATIONS
    ]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv_write(
    path: Path,
    *,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor,
            "w",
            newline="",
            encoding="utf-8",
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--discard-start", type=int, default=36)
    parser.add_argument("--discard-end", type=int, default=43)
    parser.add_argument(
        "--allow-missing-tail-start",
        type=int,
        help=(
            "Permit only the exact contiguous missing tail from this D10 "
            "ordinal through 80; all earlier non-discarded ordinals must "
            "still be present exactly once."
        ),
    )
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.output}")
    if args.discard_start < 1 or args.discard_end > 80:
        raise SystemExit("Discard interval must be within D10 ordinals 1--80")
    if args.discard_start > args.discard_end:
        raise SystemExit("--discard-start cannot exceed --discard-end")
    if (
        args.allow_missing_tail_start is not None
        and not 1 <= args.allow_missing_tail_start <= 80
    ):
        raise SystemExit("--allow-missing-tail-start must be within 1--80")

    expected = expected_keys()
    ordinal_by_key = {
        key: ordinal for ordinal, key in enumerate(expected, start=1)
    }
    with args.input.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise SystemExit(f"{args.input} has no CSV header")
        missing = set(IDENTITY) - set(reader.fieldnames)
        if missing:
            raise SystemExit(
                f"{args.input} is missing identity columns: {sorted(missing)}"
            )
        fieldnames = list(reader.fieldnames)
        observed_rows = list(reader)

    rows_by_ordinal: dict[int, list[dict[str, str]]] = {
        ordinal: [] for ordinal in range(1, 81)
    }
    unexpected: list[tuple[str, str, str, str]] = []
    for row in observed_rows:
        key = tuple(str(row[column]) for column in IDENTITY)
        ordinal = ordinal_by_key.get(key)
        if ordinal is None:
            unexpected.append(key)
            continue
        rows_by_ordinal[ordinal].append(row)
    if unexpected:
        raise SystemExit(f"Unexpected D10 identities: {unexpected[:5]}")

    missing_ordinals = [
        ordinal
        for ordinal, rows in rows_by_ordinal.items()
        if not rows
    ]
    allowed_missing = (
        list(range(args.allow_missing_tail_start, 81))
        if args.allow_missing_tail_start is not None
        else []
    )
    if missing_ordinals != allowed_missing:
        raise SystemExit(
            "Quarantine missing-ordinal set does not match the explicitly "
            f"allowed tail: observed={missing_ordinals}, "
            f"allowed={allowed_missing}"
        )

    discard = set(range(args.discard_start, args.discard_end + 1))
    missing_set = set(missing_ordinals)
    ambiguous_retained = {
        ordinal: len(rows)
        for ordinal, rows in rows_by_ordinal.items()
        if (
            ordinal not in discard
            and ordinal not in missing_set
            and len(rows) != 1
        )
    }
    if ambiguous_retained:
        raise SystemExit(
            "Retained ordinals must each have exactly one row: "
            f"{ambiguous_retained}"
        )

    retained_rows = [
        rows_by_ordinal[ordinal][0]
        for ordinal in range(1, 81)
        if ordinal not in discard and ordinal not in missing_set
    ]
    atomic_csv_write(
        args.output,
        fieldnames=fieldnames,
        rows=retained_rows,
    )

    manifest = {
        "operation": "recover_deployment_d10_writer_overlap",
        "input": str(args.input),
        "input_sha256": file_sha256(args.input),
        "input_rows": len(observed_rows),
        "expected_unique_keys": len(expected),
        "discarded_ordinal_start": args.discard_start,
        "discarded_ordinal_end": args.discard_end,
        "discarded_unique_keys": len(discard),
        "discarded_observed_rows": sum(
            len(rows_by_ordinal[ordinal]) for ordinal in discard
        ),
        "allowed_missing_tail_start": args.allow_missing_tail_start,
        "missing_ordinals": missing_ordinals,
        "retained_rows": len(retained_rows),
        "output": str(args.output),
        "output_sha256": file_sha256(args.output),
        "resume_expected_new_rows": len(discard) + len(missing_ordinals),
        "duplicate_observations_by_ordinal": {
            str(ordinal): len(rows)
            for ordinal, rows in rows_by_ordinal.items()
            if len(rows) > 1
        },
        "status": "ready_for_single_writer_resume",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
