#!/usr/bin/env python3
"""Merge validated experiment shards without mixing schemas or case keys."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


IDENTITIES = {
    "deadline": [
        "family",
        "workflow_size",
        "deadline_factor",
        "seed",
        "configuration",
    ],
    "deployment": ["topology", "family", "seed", "configuration"],
    "bos": [
        "controlled_bos_size",
        "deadline_factor",
        "seed",
        "configuration",
        "candidate_count",
    ],
}

EXPECTED_ROWS = {
    "deadline": 24_000,
    "deployment": 800,
    "bos": 380,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("study", choices=sorted(IDENTITIES))
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int)
    args = parser.parse_args()

    frames = [pd.read_csv(path) for path in args.inputs]
    reference_columns = list(frames[0].columns)
    for path, frame in zip(args.inputs, frames):
        if list(frame.columns) != reference_columns:
            raise SystemExit(f"Schema mismatch in {path}")

    identity = IDENTITIES[args.study]
    missing = set(identity) - set(reference_columns)
    if missing:
        raise SystemExit(f"Missing identity columns: {sorted(missing)}")
    merged = pd.concat(frames, ignore_index=True)
    duplicates = merged.duplicated(identity, keep=False)
    if bool(duplicates.any()):
        examples = merged.loc[duplicates, identity].head().to_dict("records")
        raise SystemExit(f"Duplicate {args.study} cases: {examples}")

    expected_rows = args.expected_rows or EXPECTED_ROWS[args.study]
    if len(merged) != expected_rows:
        raise SystemExit(
            f"Expected {expected_rows} rows, observed {len(merged)}"
        )
    merged = merged.sort_values(identity).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(
        f"Merged {len(merged)} unique {args.study} cases into {args.output}"
    )


if __name__ == "__main__":
    main()
