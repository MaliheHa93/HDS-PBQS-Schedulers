#!/usr/bin/env python3
"""Merge independently generated deployment-family shards safely."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


IDENTITY = ["topology", "family", "seed", "configuration"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int, default=800)
    args = parser.parse_args()

    input_frames = [pd.read_csv(path) for path in args.inputs]
    if not input_frames:
        raise SystemExit("At least one shard is required")
    reference_columns = list(input_frames[0].columns)
    reference_set = set(reference_columns)
    frames: list[pd.DataFrame] = []
    for path, frame in zip(args.inputs, input_frames):
        observed_set = set(frame.columns)
        if observed_set != reference_set:
            raise SystemExit(
                f"Schema mismatch in {path}: "
                f"missing={sorted(reference_set - observed_set)}, "
                f"extra={sorted(observed_set - reference_set)}"
            )
        frames.append(frame.loc[:, reference_columns])
    merged = pd.concat(frames, ignore_index=True)
    missing = set(IDENTITY) - set(merged.columns)
    if missing:
        raise SystemExit(f"Missing identity columns: {sorted(missing)}")
    duplicates = merged.duplicated(IDENTITY, keep=False)
    if duplicates.any():
        examples = merged.loc[duplicates, IDENTITY].head().to_dict("records")
        raise SystemExit(f"Duplicate deployment cases: {examples}")
    if len(merged) != args.expected_rows:
        raise SystemExit(
            f"Expected {args.expected_rows} rows, observed {len(merged)}"
        )
    merged = merged.sort_values(IDENTITY).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Merged {len(merged)} unique deployment cases into {args.output}")


if __name__ == "__main__":
    main()
