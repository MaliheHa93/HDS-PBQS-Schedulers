"""Streaming CSV helpers used by long experiment runs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping


class DetailCsvWriter:
    """Write long-format detail rows incrementally with schema checks."""

    def __init__(
        self,
        path: Path | None,
        *,
        append: bool,
    ) -> None:
        self.path = path
        self.stream = None
        self.writer: csv.DictWriter | None = None
        self.fieldnames: list[str] | None = None
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append and path.exists() else "w"
        if mode == "a":
            with path.open(newline="", encoding="utf-8") as existing:
                reader = csv.DictReader(existing)
                self.fieldnames = reader.fieldnames
        self.stream = path.open(mode, newline="", encoding="utf-8")
        if self.fieldnames:
            self.writer = csv.DictWriter(
                self.stream,
                fieldnames=self.fieldnames,
            )

    def write(
        self,
        rows: Iterable[Mapping[str, object]],
        *,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        if self.stream is None:
            return
        enriched = [
            {**row, **(extra or {})}
            for row in rows
        ]
        if not enriched:
            return
        if self.writer is None:
            self.fieldnames = list(enriched[0])
            self.writer = csv.DictWriter(
                self.stream,
                fieldnames=self.fieldnames,
            )
            self.writer.writeheader()
        for row in enriched:
            if list(row) != self.fieldnames:
                missing = set(self.fieldnames) - set(row)
                extra_columns = set(row) - set(self.fieldnames)
                raise ValueError(
                    "Detail CSV schema changed: "
                    f"missing={sorted(missing)}, "
                    f"extra={sorted(extra_columns)}"
                )
            self.writer.writerow(row)
        self.stream.flush()

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()

    def __enter__(self) -> "DetailCsvWriter":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
