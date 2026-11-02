"""Replay a recorded tracking stream from CSV in arrival order."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from .events import FIELDS


def read_events(path: Path) -> Iterator[dict[str, str]]:
    """Yield raw event rows in file order (the file is already in ingest order).

    The header must contain every event field; rows are passed through as raw
    mappings so that malformed values reach the monitor's dead-letter list
    instead of stopping the replay.
    """
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        missing = [f for f in FIELDS if f not in header]
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
        for row in reader:
            yield {k: v for k, v in row.items() if k is not None}
