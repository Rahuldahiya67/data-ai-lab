"""CDC event model for the carrier rate table.

The source table is ``rate_card`` keyed by ``lane_id``. Each change is published
as one event with the full after-image of the row (deletes carry the key only).
The stream is at-least-once, so the loader must tolerate duplicates,
out-of-order arrival inside a batch and replayed batches.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

OPS = ("c", "u", "d")  # create, update, delete (Debezium style)
ATTRIBUTES = ("carrier", "origin", "destination", "rate_per_kg", "min_charge", "currency")


@dataclass(frozen=True)
class ChangeEvent:
    lsn: int
    op: str
    commit_ts: datetime
    lane_id: str
    carrier: str | None = None
    origin: str | None = None
    destination: str | None = None
    rate_per_kg: float | None = None
    min_charge: float | None = None
    currency: str | None = None

    def to_json(self) -> str:
        record = asdict(self)
        record["commit_ts"] = self.commit_ts.isoformat()
        return json.dumps(record, sort_keys=True)


def write_jsonl(events: Iterable[ChangeEvent], path: Path) -> int:
    """Write events to a JSON Lines file and return the number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(event.to_json() + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[dict]:
    """Yield raw event records from a JSON Lines file.

    Records are returned as plain dicts, not ``ChangeEvent`` objects, because
    they have not been validated yet. Blank lines are skipped. A line that is
    not valid JSON is yielded as ``{"_raw": line, "_error": ...}`` so that the
    validator can quarantine it instead of the whole file failing.
    """
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                yield {"_raw": line, "_error": f"line {line_no}: invalid JSON ({exc.msg})"}
                continue
            if not isinstance(record, dict):
                yield {"_raw": line, "_error": f"line {line_no}: expected a JSON object"}
                continue
            yield record
