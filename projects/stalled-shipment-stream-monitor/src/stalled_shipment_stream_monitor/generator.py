"""Synthetic carrier tracking stream with planted stalls.

Each shipment follows a normal journey: pickup, one or two hubs, out for
delivery, delivered. Normal gaps between scans stay well under the stall
thresholds. A share of shipments get one planted stall: a long gap after some
scan, after which the shipment either resumes or goes silent until the end of
the observation window.

The stream is written in arrival (ingest) order and has the faults real feeds
have: most scans arrive within minutes, some arrive hours late in a batch
upload, some are delivered twice, and a few are malformed.

Two files are written:

- ``events.csv``: the stream, one row per delivered message, in ingest order.
- ``stalls.csv``: ground truth, one row per planted stall.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .events import FIELDS

START = datetime(2026, 10, 26, tzinfo=timezone.utc)

# Normal gap ranges in hours after an event of this type, before the next scan.
NORMAL_GAP_HOURS = {
    "PICKED_UP": (1.0, 5.0),
    "ARRIVED_HUB": (1.0, 8.0),
    "DEPARTED_HUB": (2.0, 10.0),
    "OUT_FOR_DELIVERY": (1.0, 5.0),
}
STALL_GAP_HOURS = (24.0, 72.0)

HUBS = ["DEL-HUB", "GGN-HUB", "JAI-HUB", "LKO-HUB", "AMB-HUB", "AGR-HUB"]
CITIES = ["Delhi", "Gurugram", "Noida", "Jaipur", "Lucknow", "Chandigarh", "Agra"]

STALL_FIELDS = ("shipment_id", "stalled_after", "last_event_time", "resumed_at")

# One CSV row, field name -> value, as written to disk.
Row = dict[str, str]


@dataclass(frozen=True)
class GeneratorConfig:
    shipments: int = 600
    days: int = 6
    stall_rate: float = 0.08
    resume_rate: float = 0.5
    late_rate: float = 0.10
    duplicate_rate: float = 0.03
    malformed_rate: float = 0.005
    seed: int = 7

    def __post_init__(self) -> None:
        if self.shipments < 1:
            raise ValueError("shipments must be at least 1")
        if self.days < 1:
            raise ValueError("days must be at least 1")
        for name in ("stall_rate", "resume_rate", "late_rate", "duplicate_rate", "malformed_rate"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")


def _journey(rng: random.Random) -> list[tuple[str, str]]:
    """Event types and locations for one shipment, in order."""
    origin, dest = rng.sample(CITIES, 2)
    hubs = rng.sample(HUBS, rng.choice([1, 2]))
    steps = [("PICKED_UP", origin)]
    for hub in hubs:
        steps += [("ARRIVED_HUB", hub), ("DEPARTED_HUB", hub)]
    steps += [("OUT_FOR_DELIVERY", dest), ("DELIVERED", dest)]
    return steps


def _ingest_delay(rng: random.Random, late_rate: float) -> timedelta:
    if rng.random() < late_rate:
        return timedelta(hours=rng.uniform(1.0, 4.0))
    return timedelta(minutes=rng.uniform(0.5, 20.0))


def _fmt(ts: datetime) -> str:
    return ts.isoformat(timespec="seconds")


def generate(config: GeneratorConfig = GeneratorConfig()) -> tuple[list[Row], list[Row]]:
    """Return (events in ingest order, ground-truth stalls)."""
    rng = random.Random(config.seed)
    horizon = START + timedelta(days=config.days)
    # Leave the last day for journeys to play out.
    start_span = (config.days - 1) * 24.0

    rows: list[tuple[datetime, int, Row]] = []
    stalls: list[Row] = []
    seq = 0

    for n in range(1, config.shipments + 1):
        shipment_id = f"SHP{n:05d}"
        steps = _journey(rng)
        stall_at = rng.randrange(len(steps) - 1) if rng.random() < config.stall_rate else None
        resumes = rng.random() < config.resume_rate

        t = START + timedelta(hours=rng.uniform(0.0, start_span))
        for i, (event_type, location) in enumerate(steps):
            if t > horizon:
                break
            event_id = f"{shipment_id}-{i + 1}"
            ingest = t + _ingest_delay(rng, config.late_rate)
            if ingest <= horizon:
                row = {
                    "event_id": event_id,
                    "shipment_id": shipment_id,
                    "event_type": event_type,
                    "event_time": _fmt(t),
                    "ingest_time": _fmt(ingest),
                    "location": location,
                }
                if rng.random() < config.malformed_rate:
                    row = _corrupt(rng, row)
                rows.append((ingest, seq, row))
                seq += 1
                if rng.random() < config.duplicate_rate:
                    dup_ingest = ingest + timedelta(minutes=rng.uniform(1.0, 90.0))
                    if dup_ingest <= horizon:
                        rows.append((dup_ingest, seq, {**row, "ingest_time": _fmt(dup_ingest)}))
                        seq += 1

            if event_type == "DELIVERED":
                break
            if i == stall_at:
                gap = timedelta(hours=rng.uniform(*STALL_GAP_HOURS))
                resumed = t + gap if resumes else None
                if resumed is not None and resumed > horizon:
                    resumed = None
                stalls.append({
                    "shipment_id": shipment_id,
                    "stalled_after": event_type,
                    "last_event_time": _fmt(t),
                    "resumed_at": _fmt(resumed) if resumed else "",
                })
                if resumed is None:
                    break
                t = resumed
            else:
                t += timedelta(hours=rng.uniform(*NORMAL_GAP_HOURS[event_type]))

    rows.sort(key=lambda r: (r[0], r[1]))
    return [r[2] for r in rows], stalls


def _corrupt(rng: random.Random, row: Row) -> Row:
    kind = rng.choice(["timestamp", "event_type", "shipment_id"])
    bad = dict(row)
    if kind == "timestamp":
        bad["event_time"] = bad["event_time"].replace("T", " ")[:16]  # drops the offset
    elif kind == "event_type":
        bad["event_type"] = "SCANNED"
    else:
        bad["shipment_id"] = ""
    return bad


def write_csv(path: Path, rows: list[Row], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def write_dataset(out_dir: Path, config: GeneratorConfig = GeneratorConfig()) -> tuple[int, int]:
    events, stalls = generate(config)
    write_csv(Path(out_dir) / "events.csv", events, FIELDS)
    write_csv(Path(out_dir) / "stalls.csv", stalls, STALL_FIELDS)
    return len(events), len(stalls)


if __name__ == "__main__":
    n_events, n_stalls = write_dataset(Path("data"))
    print(f"wrote {n_events} events and {n_stalls} planted stalls to data/")
