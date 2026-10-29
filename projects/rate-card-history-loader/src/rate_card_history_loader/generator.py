"""Synthetic rate-card CDC stream.

Produces a realistic change history for a carrier rate table and then delivers
it the way an at-least-once CDC pipeline would: in batches, shuffled within a
batch, with redelivered duplicates, one replayed batch and a few malformed
events. Everything is driven by a seed so runs are reproducible.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

from .events import ChangeEvent

CARRIERS = ("BlueDart Surface", "Delhivery Freight", "Gati Express", "Safexpress", "TCI Freight")
CITIES = ("DEL", "BOM", "BLR", "MAA", "HYD", "CCU", "AMD", "PNQ", "JAI", "LKO")


@dataclass
class GeneratedStream:
    history: list[ChangeEvent]  # true change history in commit order
    batches: list[list[dict]]  # what the loader actually receives


def _round_rate(value: float) -> float:
    return round(value, 2)


def generate_history(
    n_lanes: int = 40,
    days: int = 120,
    start: datetime = datetime(2026, 6, 1),
    seed: int = 7,
) -> list[ChangeEvent]:
    """Generate the true change history of the rate table in commit order."""
    rng = random.Random(seed)
    lsn = 1000
    ts = start
    history: list[ChangeEvent] = []
    current: dict[str, ChangeEvent] = {}

    def next_ts() -> datetime:
        nonlocal ts
        ts = ts + timedelta(minutes=rng.randint(1, 90))
        return ts

    def emit(event: ChangeEvent) -> None:
        nonlocal lsn
        history.append(event)
        lsn += rng.randint(1, 5)

    # Initial load: every lane is created on day 0.
    lanes = []
    for i in range(n_lanes):
        origin, destination = rng.sample(CITIES, 2)
        lane_id = f"L{i + 1:03d}-{origin}-{destination}"
        lanes.append(lane_id)
        event = ChangeEvent(
            lsn=lsn,
            op="c",
            commit_ts=next_ts(),
            lane_id=lane_id,
            carrier=rng.choice(CARRIERS),
            origin=origin,
            destination=destination,
            rate_per_kg=_round_rate(rng.uniform(6.0, 22.0)),
            min_charge=float(rng.choice((250, 300, 400, 500))),
            currency="INR",
        )
        current[lane_id] = event
        emit(event)

    deleted: dict[str, ChangeEvent] = {}
    for day in range(1, days):
        ts = max(ts, start + timedelta(days=day, hours=9))
        for _ in range(rng.randint(0, 4)):
            roll = rng.random()
            if deleted and roll < 0.06:
                # A withdrawn lane is re-contracted.
                lane_id = rng.choice(sorted(deleted))
                old = deleted.pop(lane_id)
                event = replace(
                    old,
                    lsn=lsn,
                    op="c",
                    commit_ts=next_ts(),
                    rate_per_kg=_round_rate(old.rate_per_kg * rng.uniform(1.0, 1.1)),
                )
                current[lane_id] = event
                emit(event)
            elif roll < 0.10 and len(current) > n_lanes // 2:
                lane_id = rng.choice(sorted(current))
                current.pop(lane_id)
                old = [e for e in history if e.lane_id == lane_id][-1]
                deleted[lane_id] = old
                emit(ChangeEvent(lsn=lsn, op="d", commit_ts=next_ts(), lane_id=lane_id))
            elif roll < 0.25:
                # The source touched the row without changing any tracked value.
                lane_id = rng.choice(sorted(current))
                event = replace(current[lane_id], lsn=lsn, op="u", commit_ts=next_ts())
                current[lane_id] = event
                emit(event)
            else:
                lane_id = rng.choice(sorted(current))
                old = current[lane_id]
                change = {"rate_per_kg": _round_rate(old.rate_per_kg * rng.uniform(0.92, 1.12))}
                if rng.random() < 0.2:
                    change["min_charge"] = float(rng.choice((250, 300, 400, 500, 600)))
                event = replace(old, lsn=lsn, op="u", commit_ts=next_ts(), **change)
                current[lane_id] = event
                emit(event)
    return history


def _as_record(event: ChangeEvent) -> dict:
    return json.loads(event.to_json())


def deliver(
    history: list[ChangeEvent],
    batch_size: int = 60,
    duplicate_rate: float = 0.05,
    n_bad: int = 4,
    seed: int = 11,
) -> list[list[dict]]:
    """Split the history into delivery batches with at-least-once defects.

    - events inside a batch are shuffled (arrival order differs from LSN order)
    - about ``duplicate_rate`` of events are delivered twice, sometimes in the
      next batch
    - the second batch is delivered again at the end (a replay after a failure)
    - ``n_bad`` malformed events are mixed into random batches
    """
    rng = random.Random(seed)
    batches = [
        [_as_record(e) for e in history[i : i + batch_size]]
        for i in range(0, len(history), batch_size)
    ]
    for idx, batch in enumerate(batches):
        dupes = [dict(r) for r in batch if rng.random() < duplicate_rate]
        for dupe in dupes:
            target = idx + 1 if idx + 1 < len(batches) and rng.random() < 0.3 else idx
            batches[target].append(dupe)

    bad_templates = [
        lambda r: {**r, "rate_per_kg": -abs(r["rate_per_kg"] or 1.0)},
        lambda r: {**r, "op": "x"},
        lambda r: {k: v for k, v in r.items() if k != "commit_ts"},
        lambda r: {**r, "currency": "rupees"},
    ]
    max_lsn = history[-1].lsn if history else 0
    for i in range(n_bad):
        idx = rng.randrange(len(batches))
        template = next(r for r in batches[idx] if r["op"] in ("c", "u"))
        bad = bad_templates[i % len(bad_templates)](dict(template))
        bad["lsn"] = max_lsn + 1000 + i  # never collides with a real change
        batches[idx].append(bad)

    for batch in batches:
        rng.shuffle(batch)
    if len(batches) > 1:
        batches.append([dict(r) for r in batches[1]])
    return batches


def generate_stream(seed: int = 7, **kwargs) -> GeneratedStream:
    history = generate_history(seed=seed, **kwargs)
    return GeneratedStream(history=history, batches=deliver(history, seed=seed + 4))


def write_stream(stream: GeneratedStream, out_dir: Path) -> list[Path]:
    """Write each delivery batch to ``out_dir/batch_NNNN.jsonl``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, batch in enumerate(stream.batches, start=1):
        path = out_dir / f"batch_{i:04d}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for record in batch:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
        paths.append(path)
    return paths


__all__ = ["GeneratedStream", "generate_history", "deliver", "generate_stream", "write_stream"]


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/cdc_batches")
    written = write_stream(generate_stream(), target)
    print(f"wrote {len(written)} batches to {target}")
