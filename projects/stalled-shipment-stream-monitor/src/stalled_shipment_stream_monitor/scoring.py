"""Score stall alerts against the planted ground truth.

A planted stall is *due* when the monitor could have confirmed it before the
stream ended: the watermark at the end of the replay is past the stall's last
scan plus the threshold for its stage. Stalls that are not due are reported
separately and left out of recall, because no setting could have flagged them.

False alerts are broken down into those later retracted by a late scan and
those on shipments with dead-lettered scans, the two causes seen on the sample.
The rest are reported as unexplained. A malformed scan with a blank shipment ID
cannot be tied to its shipment, so a false alert it causes lands there.

Detection delay is measured from the moment the stall became real (last scan
plus stage threshold) to the ingest time of the event that raised the alert.
It is the price paid for allowed lateness.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

from .events import EVENT_STAGE, Stage, parse_timestamp
from .monitor import Alert, AlertKind, StallMonitor
from .stream import read_events


@dataclass(frozen=True)
class PlantedStall:
    shipment_id: str
    stage: Stage
    last_event_time: datetime
    resumed_at: datetime | None


TRUTH_COLUMNS = ("shipment_id", "stalled_after", "last_event_time")


def load_truth(path: Path) -> list[PlantedStall]:
    """Read ``stalls.csv`` as written by the generator.

    ``resumed_at`` is optional. A file without the required columns, or with a
    row that is short or has a blank shipment ID, raises ``ValueError`` naming
    the file and line, so the CLI reports it instead of crashing.
    """
    stalls = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = [name.strip() for name in reader.fieldnames or []]
        missing = [c for c in TRUTH_COLUMNS if c not in header]
        if missing:
            raise ValueError(f"{path}: missing column(s) {', '.join(missing)}; "
                             f"expected {', '.join(TRUTH_COLUMNS)}[, resumed_at]")
        reader.fieldnames = header
        for row in reader:
            line = reader.line_num
            if any(row.get(c) is None for c in TRUTH_COLUMNS):
                raise ValueError(f"{path}: line {line} has fewer fields than the header")
            shipment_id = row["shipment_id"].strip()
            if not shipment_id:
                raise ValueError(f"{path}: line {line} has a blank shipment_id")
            event_type = row["stalled_after"].strip().upper()
            if event_type not in EVENT_STAGE:
                raise ValueError(f"{path}: unknown stalled_after value {row['stalled_after']!r}")
            resumed = (row.get("resumed_at") or "").strip()
            stalls.append(PlantedStall(
                shipment_id=shipment_id,
                stage=EVENT_STAGE[event_type],
                last_event_time=parse_timestamp(row["last_event_time"], "last_event_time"),
                resumed_at=parse_timestamp(resumed, "resumed_at") if resumed else None,
            ))
    ids = [s.shipment_id for s in stalls]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: a shipment has more than one planted stall")
    return stalls


@dataclass(frozen=True)
class Score:
    allowed_lateness: timedelta
    alerts: int
    true_alerts: int
    false_alerts: int
    retracted: int
    false_with_dead_letters: int
    unexplained: int
    due: int
    detected: int
    not_due: int
    delay_hours: tuple[float, ...]

    @property
    def precision(self) -> float:
        return self.true_alerts / self.alerts if self.alerts else 0.0

    @property
    def recall(self) -> float:
        return self.detected / self.due if self.due else 0.0

    @property
    def median_delay_hours(self) -> float | None:
        return statistics.median(self.delay_hours) if self.delay_hours else None

    @property
    def p90_delay_hours(self) -> float | None:
        if not self.delay_hours:
            return None
        ordered = sorted(self.delay_hours)
        # Nearest-rank percentile: no interpolation between alert delays.
        rank = max(1, -(-9 * len(ordered) // 10))
        return ordered[rank - 1]


def score(
    alerts: Iterable[Alert],
    truth: Sequence[PlantedStall],
    thresholds: dict[Stage, timedelta],
    final_watermark: datetime | None,
    allowed_lateness: timedelta,
) -> Score:
    """Compare STALLED alerts with planted stalls, one stall per shipment."""
    planted = {s.shipment_id: s for s in truth}
    stalled: dict[str, Alert] = {}
    retracted: set[str] = set()
    for alert in alerts:
        if alert.kind is AlertKind.STALLED:
            stalled.setdefault(alert.shipment_id, alert)
        elif alert.kind is AlertKind.RETRACTED:
            retracted.add(alert.shipment_id)

    due = {
        sid for sid, s in planted.items()
        if final_watermark is not None and s.last_event_time + thresholds[s.stage] < final_watermark
    }
    true_ids = set(stalled) & set(planted)
    detected = true_ids & due
    delays = tuple(
        (stalled[sid].raised_at_ingest - (planted[sid].last_event_time + thresholds[planted[sid].stage]))
        .total_seconds() / 3600
        for sid in sorted(detected)
    )
    false_ids = set(stalled) - set(planted)
    with_dead_letters = {sid for sid in false_ids if stalled[sid].dead_letters}
    return Score(
        allowed_lateness=allowed_lateness,
        alerts=len(stalled),
        true_alerts=len(true_ids),
        false_alerts=len(false_ids),
        retracted=len(false_ids & retracted),
        false_with_dead_letters=len(with_dead_letters),
        unexplained=len(false_ids - retracted - with_dead_letters),
        due=len(due),
        detected=len(detected),
        not_due=len(planted) - len(due),
        delay_hours=delays,
    )


def replay_and_score(events_path: Path, truth_path: Path, monitor: StallMonitor) -> Score:
    """Replay ``events_path`` through a fresh ``monitor`` and score the result."""
    alerts = monitor.run(read_events(events_path))
    return score(alerts, load_truth(truth_path), monitor.thresholds, monitor.watermark,
                 monitor.allowed_lateness)


def sweep(events_path: Path, truth_path: Path, lateness_hours: Sequence[float]) -> list[Score]:
    """Score the same stream at several allowed-lateness settings."""
    if not lateness_hours:
        raise ValueError("give at least one lateness value")
    return [
        replay_and_score(events_path, truth_path, StallMonitor(allowed_lateness=timedelta(hours=h)))
        for h in lateness_hours
    ]


def format_table(scores: Sequence[Score]) -> str:
    header = ("lateness_h", "alerts", "false", "retracted", "dead_letter", "unexplained", "precision", "recall",
              "due", "not_due", "median_delay_h", "p90_delay_h")
    lines = [" | ".join(header), " | ".join("---" for _ in header)]
    for s in scores:
        lines.append(" | ".join([
            f"{s.allowed_lateness.total_seconds() / 3600:g}",
            str(s.alerts),
            str(s.false_alerts),
            str(s.retracted),
            str(s.false_with_dead_letters),
            str(s.unexplained),
            f"{s.precision:.3f}",
            f"{s.recall:.3f}",
            str(s.due),
            str(s.not_due),
            _hours(s.median_delay_hours),
            _hours(s.p90_delay_hours),
        ]))
    return "\n".join(lines)


def _hours(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"
