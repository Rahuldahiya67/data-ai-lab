"""Stall detection over a tracking stream processed in arrival order.

Time in the monitor is event time, never wall-clock time. The watermark is the
largest event time seen so far minus the allowed lateness: the monitor assumes
that by now it has received every scan with an event time at or before the
watermark. A shipment is stalled when the watermark has moved more than the
threshold for its current stage past its latest scan.

Deadlines are kept in a min-heap, so each event costs O(log n) instead of a
scan over every open shipment. Heap entries carry a version number; an entry
whose version no longer matches the shipment's state is stale and skipped.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Iterable, Mapping

from .events import InvalidEvent, Stage, TrackingEvent, parse_event

DEFAULT_THRESHOLDS: dict[Stage, timedelta] = {
    Stage.PICKED_UP: timedelta(hours=8),
    Stage.AT_HUB: timedelta(hours=12),
    Stage.IN_TRANSIT: timedelta(hours=14),
    Stage.OUT_FOR_DELIVERY: timedelta(hours=8),
}
DEFAULT_ALLOWED_LATENESS = timedelta(hours=4)


class AlertKind(str, Enum):
    STALLED = "STALLED"
    RECOVERED = "RECOVERED"


@dataclass(frozen=True)
class Alert:
    kind: AlertKind
    shipment_id: str
    stage: Stage
    location: str
    last_event_time: datetime
    watermark: datetime | None
    raised_at_ingest: datetime


@dataclass
class ShipmentState:
    stage: Stage
    location: str
    last_event_time: datetime
    version: int = 0
    stalled: bool = False

    @property
    def closed(self) -> bool:
        return self.stage is Stage.DELIVERED


@dataclass
class StreamStats:
    accepted: int = 0
    duplicates: int = 0
    invalid: int = 0
    late: int = 0
    out_of_order: int = 0


@dataclass
class DeadLetter:
    raw: Mapping[str, object]
    reason: str


@dataclass
class StallMonitor:
    thresholds: dict[Stage, timedelta] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    allowed_lateness: timedelta = DEFAULT_ALLOWED_LATENESS

    def __post_init__(self) -> None:
        open_stages = [s for s in Stage if s is not Stage.DELIVERED]
        missing = [s.value for s in open_stages if s not in self.thresholds]
        if missing:
            raise ValueError(f"no stall threshold for stages: {', '.join(missing)}")
        for stage, limit in self.thresholds.items():
            if limit <= timedelta(0):
                raise ValueError(f"threshold for {stage.value} must be positive")
        if self.allowed_lateness < timedelta(0):
            raise ValueError("allowed_lateness cannot be negative")
        self.shipments: dict[str, ShipmentState] = {}
        self.stats = StreamStats()
        self.dead_letters: list[DeadLetter] = []
        self._seen_ids: set[str] = set()
        self._deadlines: list[tuple[datetime, str, int]] = []
        self._max_event_time: datetime | None = None

    @property
    def watermark(self) -> datetime | None:
        if self._max_event_time is None:
            return None
        return self._max_event_time - self.allowed_lateness

    def process_raw(self, raw: Mapping[str, object]) -> list[Alert]:
        """Validate and process one raw message; invalid ones go to dead letters."""
        try:
            event = parse_event(raw)
        except InvalidEvent as exc:
            self.stats.invalid += 1
            self.dead_letters.append(DeadLetter(raw=raw, reason=str(exc)))
            return []
        return self.process(event)

    def process(self, event: TrackingEvent) -> list[Alert]:
        if event.event_id in self._seen_ids:
            self.stats.duplicates += 1
            return []
        self._seen_ids.add(event.event_id)
        self.stats.accepted += 1

        mark = self.watermark
        if mark is not None and event.event_time <= mark:
            self.stats.late += 1

        alerts: list[Alert] = []
        state = self.shipments.get(event.shipment_id)
        if state is None:
            state = ShipmentState(event.stage, event.location, event.event_time)
            self.shipments[event.shipment_id] = state
            self._schedule(event.shipment_id, state)
        elif event.event_time < state.last_event_time:
            # An older scan arriving late does not move the shipment backwards.
            self.stats.out_of_order += 1
        else:
            if state.stalled:
                alerts.append(Alert(
                    kind=AlertKind.RECOVERED,
                    shipment_id=event.shipment_id,
                    stage=event.stage,
                    location=event.location,
                    last_event_time=event.event_time,
                    watermark=self.watermark,
                    raised_at_ingest=event.ingest_time,
                ))
                state.stalled = False
            state.stage = event.stage
            state.location = event.location
            state.last_event_time = event.event_time
            state.version += 1
            self._schedule(event.shipment_id, state)

        if self._max_event_time is None or event.event_time > self._max_event_time:
            self._max_event_time = event.event_time
        alerts.extend(self._expire(event.ingest_time))
        return alerts

    def run(self, raws: Iterable[Mapping[str, object]]) -> list[Alert]:
        alerts: list[Alert] = []
        for raw in raws:
            alerts.extend(self.process_raw(raw))
        return alerts

    def open_stalls(self) -> list[str]:
        return sorted(sid for sid, s in self.shipments.items() if s.stalled)

    def _schedule(self, shipment_id: str, state: ShipmentState) -> None:
        if state.closed:
            return
        deadline = state.last_event_time + self.thresholds[state.stage]
        heapq.heappush(self._deadlines, (deadline, shipment_id, state.version))

    def _expire(self, ingest_time: datetime) -> list[Alert]:
        mark = self.watermark
        alerts: list[Alert] = []
        while self._deadlines and self._deadlines[0][0] < mark:
            _, shipment_id, version = heapq.heappop(self._deadlines)
            state = self.shipments[shipment_id]
            if version != state.version or state.closed or state.stalled:
                continue
            state.stalled = True
            alerts.append(Alert(
                kind=AlertKind.STALLED,
                shipment_id=shipment_id,
                stage=state.stage,
                location=state.location,
                last_event_time=state.last_event_time,
                watermark=mark,
                raised_at_ingest=ingest_time,
            ))
        return alerts
