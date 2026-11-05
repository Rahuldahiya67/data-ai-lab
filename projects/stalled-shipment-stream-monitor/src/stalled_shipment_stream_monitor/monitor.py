"""Stall detection over a tracking stream processed in arrival order.

Time in the monitor is event time, never wall-clock time. The watermark is the
largest event time seen so far minus the allowed lateness: the monitor assumes
that by now it has received every scan with an event time at or before the
watermark. A shipment is stalled when the watermark has moved more than the
threshold for its current stage past its latest scan.

Deadlines are kept in a min-heap, so each event costs O(log n) instead of a
scan over every open shipment. Heap entries carry a version number; an entry
whose version no longer matches the shipment's state is stale and skipped.

State stays bounded by the number of open shipments: once the watermark is more
than ``state_retention`` past a delivered shipment's last scan, its state and
event IDs are released. Any event older than ``watermark - state_retention`` is
rejected as too late, so a very late duplicate cannot re-create an evicted
shipment and raise a false stall.
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
DEFAULT_STATE_RETENTION = timedelta(hours=24)


class AlertKind(str, Enum):
    """``STALLED``: no scan for longer than the stage threshold.

    ``RECOVERED``: a stalled shipment scanned again after a real gap.
    ``RETRACTED``: a late-arriving scan shows the gap never exceeded the
    threshold, so the earlier ``STALLED`` alert was a false alarm.
    """

    STALLED = "STALLED"
    RECOVERED = "RECOVERED"
    RETRACTED = "RETRACTED"


@dataclass(frozen=True)
class Alert:
    kind: AlertKind
    shipment_id: str
    stage: Stage
    location: str
    last_event_time: datetime
    watermark: datetime | None
    raised_at_ingest: datetime
    # Scans for this shipment rejected by validation. A non-zero count on a
    # STALLED alert means the missing scan may be in the dead-letter list.
    dead_letters: int = 0


@dataclass
class ShipmentState:
    stage: Stage
    location: str
    last_event_time: datetime
    version: int = 0
    stalled: bool = False
    event_ids: set[str] = field(default_factory=set)

    @property
    def closed(self) -> bool:
        return self.stage is Stage.DELIVERED


@dataclass
class StreamStats:
    """Counters for one replay.

    ``accepted``: events processed (not duplicates, not rejected).
    ``duplicates``: events whose ID was already seen for the shipment.
    ``invalid``: raw messages that failed validation.
    ``late``: accepted events whose event time was at or behind the watermark
    when they arrived. They are still processed; the count shows how many scans
    the current allowed lateness did not cover. With zero lateness every event
    that is not a new maximum event time counts.
    ``out_of_order``: events older than the shipment's latest scan.
    ``too_late``: events more than ``state_retention`` behind the watermark,
    rejected to the dead-letter list.
    ``evicted``: delivered shipments released from state.
    """

    accepted: int = 0
    duplicates: int = 0
    invalid: int = 0
    late: int = 0
    out_of_order: int = 0
    too_late: int = 0
    evicted: int = 0


@dataclass
class DeadLetter:
    raw: Mapping[str, object]
    reason: str


@dataclass
class StallMonitor:
    thresholds: dict[Stage, timedelta] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    allowed_lateness: timedelta = DEFAULT_ALLOWED_LATENESS
    state_retention: timedelta = DEFAULT_STATE_RETENTION

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
        if self.state_retention <= timedelta(0):
            raise ValueError("state_retention must be positive")
        self.shipments: dict[str, ShipmentState] = {}
        self.stats = StreamStats()
        self.dead_letters: list[DeadLetter] = []
        self._dead_letters_by_shipment: dict[str, int] = {}
        self._deadlines: list[tuple[datetime, str, int]] = []
        self._closures: list[tuple[datetime, str, int]] = []
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
            shipment_id = raw.get("shipment_id")
            if isinstance(shipment_id, str) and shipment_id.strip():
                key = shipment_id.strip()
                self._dead_letters_by_shipment[key] = self._dead_letters_by_shipment.get(key, 0) + 1
            return []
        return self.process(event)

    def process(self, event: TrackingEvent) -> list[Alert]:
        state = self.shipments.get(event.shipment_id)
        if state is not None and event.event_id in state.event_ids:
            self.stats.duplicates += 1
            return []

        mark = self.watermark
        if mark is not None and event.event_time < mark - self.state_retention:
            self.stats.too_late += 1
            self.dead_letters.append(DeadLetter(
                raw=_event_row(event),
                reason=f"event_time is more than {self.state_retention} behind the watermark",
            ))
            return []

        self.stats.accepted += 1
        if mark is not None and event.event_time <= mark:
            self.stats.late += 1

        alerts: list[Alert] = []
        if state is None:
            state = ShipmentState(event.stage, event.location, event.event_time)
            state.event_ids.add(event.event_id)
            self.shipments[event.shipment_id] = state
            self._schedule(event.shipment_id, state)
        elif event.event_time < state.last_event_time:
            # An older scan arriving late does not move the shipment backwards.
            state.event_ids.add(event.event_id)
            self.stats.out_of_order += 1
        else:
            if state.stalled:
                within = event.event_time <= state.last_event_time + self.thresholds[state.stage]
                alerts.append(Alert(
                    kind=AlertKind.RETRACTED if within else AlertKind.RECOVERED,
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
            state.event_ids.add(event.event_id)
            self._schedule(event.shipment_id, state)

        if self._max_event_time is None or event.event_time > self._max_event_time:
            self._max_event_time = event.event_time
        alerts.extend(self._expire(event.ingest_time))
        self._evict()
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
            release = state.last_event_time + self.state_retention
            heapq.heappush(self._closures, (release, shipment_id, state.version))
            return
        deadline = state.last_event_time + self.thresholds[state.stage]
        heapq.heappush(self._deadlines, (deadline, shipment_id, state.version))

    def _expire(self, ingest_time: datetime) -> list[Alert]:
        mark = self.watermark
        alerts: list[Alert] = []
        if mark is None:
            return alerts
        while self._deadlines and self._deadlines[0][0] < mark:
            _, shipment_id, version = heapq.heappop(self._deadlines)
            # With a retention shorter than a stage threshold, a delivered
            # shipment can be released before its old deadline entry pops.
            state = self.shipments.get(shipment_id)
            if state is None or version != state.version or state.closed or state.stalled:
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
                dead_letters=self._dead_letters_by_shipment.get(shipment_id, 0),
            ))
        return alerts

    def _evict(self) -> None:
        mark = self.watermark
        if mark is None:
            return
        while self._closures and self._closures[0][0] < mark:
            _, shipment_id, version = heapq.heappop(self._closures)
            state = self.shipments.get(shipment_id)
            if state is None or state.version != version or not state.closed:
                continue
            del self.shipments[shipment_id]
            self._dead_letters_by_shipment.pop(shipment_id, None)
            self.stats.evicted += 1


def _event_row(event: TrackingEvent) -> dict[str, str]:
    return {
        "event_id": event.event_id,
        "shipment_id": event.shipment_id,
        "event_type": event.event_type,
        "event_time": event.event_time.isoformat(),
        "ingest_time": event.ingest_time.isoformat(),
        "location": event.location,
    }
