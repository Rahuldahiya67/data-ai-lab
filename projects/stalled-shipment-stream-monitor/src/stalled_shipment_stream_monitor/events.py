"""Tracking event model and validation.

A raw event is a mapping of strings, as it would arrive from a CSV file or a
JSON message. ``parse_event`` turns it into a typed ``TrackingEvent`` or raises
``InvalidEvent`` with a reason that can go to a dead-letter list.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Mapping

FIELDS = ("event_id", "shipment_id", "event_type", "event_time", "ingest_time", "location")

# Scanners batch-upload, so ingest is normally after the event. A small negative
# gap is tolerated for clock skew between the device and the ingest service.
MAX_CLOCK_SKEW = timedelta(minutes=5)


class Stage(str, Enum):
    """Where a shipment is in its journey after a given event."""

    PICKED_UP = "PICKED_UP"
    AT_HUB = "AT_HUB"
    IN_TRANSIT = "IN_TRANSIT"
    OUT_FOR_DELIVERY = "OUT_FOR_DELIVERY"
    DELIVERED = "DELIVERED"


# Event type -> stage the shipment is in after that event.
EVENT_STAGE: dict[str, Stage] = {
    "PICKED_UP": Stage.PICKED_UP,
    "ARRIVED_HUB": Stage.AT_HUB,
    "DEPARTED_HUB": Stage.IN_TRANSIT,
    "OUT_FOR_DELIVERY": Stage.OUT_FOR_DELIVERY,
    "DELIVERED": Stage.DELIVERED,
}


class InvalidEvent(ValueError):
    """Raised when a raw event cannot be accepted into the stream."""


@dataclass(frozen=True)
class TrackingEvent:
    event_id: str
    shipment_id: str
    event_type: str
    event_time: datetime
    ingest_time: datetime
    location: str

    @property
    def stage(self) -> Stage:
        return EVENT_STAGE[self.event_type]


def parse_timestamp(value: str, field: str) -> datetime:
    """Parse an ISO 8601 timestamp. Naive values are rejected, not guessed."""
    try:
        ts = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise InvalidEvent(f"{field} is not an ISO 8601 timestamp: {value!r}") from exc
    if ts.tzinfo is None:
        raise InvalidEvent(f"{field} has no timezone offset: {value!r}")
    return ts.astimezone(timezone.utc)


def parse_event(raw: Mapping[str, object]) -> TrackingEvent:
    """Validate a raw event and return a ``TrackingEvent``."""
    missing = [f for f in FIELDS if f not in raw]
    if missing:
        raise InvalidEvent(f"missing fields: {', '.join(missing)}")

    values = {}
    for field in FIELDS:
        value = raw[field]
        if value is None or not isinstance(value, str):
            raise InvalidEvent(f"{field} must be a string, got {type(value).__name__}")
        values[field] = value.strip()

    for field in ("event_id", "shipment_id", "event_type"):
        if not values[field]:
            raise InvalidEvent(f"{field} is empty")

    event_type = values["event_type"].upper()
    if event_type not in EVENT_STAGE:
        raise InvalidEvent(f"unknown event_type: {values['event_type']!r}")

    event_time = parse_timestamp(values["event_time"], "event_time")
    ingest_time = parse_timestamp(values["ingest_time"], "ingest_time")
    if ingest_time < event_time - MAX_CLOCK_SKEW:
        raise InvalidEvent(
            f"ingest_time {ingest_time.isoformat()} is before event_time "
            f"{event_time.isoformat()} by more than {MAX_CLOCK_SKEW}"
        )

    return TrackingEvent(
        event_id=values["event_id"],
        shipment_id=values["shipment_id"],
        event_type=event_type,
        event_time=event_time,
        ingest_time=ingest_time,
        location=values["location"],
    )
