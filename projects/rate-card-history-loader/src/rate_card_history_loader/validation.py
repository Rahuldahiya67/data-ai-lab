"""Validate raw CDC records before they reach the dimension.

Bad events are quarantined with a reason instead of failing the batch: one
malformed event from the source should not stop every valid rate change behind
it from loading.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime

from .events import ATTRIBUTES, OPS, ChangeEvent

CURRENCY_RE = re.compile(r"^[A-Z]{3}$")


@dataclass
class ValidationResult:
    valid: list[ChangeEvent] = field(default_factory=list)
    rejected: list[tuple[dict, str]] = field(default_factory=list)
    duplicates: int = 0  # exact redeliveries collapsed inside the batch


class InvalidEvent(ValueError):
    pass


def _require_int(record: dict, name: str) -> int:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidEvent(f"{name} must be an integer, got {value!r}")
    if value <= 0:
        raise InvalidEvent(f"{name} must be positive, got {value}")
    return value


def _require_text(record: dict, name: str) -> str:
    value = record.get(name)
    if not isinstance(value, str) or not value.strip():
        raise InvalidEvent(f"{name} is required and must be a non-empty string")
    return value.strip()


def _require_number(record: dict, name: str, minimum: float, strict: bool) -> float:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidEvent(f"{name} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise InvalidEvent(f"{name} must be finite, got {value}")
    if value < minimum or (strict and value == minimum):
        bound = ">" if strict else ">="
        raise InvalidEvent(f"{name} must be {bound} {minimum:g}, got {value:g}")
    return value


def _parse_ts(value) -> datetime:
    if not isinstance(value, str):
        raise InvalidEvent(f"commit_ts must be an ISO-8601 string, got {value!r}")
    try:
        ts = datetime.fromisoformat(value)
    except ValueError as exc:
        raise InvalidEvent(f"commit_ts is not ISO-8601: {value!r}") from exc
    if ts.tzinfo is not None:
        raise InvalidEvent("commit_ts must be naive UTC (no offset), as the source publishes it")
    return ts


def parse_event(record: dict) -> ChangeEvent:
    """Turn one raw record into a ``ChangeEvent`` or raise ``InvalidEvent``."""
    if "_error" in record:
        raise InvalidEvent(record["_error"])
    op = record.get("op")
    if op not in OPS:
        raise InvalidEvent(f"op must be one of {', '.join(OPS)}, got {op!r}")
    lsn = _require_int(record, "lsn")
    commit_ts = _parse_ts(record.get("commit_ts"))
    lane_id = _require_text(record, "lane_id")
    if op == "d":
        return ChangeEvent(lsn=lsn, op=op, commit_ts=commit_ts, lane_id=lane_id)

    currency = _require_text(record, "currency")
    if not CURRENCY_RE.match(currency):
        raise InvalidEvent(f"currency must be a 3-letter ISO code, got {currency!r}")
    return ChangeEvent(
        lsn=lsn,
        op=op,
        commit_ts=commit_ts,
        lane_id=lane_id,
        carrier=_require_text(record, "carrier"),
        origin=_require_text(record, "origin"),
        destination=_require_text(record, "destination"),
        rate_per_kg=_require_number(record, "rate_per_kg", 0.0, strict=True),
        min_charge=_require_number(record, "min_charge", 0.0, strict=False),
        currency=currency,
    )


def validate_batch(records: list[dict]) -> ValidationResult:
    """Split a batch into valid events and (record, reason) rejects.

    Exact redeliveries (same LSN, same content) are collapsed to one event.
    Two different payloads under the same LSN cannot both be right, so both
    copies are rejected rather than guessing which one the source meant.
    """
    result = ValidationResult()
    by_lsn: dict[int, ChangeEvent] = {}
    conflicted: set[int] = set()
    for record in records:
        try:
            event = parse_event(record)
        except InvalidEvent as exc:
            result.rejected.append((record, str(exc)))
            continue
        seen = by_lsn.get(event.lsn)
        if seen is None:
            by_lsn[event.lsn] = event
        elif seen == event:
            result.duplicates += 1
        else:
            conflicted.add(event.lsn)
            result.rejected.append((record, f"lsn {event.lsn} delivered with conflicting payloads"))
    for lsn in conflicted:
        first = by_lsn.pop(lsn)
        result.rejected.append((event_record(first), f"lsn {lsn} delivered with conflicting payloads"))
    result.valid = sorted(by_lsn.values(), key=lambda e: e.lsn)
    return result


def event_record(event: ChangeEvent) -> dict:
    record = {name: getattr(event, name) for name in ("lsn", "op", "lane_id", *ATTRIBUTES)}
    record["commit_ts"] = event.commit_ts.isoformat()
    return record
