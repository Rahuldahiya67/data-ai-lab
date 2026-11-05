from datetime import timezone

import pytest

from stalled_shipment_stream_monitor.events import InvalidEvent, Stage, parse_event


def raw(**overrides):
    base = {
        "event_id": "E1",
        "shipment_id": "S1",
        "event_type": "ARRIVED_HUB",
        "event_time": "2026-11-02T08:00:00+05:30",
        "ingest_time": "2026-11-02T08:30:00+05:30",
        "location": "DEL-HUB",
    }
    base.update(overrides)
    return base


def test_valid_event_is_parsed_to_utc_with_stage():
    event = parse_event(raw())
    assert event.event_time.tzinfo == timezone.utc
    assert event.event_time.hour == 2 and event.event_time.minute == 30
    assert event.stage is Stage.AT_HUB


def test_event_type_is_case_insensitive_and_trimmed():
    event = parse_event(raw(event_type=" delivered "))
    assert event.event_type == "DELIVERED"
    assert event.stage is Stage.DELIVERED


def test_z_suffix_is_accepted():
    event = parse_event(raw(event_time="2026-11-02T02:30:00Z", ingest_time="2026-11-02T03:00:00Z"))
    assert event.event_time.hour == 2


@pytest.mark.parametrize("field", ["event_id", "shipment_id", "event_time"])
def test_missing_field_is_rejected(field):
    data = raw()
    del data[field]
    with pytest.raises(InvalidEvent, match="missing fields"):
        parse_event(data)


@pytest.mark.parametrize("field", ["event_id", "shipment_id", "event_type"])
def test_blank_identifier_is_rejected(field):
    with pytest.raises(InvalidEvent, match="empty"):
        parse_event(raw(**{field: "  "}))


def test_non_string_value_is_rejected():
    with pytest.raises(InvalidEvent, match="must be a string"):
        parse_event(raw(location=None))


def test_unknown_event_type_is_rejected():
    with pytest.raises(InvalidEvent, match="unknown event_type"):
        parse_event(raw(event_type="TELEPORTED"))


def test_naive_timestamp_is_rejected():
    with pytest.raises(InvalidEvent, match="no timezone"):
        parse_event(raw(event_time="2026-11-02T08:00:00"))


def test_garbage_timestamp_is_rejected():
    with pytest.raises(InvalidEvent, match="not an ISO 8601"):
        parse_event(raw(ingest_time="yesterday"))


def test_ingest_far_before_event_is_rejected():
    with pytest.raises(InvalidEvent, match="before event_time"):
        parse_event(raw(ingest_time="2026-11-02T07:00:00+05:30"))


def test_small_clock_skew_is_tolerated():
    event = parse_event(raw(ingest_time="2026-11-02T07:57:00+05:30"))
    assert event.ingest_time < event.event_time
