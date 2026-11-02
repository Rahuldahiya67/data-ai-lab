from datetime import datetime, timedelta, timezone

import pytest

from stalled_shipment_stream_monitor.events import Stage, TrackingEvent
from stalled_shipment_stream_monitor.monitor import AlertKind, StallMonitor

T0 = datetime(2026, 11, 2, tzinfo=timezone.utc)
_counter = iter(range(1, 10**6))


def ev(shipment, event_type, hours, ingest_delay=0.1, event_id=None, location="DEL-HUB"):
    t = T0 + timedelta(hours=hours)
    return TrackingEvent(
        event_id=event_id or f"E{next(_counter)}",
        shipment_id=shipment,
        event_type=event_type,
        event_time=t,
        ingest_time=t + timedelta(hours=ingest_delay),
        location=location,
    )


def monitor(lateness_hours=2):
    return StallMonitor(allowed_lateness=timedelta(hours=lateness_hours))


def test_shipment_at_hub_past_threshold_is_flagged_once():
    m = monitor()
    m.process(ev("A", "ARRIVED_HUB", 0))
    # Another shipment's scans move the watermark forward.
    assert m.process(ev("B", "PICKED_UP", 13)) == []  # watermark 11h < 12h
    alerts = m.process(ev("B", "ARRIVED_HUB", 14.5))  # watermark 12.5h > 12h
    assert [(a.kind, a.shipment_id, a.stage) for a in alerts] == [(AlertKind.STALLED, "A", Stage.AT_HUB)]
    assert alerts[0].last_event_time == T0
    assert m.process(ev("B", "DEPARTED_HUB", 20)) == []
    assert m.open_stalls() == ["A"]


def test_newer_scan_recovers_a_stalled_shipment():
    m = monitor()
    m.process(ev("A", "ARRIVED_HUB", 0))
    m.process(ev("B", "PICKED_UP", 15))
    alerts = m.process(ev("A", "DEPARTED_HUB", 16))
    assert [a.kind for a in alerts] == [AlertKind.RECOVERED]
    assert alerts[0].stage is Stage.IN_TRANSIT
    assert m.open_stalls() == []


def test_regular_scans_never_alert():
    m = monitor()
    alerts = []
    for i, etype in enumerate(["PICKED_UP", "ARRIVED_HUB", "DEPARTED_HUB", "OUT_FOR_DELIVERY", "DELIVERED"]):
        alerts += m.process(ev("A", etype, i * 5))
    alerts += m.process(ev("B", "PICKED_UP", 200))
    assert alerts == []


def test_delivered_shipment_is_never_flagged():
    m = monitor()
    m.process(ev("A", "DELIVERED", 0))
    assert m.process(ev("B", "PICKED_UP", 500)) == []


def test_duplicate_event_id_is_ignored():
    m = monitor()
    first = ev("A", "ARRIVED_HUB", 0, event_id="X")
    m.process(first)
    m.process(ev("A", "ARRIVED_HUB", 0, event_id="X", ingest_delay=1.0))
    assert m.stats.duplicates == 1
    assert m.stats.accepted == 1


def test_late_scan_within_lateness_prevents_a_false_stall():
    # A's departure scan is uploaded 3 hours late. Without lateness the monitor
    # would flag A before the late scan arrives.
    m = monitor(lateness_hours=4)
    m.process(ev("A", "ARRIVED_HUB", 0))
    m.process(ev("B", "PICKED_UP", 15))  # watermark 11h: A not yet due
    alerts = m.process(ev("A", "DEPARTED_HUB", 11.5, ingest_delay=3.5))
    alerts += m.process(ev("B", "ARRIVED_HUB", 16))  # watermark 12h
    assert alerts == []


def test_without_lateness_the_same_stream_raises_a_false_stall():
    m = monitor(lateness_hours=0)
    m.process(ev("A", "ARRIVED_HUB", 0))
    alerts = m.process(ev("B", "PICKED_UP", 15))
    assert [a.kind for a in alerts] == [AlertKind.STALLED]


def test_older_out_of_order_scan_does_not_move_stage_back():
    m = monitor()
    m.process(ev("A", "DEPARTED_HUB", 5))
    m.process(ev("A", "ARRIVED_HUB", 1, ingest_delay=5))
    assert m.shipments["A"].stage is Stage.IN_TRANSIT
    assert m.stats.out_of_order == 1


def test_late_counter_counts_scans_behind_the_watermark():
    m = monitor(lateness_hours=1)
    m.process(ev("B", "PICKED_UP", 10))
    m.process(ev("A", "PICKED_UP", 2, ingest_delay=8))
    assert m.stats.late == 1


def test_invalid_raw_event_goes_to_dead_letters():
    m = monitor()
    alerts = m.process_raw({"event_id": "E1"})
    assert alerts == []
    assert m.stats.invalid == 1
    assert "missing fields" in m.dead_letters[0].reason


def test_run_processes_raw_rows():
    m = monitor()
    rows = [
        {"event_id": "1", "shipment_id": "A", "event_type": "ARRIVED_HUB",
         "event_time": "2026-11-02T00:00:00+00:00", "ingest_time": "2026-11-02T00:05:00+00:00", "location": "H"},
        {"event_id": "2", "shipment_id": "B", "event_type": "PICKED_UP",
         "event_time": "2026-11-02T20:00:00+00:00", "ingest_time": "2026-11-02T20:05:00+00:00", "location": "C"},
    ]
    alerts = m.run(rows)
    assert [(a.kind, a.shipment_id) for a in alerts] == [(AlertKind.STALLED, "A")]


def test_threshold_must_cover_every_open_stage():
    with pytest.raises(ValueError, match="no stall threshold"):
        StallMonitor(thresholds={Stage.AT_HUB: timedelta(hours=1)})


def test_threshold_must_be_positive():
    thresholds = {s: timedelta(hours=1) for s in Stage if s is not Stage.DELIVERED}
    thresholds[Stage.AT_HUB] = timedelta(0)
    with pytest.raises(ValueError, match="must be positive"):
        StallMonitor(thresholds=thresholds)


def test_negative_lateness_is_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        StallMonitor(allowed_lateness=timedelta(hours=-1))
