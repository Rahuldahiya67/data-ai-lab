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


def test_late_scan_inside_the_threshold_retracts_the_stall():
    # No lateness, so A is flagged as soon as B's scan moves the watermark.
    # A's departure at 10h is then uploaded late: the gap was 10h < 12h.
    m = monitor(lateness_hours=0)
    m.process(ev("A", "ARRIVED_HUB", 0))
    assert [a.kind for a in m.process(ev("B", "PICKED_UP", 13))] == [AlertKind.STALLED]
    alerts = m.process(ev("A", "DEPARTED_HUB", 10, ingest_delay=3.5))
    assert [a.kind for a in alerts] == [AlertKind.RETRACTED]
    assert m.open_stalls() == []


def test_scan_exactly_at_the_threshold_is_a_retraction():
    m = monitor(lateness_hours=0)
    m.process(ev("A", "ARRIVED_HUB", 0))
    m.process(ev("B", "PICKED_UP", 13))
    alerts = m.process(ev("A", "DEPARTED_HUB", 12, ingest_delay=2))
    assert [a.kind for a in alerts] == [AlertKind.RETRACTED]


def test_scan_after_a_real_gap_is_still_a_recovery():
    m = monitor(lateness_hours=0)
    m.process(ev("A", "ARRIVED_HUB", 0))
    m.process(ev("B", "PICKED_UP", 13))
    alerts = m.process(ev("A", "DEPARTED_HUB", 12.5))
    assert [a.kind for a in alerts] == [AlertKind.RECOVERED]


def test_delivered_shipment_is_evicted_after_the_retention():
    m = StallMonitor(allowed_lateness=timedelta(hours=2), state_retention=timedelta(hours=24))
    m.process(ev("A", "PICKED_UP", 0))
    m.process(ev("A", "DELIVERED", 5))
    m.process(ev("B", "PICKED_UP", 30))  # watermark 28h < 5h + 24h
    assert "A" in m.shipments
    m.process(ev("B", "ARRIVED_HUB", 32))  # watermark 30h > 29h
    assert "A" not in m.shipments
    assert m.stats.evicted == 1


def test_open_shipments_are_never_evicted():
    m = StallMonitor(allowed_lateness=timedelta(hours=2), state_retention=timedelta(hours=1))
    m.process(ev("A", "ARRIVED_HUB", 0))
    for h in range(10, 100, 10):
        m.process(ev("B", "PICKED_UP", h, event_id=f"B{h}"))
    assert "A" in m.shipments
    assert m.open_stalls() == ["A"]


def test_duplicate_within_retention_is_still_caught():
    m = StallMonitor(allowed_lateness=timedelta(hours=2), state_retention=timedelta(hours=24))
    m.process(ev("A", "DELIVERED", 0, event_id="X"))
    m.process(ev("B", "PICKED_UP", 20))
    m.process(ev("A", "DELIVERED", 0, event_id="X", ingest_delay=20))
    assert m.stats.duplicates == 1


def test_event_older_than_retention_is_rejected_not_reopened():
    m = StallMonitor(allowed_lateness=timedelta(hours=2), state_retention=timedelta(hours=24))
    m.process(ev("A", "PICKED_UP", 0, event_id="A1"))
    m.process(ev("A", "DELIVERED", 3, event_id="A2"))
    m.process(ev("B", "PICKED_UP", 40))  # A evicted
    assert "A" not in m.shipments
    # A very late copy of A's pickup must not bring A back as an open shipment.
    alerts = m.process(ev("A", "PICKED_UP", 0, event_id="A1", ingest_delay=41))
    alerts += m.process(ev("B", "ARRIVED_HUB", 60))
    assert alerts == []
    assert "A" not in m.shipments
    assert m.stats.too_late == 1
    assert "behind the watermark" in m.dead_letters[-1].reason


def test_retention_must_be_positive():
    with pytest.raises(ValueError, match="state_retention"):
        StallMonitor(state_retention=timedelta(0))
