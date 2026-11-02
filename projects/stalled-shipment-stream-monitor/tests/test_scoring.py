from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from stalled_shipment_stream_monitor.events import Stage
from stalled_shipment_stream_monitor.monitor import DEFAULT_THRESHOLDS, Alert, AlertKind
from stalled_shipment_stream_monitor.scoring import (
    PlantedStall,
    format_table,
    load_truth,
    score,
    sweep,
)

SAMPLE = Path(__file__).resolve().parents[1] / "data"
T0 = datetime(2026, 11, 2, tzinfo=timezone.utc)
H = timedelta(hours=1)


def alert(kind, shipment, last_h, raised_h, stage=Stage.AT_HUB, dead_letters=0):
    return Alert(kind=kind, shipment_id=shipment, stage=stage, location="X",
                 last_event_time=T0 + last_h * H, watermark=None,
                 raised_at_ingest=T0 + raised_h * H, dead_letters=dead_letters)


def planted(shipment, last_h, stage=Stage.AT_HUB):
    return PlantedStall(shipment, stage, T0 + last_h * H, None)


def test_score_counts_true_false_and_missed_stalls():
    truth = [planted("A", 0), planted("B", 0), planted("C", 0)]
    alerts = [
        alert(AlertKind.STALLED, "A", 0, 15),       # 3h after the 12h threshold
        alert(AlertKind.STALLED, "B", 0, 13),       # 1h after
        alert(AlertKind.STALLED, "X", 0, 13),       # not planted
        alert(AlertKind.STALLED, "A", 0, 40),       # repeat for A counts once
    ]
    s = score(alerts, truth, DEFAULT_THRESHOLDS, T0 + 100 * H, 4 * H)
    assert (s.alerts, s.true_alerts, s.false_alerts) == (3, 2, 1)
    assert (s.due, s.detected, s.not_due) == (3, 2, 0)
    assert s.precision == pytest.approx(2 / 3)
    assert s.recall == pytest.approx(2 / 3)
    assert sorted(s.delay_hours) == pytest.approx([1.0, 3.0])
    assert s.median_delay_hours == pytest.approx(2.0)


def test_stall_not_confirmable_before_the_stream_ends_is_not_due():
    truth = [planted("A", 0), planted("B", 90)]  # B due at 102h, stream ends at 100h
    alerts = [alert(AlertKind.STALLED, "A", 0, 13)]
    s = score(alerts, truth, DEFAULT_THRESHOLDS, T0 + 100 * H, 4 * H)
    assert (s.due, s.not_due, s.recall) == (1, 1, 1.0)


def test_due_uses_the_threshold_of_the_stalled_stage():
    # IN_TRANSIT allows 14h, PICKED_UP 8h; both last scanned at 0h, watermark 10h.
    truth = [planted("A", 0, Stage.IN_TRANSIT), planted("B", 0, Stage.PICKED_UP)]
    s = score([], truth, DEFAULT_THRESHOLDS, T0 + 10 * H, 0 * H)
    assert (s.due, s.not_due) == (1, 1)


def test_false_alerts_are_split_by_cause():
    alerts = [
        alert(AlertKind.STALLED, "X", 0, 13),
        alert(AlertKind.RETRACTED, "X", 10, 14),
        alert(AlertKind.STALLED, "Y", 0, 13, dead_letters=1),
        alert(AlertKind.STALLED, "Z", 0, 13),
    ]
    s = score(alerts, [], DEFAULT_THRESHOLDS, T0 + 50 * H, 0 * H)
    assert (s.false_alerts, s.retracted, s.false_with_dead_letters) == (3, 1, 1)
    assert s.precision == 0.0


def test_empty_inputs_do_not_divide_by_zero():
    s = score([], [], DEFAULT_THRESHOLDS, None, 0 * H)
    assert (s.precision, s.recall, s.median_delay_hours, s.p90_delay_hours) == (0.0, 0.0, None, None)


def test_p90_is_nearest_rank():
    truth = [planted(f"S{i}", 0) for i in range(10)]
    alerts = [alert(AlertKind.STALLED, f"S{i}", 0, 12 + i + 1) for i in range(10)]
    s = score(alerts, truth, DEFAULT_THRESHOLDS, T0 + 100 * H, 0 * H)
    assert s.p90_delay_hours == pytest.approx(9.0)


def test_load_truth_reads_the_generator_file():
    truth = load_truth(SAMPLE / "stalls.csv")
    assert len(truth) == 47
    resumed = [s for s in truth if s.resumed_at is not None]
    assert resumed and all(s.resumed_at > s.last_event_time for s in resumed)


def test_load_truth_rejects_unknown_stage_and_duplicates(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("shipment_id,stalled_after,last_event_time,resumed_at\n"
                   "A,LOST,2026-11-02T00:00:00+00:00,\n")
    with pytest.raises(ValueError, match="unknown stalled_after"):
        load_truth(bad)
    dup = tmp_path / "dup.csv"
    dup.write_text("shipment_id,stalled_after,last_event_time,resumed_at\n"
                   "A,PICKED_UP,2026-11-02T00:00:00+00:00,\n"
                   "A,ARRIVED_HUB,2026-11-02T05:00:00+00:00,\n")
    with pytest.raises(ValueError, match="more than one planted stall"):
        load_truth(dup)


def test_sweep_on_sample_shows_the_lateness_trade_off():
    scores = sweep(SAMPLE / "events.csv", SAMPLE / "stalls.csv", [0, 4])
    no_lateness, four_hours = scores
    assert no_lateness.false_alerts > four_hours.false_alerts
    assert no_lateness.retracted > 0
    assert four_hours.median_delay_hours > no_lateness.median_delay_hours
    assert four_hours.recall == 1.0
    table = format_table(scores)
    assert table.splitlines()[0].startswith("lateness_h")
    assert len(table.splitlines()) == 4


def test_sweep_needs_a_lateness_value():
    with pytest.raises(ValueError, match="at least one"):
        sweep(SAMPLE / "events.csv", SAMPLE / "stalls.csv", [])
