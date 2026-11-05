from collections import Counter
from datetime import datetime, timedelta

import pytest

from stalled_shipment_stream_monitor.events import InvalidEvent, parse_event
from stalled_shipment_stream_monitor.generator import (
    NORMAL_GAP_HOURS,
    GeneratorConfig,
    generate,
    write_dataset,
)


@pytest.fixture(scope="module")
def dataset():
    return generate(GeneratorConfig(shipments=400, seed=11))


def test_generation_is_deterministic():
    cfg = GeneratorConfig(shipments=50, seed=3)
    assert generate(cfg) == generate(cfg)


def test_stream_is_in_ingest_order(dataset):
    events, _ = dataset
    ingest = [e["ingest_time"] for e in events]
    assert ingest == sorted(ingest)


def test_stream_has_late_duplicate_and_malformed_events(dataset):
    events, _ = dataset
    counts = Counter(e["event_id"] for e in events)
    assert any(c > 1 for c in counts.values()), "expected duplicate deliveries"

    bad, late = 0, 0
    for raw in events:
        try:
            ev = parse_event(raw)
        except InvalidEvent:
            bad += 1
            continue
        if ev.ingest_time - ev.event_time > timedelta(hours=1):
            late += 1
    assert bad > 0
    assert late > 0


def test_planted_stalls_exist_with_both_outcomes(dataset):
    _, stalls = dataset
    assert stalls
    assert any(s["resumed_at"] for s in stalls)
    assert any(not s["resumed_at"] for s in stalls)


def test_normal_gaps_stay_below_stall_gaps(dataset):
    events, stalls = dataset
    stalled = {s["shipment_id"] for s in stalls}
    by_shipment = {}
    for raw in events:
        try:
            ev = parse_event(raw)
        except InvalidEvent:
            continue
        by_shipment.setdefault(ev.shipment_id, {})[ev.event_id] = ev
    longest_normal = max(hi for _, hi in NORMAL_GAP_HOURS.values())
    for shipment_id, evs in by_shipment.items():
        if shipment_id in stalled:
            continue
        times = sorted(e.event_time for e in evs.values())
        for a, b in zip(times, times[1:]):
            # Malformed rows can remove a scan, which can double one gap at most.
            assert (b - a) <= timedelta(hours=2 * longest_normal)


def test_resumed_stall_has_a_later_event(dataset):
    events, stalls = dataset
    times = {}
    for raw in events:
        try:
            ev = parse_event(raw)
        except InvalidEvent:
            continue
        times.setdefault(ev.shipment_id, []).append(ev.event_time)
    for stall in stalls:
        last = datetime.fromisoformat(stall["last_event_time"])
        later = [t for t in times.get(stall["shipment_id"], []) if t > last]
        if stall["resumed_at"]:
            # The resumed scan may itself be malformed or arrive after the horizon.
            assert all(t >= datetime.fromisoformat(stall["resumed_at"]) for t in later)
        else:
            assert later == []


@pytest.mark.parametrize("field,value", [("shipments", 0), ("days", 0), ("stall_rate", 1.5), ("late_rate", -0.1)])
def test_config_rejects_bad_values(field, value):
    with pytest.raises(ValueError):
        GeneratorConfig(**{field: value})


def test_write_dataset(tmp_path):
    n_events, n_stalls = write_dataset(tmp_path, GeneratorConfig(shipments=30, seed=1))
    lines = (tmp_path / "events.csv").read_text().splitlines()
    assert len(lines) == n_events + 1
    assert lines[0].startswith("event_id,shipment_id,event_type")
    assert len((tmp_path / "stalls.csv").read_text().splitlines()) == n_stalls + 1
