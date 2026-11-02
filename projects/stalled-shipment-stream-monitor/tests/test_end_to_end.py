from pathlib import Path

import pytest

from stalled_shipment_stream_monitor.generator import GeneratorConfig, write_dataset
from stalled_shipment_stream_monitor.monitor import AlertKind, StallMonitor
from stalled_shipment_stream_monitor.stream import read_events

SAMPLE = Path(__file__).resolve().parents[1] / "data"


def flagged(alerts):
    return {a.shipment_id for a in alerts if a.kind is AlertKind.STALLED}


def test_replay_of_generated_stream_finds_most_planted_stalls(tmp_path):
    write_dataset(tmp_path, GeneratorConfig(shipments=300, seed=5))
    monitor = StallMonitor()
    alerts = monitor.run(read_events(tmp_path / "events.csv"))
    truth = {row["shipment_id"] for row in read_events_stalls(tmp_path / "stalls.csv")}
    found = flagged(alerts)
    assert truth, "generator planted no stalls"
    assert len(found & truth) / len(truth) >= 0.8
    assert monitor.stats.invalid == len(monitor.dead_letters) > 0
    assert monitor.stats.duplicates > 0


def test_sample_data_is_the_default_generator_output(tmp_path):
    write_dataset(tmp_path)
    for name in ("events.csv", "stalls.csv"):
        assert (tmp_path / name).read_text() == (SAMPLE / name).read_text(), name


def test_reader_rejects_a_file_without_required_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("event_id,shipment_id\n1,A\n")
    with pytest.raises(ValueError, match="missing columns"):
        list(read_events(path))


def test_reader_passes_short_rows_to_the_dead_letter_list(tmp_path):
    path = tmp_path / "short.csv"
    path.write_text("event_id,shipment_id,event_type,event_time,ingest_time,location\n1,A\n")
    monitor = StallMonitor()
    monitor.run(read_events(path))
    assert monitor.stats.invalid == 1


def read_events_stalls(path):
    import csv

    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))
