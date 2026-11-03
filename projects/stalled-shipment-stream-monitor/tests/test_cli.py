import csv
from pathlib import Path

import pytest

from stalled_shipment_stream_monitor.cli import ALERT_FIELDS, main

SAMPLE = Path(__file__).resolve().parents[1] / "data"


def test_generate_writes_both_files(tmp_path, capsys):
    assert main(["generate", "--out", str(tmp_path), "--shipments", "50", "--seed", "3"]) == 0
    assert (tmp_path / "events.csv").exists() and (tmp_path / "stalls.csv").exists()
    assert "planted stalls" in capsys.readouterr().out


def test_run_prints_summary_and_writes_alerts(tmp_path, capsys):
    out = tmp_path / "alerts.csv"
    assert main(["run", str(SAMPLE / "events.csv"), "--alerts-out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "3473 accepted, 105 duplicates, 29 invalid" in text
    assert "49 stalled, 10 recovered, 0 retracted" in text
    assert "SHP00125" in text  # stall with a dead-lettered DELIVERED scan
    with out.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == list(ALERT_FIELDS)
    assert len(rows) == 59


def test_run_without_lateness_reports_retractions(capsys):
    assert main(["run", str(SAMPLE / "events.csv"), "--lateness", "0"]) == 0
    assert "4 retracted" in capsys.readouterr().out


def test_score_prints_one_row_per_lateness(capsys):
    assert main(["score", str(SAMPLE / "events.csv"), str(SAMPLE / "stalls.csv"),
                 "--lateness", "0", "4"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 4
    assert lines[2].startswith("0 | 58 | 11 | 4")


def test_missing_file_exits_with_error(tmp_path, capsys):
    assert main(["run", str(tmp_path / "missing.csv")]) == 2
    assert capsys.readouterr().err.startswith("error:")


def test_negative_lateness_is_rejected_by_the_parser(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["run", str(SAMPLE / "events.csv"), "--lateness", "-1"])
    assert exc.value.code == 2
    assert "cannot be negative" in capsys.readouterr().err


def test_zero_retention_is_rejected_by_the_parser(capsys):
    with pytest.raises(SystemExit):
        main(["run", str(SAMPLE / "events.csv"), "--retention", "0"])
    assert "must be positive" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["inf", "nan", "-inf"])
def test_non_finite_hours_are_rejected_by_the_parser(value, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["run", str(SAMPLE / "events.csv"), f"--retention={value}"])
    assert exc.value.code == 2
    assert "finite" in capsys.readouterr().err


def test_non_numeric_lateness_names_the_value(capsys):
    with pytest.raises(SystemExit):
        main(["score", str(SAMPLE / "events.csv"), str(SAMPLE / "stalls.csv"), "--lateness", "four"])
    assert "not a number of hours: four" in capsys.readouterr().err
