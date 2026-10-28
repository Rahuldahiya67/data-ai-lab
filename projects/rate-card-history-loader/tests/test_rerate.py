import csv
from datetime import timedelta

import duckdb
import pytest

from rate_card_history_loader.generator import generate_stream
from rate_card_history_loader.invoices import generate_invoices, write_invoices
from rate_card_history_loader.loader import RateCardLoader
from rate_card_history_loader.rerate import Invoice, read_invoices, rerate, write_report

from conftest import T0, ev

LANE = "L001-DEL-BOM"


def inv(invoice_id, minutes, weight, billed, currency="INR", lane=LANE):
    return Invoice(invoice_id, lane, T0 + timedelta(minutes=minutes), weight, billed, currency)


@pytest.fixture
def priced(loader):
    # 10/kg min 300 from T0, 12/kg from +60 min, withdrawn at +120 min
    loader.load_batch([ev(1, op="c"), ev(2, minutes=60, rate=12.0), ev(3, op="d", minutes=120)], "b1")
    return loader.con


def test_statuses_follow_the_rate_valid_on_the_ship_date(priced):
    report = rerate(
        priced,
        [
            inv("A", 10, 50, 500.0),  # 50 x 10 = 500: ok
            inv("B", 10, 50, 600.0),  # billed at the later 12/kg rate
            inv("C", 70, 50, 500.0),  # billed at the old rate after the change
            inv("D", 10, 12, 300.0),  # minimum charge applies
            inv("E", 130, 50, 600.0),  # lane withdrawn
            inv("F", 10, 50, 500.0, currency="USD"),
            inv("G", -5, 50, 500.0),  # before the lane existed
        ],
    )
    assert [line.status for line in report.lines] == [
        "ok", "overbilled", "underbilled", "ok", "no_rate", "currency_mismatch", "no_rate",
    ]
    assert report.lines[1].difference == 100.0 and report.lines[2].difference == -100.0
    assert report.overbilled_total == 100.0 and report.underbilled_total == 100.0
    assert report.no_rate_billed_total == 1100.0


def test_ship_time_on_a_rate_change_uses_the_new_rate(priced):
    (line,) = rerate(priced, [inv("A", 60, 50, 600.0)]).lines
    assert line.status == "ok" and line.rate_per_kg == 12.0


def test_tolerance_absorbs_rounding_differences(priced):
    report = rerate(priced, [inv("A", 10, 50, 500.004), inv("B", 10, 50, 500.5)], tolerance=0.01)
    assert [line.status for line in report.lines] == ["ok", "overbilled"]
    with pytest.raises(ValueError):
        rerate(priced, [], tolerance=-1)


def test_empty_invoice_list_gives_an_empty_report(priced):
    report = rerate(priced, [])
    assert report.lines == [] and report.summary()["ok"] == 0


def test_read_invoices_reports_bad_rows_with_line_numbers(tmp_path):
    path = tmp_path / "invoices.csv"
    path.write_text(
        "invoice_id,lane_id,ship_ts,weight_kg,billed_amount,currency\n"
        "A,L1,2026-07-01T10:00:00,50,500,INR\n"
        "B,L1,not-a-date,50,500,INR\n"
        "C,L1,2026-07-01T10:00:00,nan,500,INR\n"
        "D,L1,2026-07-01T10:00:00,50,-3,INR\n"
        "A,L1,2026-07-01T11:00:00,50,500,INR\n"
        "E,,2026-07-01T10:00:00,50,500,INR\n"
        "F,L1,2026-07-01T10:00:00+05:30,50,500,INR\n"
    )
    invoices, invalid = read_invoices(path)
    assert [i.invoice_id for i in invoices] == ["A"]
    assert [n for n, _ in invalid] == [3, 4, 5, 6, 7, 8]
    assert "ship_ts is not ISO-8601" in invalid[0][1]
    assert "weight_kg must be a positive finite number" in invalid[1][1]
    assert "duplicate invoice_id A" in invalid[3][1]


def test_report_csv_lists_only_lines_that_need_attention(priced, tmp_path):
    report = rerate(priced, [inv("A", 10, 50, 500.0), inv("B", 10, 50, 600.0), inv("E", 130, 50, 600.0)])
    out = tmp_path / "out" / "report.csv"
    write_report(report, out)
    rows = list(csv.DictReader(out.open()))
    assert [(r["invoice_id"], r["status"]) for r in rows] == [("B", "overbilled"), ("E", "no_rate")]


def test_report_finds_every_planted_billing_error_in_the_synthetic_data(tmp_path):
    stream = generate_stream(seed=7)
    con = duckdb.connect()
    loader = RateCardLoader(con)
    for i, batch in enumerate(stream.batches, 1):
        loader.load_batch(batch, f"batch_{i}")
    planted = generate_invoices(stream.history, seed=21)
    path = tmp_path / "invoices.csv"
    write_invoices(planted, path)
    invoices, invalid = read_invoices(path)
    report = rerate(con, invoices, invalid=invalid)
    expected = {"": {"ok"}, "stale_rate": {"overbilled", "underbilled"}, "no_contract": {"no_rate"}}
    for truth, line in zip(planted, report.lines):
        assert line.status in expected[truth["expected_issue"]], truth
    assert report.summary()["invalid"] == 0
    con.close()
