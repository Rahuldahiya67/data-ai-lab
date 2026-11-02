import csv
import io
import json
from datetime import date

import pytest

from dispatch_metrics_layer.cli import UsageError, format_value, main, parse_where
from dispatch_metrics_layer.compiler import Filter, MetricRequest

from conftest import SPEC_PATH


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), out=out, err=err)
    return code, out.getvalue(), err.getvalue()


@pytest.mark.parametrize(
    "text, expected",
    [
        ("carrier = Metro Parcel", Filter("carrier", "=", "Metro Parcel")),
        ("status!=cancelled", Filter("status", "!=", "cancelled")),
        ("region in North, South", Filter("region", "in", ("North", "South"))),
        ("mode not_in PARCEL", Filter("mode", "not_in", ("PARCEL",))),
        ("origin_city >= M", Filter("origin_city", ">=", "M")),
    ],
)
def test_parse_where(text, expected):
    assert parse_where(text) == expected


@pytest.mark.parametrize(
    "text, message",
    [
        ("carrier ~ Metro", "expected 'dimension OP value'"),
        ("carrier =", "missing value after '='"),
        ("region in , ,", "in needs at least one value"),
    ],
)
def test_parse_where_errors(text, message):
    with pytest.raises(UsageError, match=message):
        parse_where(text)


def test_format_value():
    assert format_value(0.8456, "percent") == "84.6%"
    assert format_value(1234.5, "currency") == "1,234.50"
    assert format_value(4000, "number") == "4,000"
    assert format_value(None, "percent") == "-"
    assert format_value(date(2026, 8, 1), "number") == "2026-08-01"


def test_query_json_matches_engine(engine, data_dir):
    code, out, err = run(
        "query", "--spec", str(SPEC_PATH), "--data", str(data_dir),
        "-m", "on_time_rate,late_rate", "--by", "region", "--where", "mode != PARCEL",
        "--format", "json",
    )
    assert code == 0, err
    expected = engine.query(
        MetricRequest(["on_time_rate", "late_rate"], ["region"], [Filter("mode", "!=", "PARCEL")])
    ).records()
    assert json.loads(out) == expected


def test_query_csv_with_grain_and_dates(engine, data_dir):
    code, out, err = run(
        "query", "--spec", str(SPEC_PATH), "--data", str(data_dir),
        "-m", "shipment_count", "--grain", "month", "--start", "2026-08-01", "--end", "2026-09-01",
        "--format", "csv",
    )
    assert code == 0, err
    rows = list(csv.reader(io.StringIO(out)))
    (expected,) = engine.query(
        MetricRequest(["shipment_count"], grain="month", start=date(2026, 8, 1), end=date(2026, 9, 1))
    ).rows
    assert rows == [["period", "shipment_count"], ["2026-08-01", str(expected[1])]]


def test_query_table_formats_by_metric_format(data_dir):
    code, out, _ = run(
        "query", "--spec", str(SPEC_PATH), "--data", str(data_dir),
        "-m", "otif_rate", "-m", "cost_per_km,shipment_count", "--by", "carrier",
    )
    assert code == 0
    header, rule, first, *rest = out.splitlines()
    assert header.split() == ["carrier", "otif_rate", "cost_per_km", "shipment_count"]
    assert set(rule) <= {"-", " "}
    assert first.split()[-3].endswith("%")  # otif_rate as a percent
    assert "." in first.split()[-2]  # cost_per_km with two decimals
    assert len(rest) == 5


def test_query_sql_prints_query_and_params(data_dir):
    code, out, _ = run(
        "query", "--spec", str(SPEC_PATH), "--data", str(data_dir),
        "-m", "otif_rate", "--where", "region in North,South", "--sql",
    )
    assert code == 0
    assert 'IN (?, ?)' in out
    assert out.rstrip().endswith("-- params: ['North', 'South']")


def test_request_errors_exit_2(data_dir):
    code, _, err = run("query", "--spec", str(SPEC_PATH), "--data", str(data_dir), "-m", "nope")
    assert code == 2 and "unknown metric 'nope'" in err
    code, _, err = run(
        "query", "--spec", str(SPEC_PATH), "--data", str(data_dir), "-m", "otif_rate", "--start", "1 Aug",
    )
    assert code == 2 and "expected a date as YYYY-MM-DD" in err


def test_data_errors_exit_1(tmp_path):
    code, _, err = run("check", "--spec", str(SPEC_PATH), "--data", str(tmp_path))
    assert code == 1 and "missing data file" in err
    bad = tmp_path / "bad.yaml"
    bad.write_text("fact_table: shipments\n", encoding="utf-8")
    code, _, err = run("list", "--spec", str(bad))
    assert code == 1 and "invalid metrics spec" in err


def test_list_and_check(data_dir):
    code, out, _ = run("list", "--spec", str(SPEC_PATH))
    assert code == 0
    assert "late_rate                  derived  percent  = 1 - on_time_rate" in out
    assert "carrier                    carriers.carrier_name" in out
    code, out, _ = run("check", "--spec", str(SPEC_PATH), "--data", str(data_dir))
    assert code == 0
    assert "shipments: 1,500 rows" in out and "spec OK: 14 metrics" in out
    assert "warning" not in out


def test_generate_then_check(tmp_path):
    code, out, _ = run("generate", "--out", str(tmp_path), "--shipments", "120", "--seed", "3")
    assert code == 0 and "(120 rows)" in out
    code, out, _ = run("check", "--spec", str(SPEC_PATH), "--data", str(tmp_path))
    assert code == 0 and "shipments: 120 rows" in out
    code, _, err = run("generate", "--out", str(tmp_path), "--shipments", "0")
    assert code == 2 and "n_shipments must be at least 1" in err


def test_format_gap():
    from dispatch_metrics_layer.cli import format_gap

    assert format_gap(-0.0162, "percent") == "-1.6 pts"
    assert format_gap(-0.00004, "percent") == "+0.0 pts"
    assert format_gap(2.5, "currency") == "+2.50"
    assert format_gap(None, "percent") == "-"


def test_reconcile_command(engine, data_dir):
    from dispatch_metrics_layer.reconcile import reconcile

    code, out, err = run(
        "reconcile", "--spec", str(SPEC_PATH), "--data", str(data_dir),
        "-m", "on_time_rate,on_time_rate_dispatched", "--by", "carrier",
    )
    assert code == 0, err
    report = reconcile(engine, ["on_time_rate", "on_time_rate_dispatched"], by="carrier")
    gap = report.gap_to_baseline("on_time_rate_dispatched") * 100
    assert f"{report.overall['on_time_rate'] * 100:.1f}%" in out
    assert f"{gap:+.1f} pts" in out
    assert "By carrier" in out and "Pooled rate vs unweighted average of carrier rates" in out
    assert len([line for line in out.splitlines() if line.startswith(("Coastal", "Deccan", "Ganga", "Metro", "Northline", "Swift"))]) == 6


def test_reconcile_applies_filters(engine, data_dir):
    from dispatch_metrics_layer.reconcile import reconcile

    code, out, _ = run(
        "reconcile", "--spec", str(SPEC_PATH), "--data", str(data_dir),
        "-m", "on_time_rate,on_time_rate_units", "--where", "mode = FTL",
    )
    report = reconcile(
        engine, ["on_time_rate", "on_time_rate_units"],
        base=MetricRequest(["on_time_rate"], filters=[Filter("mode", "=", "FTL")]),
    )
    assert code == 0
    assert f"{report.overall['on_time_rate_units'] * 100:.1f}%" in out
    assert "By " not in out


@pytest.mark.parametrize(
    "argv, message",
    [
        (["-m", "on_time_rate"], "at least two metrics"),
        (["-m", "on_time_rate,otif_rate", "--by", "carrier,region"], "one dimension"),
        (["-m", "on_time_rate,otif_rate", "--by", "planet"], "unknown dimension"),
    ],
)
def test_reconcile_request_errors(data_dir, argv, message):
    code, _, err = run("reconcile", "--spec", str(SPEC_PATH), "--data", str(data_dir), *argv)
    assert code == 2 and message in err
