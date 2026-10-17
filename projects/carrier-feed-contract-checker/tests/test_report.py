import json
from datetime import datetime, timezone

import pandas as pd

from carrier_feed_contract_checker.report import build_report, csv_line, render_markdown
from carrier_feed_contract_checker.validate import validate_frame

FIXED_TIME = datetime(2026, 10, 16, 9, 0, tzinfo=timezone.utc)


def test_csv_line_counts_header():
    assert csv_line(0) == 2


def test_report_counts_match_result(contract, make_frame):
    df = make_frame({}, {"carrier": "FEDEX"}, {"weight_kg": "-1", "status": "DLVRD"})
    result = validate_frame(df, contract)
    report = build_report(result, contract, "feed.csv", generated_at=FIXED_TIME)

    assert report["rows"] == {"total": 3, "accepted": 1, "quarantined": 2, "reject_rate": 0.6667}
    assert report["verdict"] == "FAIL"  # 3 rows is below min_rows and over the reject limit
    rules = {v["rule"]: v for v in report["violations_by_rule"]}
    assert rules["carrier.allowed"] == {"rule": "carrier.allowed", "column": "carrier", "count": 1}
    assert report["sample_quarantined_rows"][1] == {
        "csv_line": 4,
        "violations": ["status.allowed", "weight_kg.min"],
    }
    json.dumps(report)  # must be serialisable as-is


def test_sample_size_limits_rows(contract, make_frame):
    df = make_frame(*[{"carrier": "FEDEX"}] * 5)
    report = build_report(validate_frame(df, contract), contract, "f", sample_size=2)
    assert len(report["sample_quarantined_rows"]) == 2


def test_markdown_contains_verdict_and_tables(contract, make_frame):
    result = validate_frame(make_frame({}, {"carrier": "FEDEX"}), contract)
    md = render_markdown(build_report(result, contract, "feed.csv", generated_at=FIXED_TIME))
    assert md.startswith("# Feed quality report: feed.csv")
    assert "Verdict: **FAIL**" in md
    assert "| carrier.allowed | carrier | 1 |" in md
    assert "| 3 | carrier.allowed |" in md


def test_markdown_for_clean_feed(contract, make_frame):
    result = validate_frame(make_frame({}), contract)
    md = render_markdown(build_report(result, contract, "f", generated_at=FIXED_TIME))
    assert "No row-level violations." in md
    assert "Sample quarantined rows" not in md
