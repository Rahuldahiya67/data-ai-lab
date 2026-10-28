import math

import pytest

from rate_card_history_loader.validation import InvalidEvent, parse_event, validate_batch

from conftest import ev


def test_parses_create_and_delete():
    create = parse_event(ev(1, op="c"))
    assert create.rate_per_kg == 10.0 and create.currency == "INR"
    delete = parse_event(ev(2, op="d"))
    assert delete.op == "d" and delete.rate_per_kg is None


@pytest.mark.parametrize(
    "change, message",
    [
        ({"op": "x"}, "op must be one of"),
        ({"lsn": 0}, "lsn must be positive"),
        ({"lsn": True}, "lsn must be an integer"),
        ({"lsn": "12"}, "lsn must be an integer"),
        ({"commit_ts": "yesterday"}, "not ISO-8601"),
        ({"commit_ts": "2026-07-01T09:00:00+05:30"}, "naive UTC"),
        ({"lane_id": "  "}, "lane_id is required"),
        ({"rate_per_kg": -2.0}, "rate_per_kg must be > 0"),
        ({"rate_per_kg": 0}, "rate_per_kg must be > 0"),
        ({"rate_per_kg": math.nan}, "must be finite"),
        ({"rate_per_kg": "12.5"}, "rate_per_kg must be a number"),
        ({"min_charge": -1}, "min_charge must be >= 0"),
        ({"currency": "rupees"}, "3-letter ISO code"),
        ({"carrier": None}, "carrier is required"),
    ],
)
def test_rejects_bad_fields(change, message):
    with pytest.raises(InvalidEvent, match=message):
        parse_event({**ev(5, op="u"), **change})


def test_missing_commit_ts_is_rejected():
    record = ev(5)
    del record["commit_ts"]
    with pytest.raises(InvalidEvent, match="commit_ts"):
        parse_event(record)


def test_zero_min_charge_is_allowed():
    assert parse_event(ev(5, min_charge=0)).min_charge == 0.0


def test_unparseable_line_is_quarantined_not_raised():
    result = validate_batch([{"_raw": "{oops", "_error": "line 3: invalid JSON"}, ev(1, op="c")])
    assert [e.lsn for e in result.valid] == [1]
    assert result.rejected[0][1] == "line 3: invalid JSON"


def test_exact_redelivery_is_collapsed_and_output_sorted_by_lsn():
    result = validate_batch([ev(3, minutes=3), ev(1, op="c"), ev(3, minutes=3), ev(2, minutes=2)])
    assert [e.lsn for e in result.valid] == [1, 2, 3]
    assert result.rejected == []


def test_conflicting_payloads_for_one_lsn_reject_every_copy():
    result = validate_batch([ev(4, rate=10.0), ev(4, rate=11.0), ev(5)])
    assert [e.lsn for e in result.valid] == [5]
    assert len(result.rejected) == 2
    assert all("conflicting payloads" in reason for _, reason in result.rejected)
