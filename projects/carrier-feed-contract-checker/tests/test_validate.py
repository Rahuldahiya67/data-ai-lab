import pandas as pd
import pytest

from carrier_feed_contract_checker.contract import parse_contract
from carrier_feed_contract_checker.generate import FeedSpec, generate_feed, write_feed
from carrier_feed_contract_checker.validate import validate_file, validate_frame


def rules_for(result, row=0):
    return set(result.quarantined.loc[row, "_violations"].split("; "))


def test_clean_rows_are_accepted(contract, make_frame):
    result = validate_frame(make_frame({}, {}, {"status": "IN_TRANSIT", "delivered_at": ""}), contract)
    assert len(result.accepted) == 3
    assert result.quarantined.empty
    assert result.violations == []


@pytest.mark.parametrize(
    "override, expected_rule",
    [
        ({"tracking_id": ""}, "tracking_id.required"),
        ({"tracking_id": "blu-0000001"}, "tracking_id.regex"),
        ({"carrier": "FEDEX"}, "carrier.allowed"),
        ({"status": "DLVRD"}, "status.allowed"),
        ({"weight_kg": "-3"}, "weight_kg.min"),
        ({"weight_kg": "9000"}, "weight_kg.max"),
        ({"weight_kg": "heavy"}, "weight_kg.type_float"),
        ({"picked_up_at": "12/10/2026"}, "picked_up_at.type_datetime"),
        ({"delivered_at": "2026-10-11 08:00:00"}, "delivered_after_pickup"),
        ({"delivered_at": ""}, "delivered_needs_timestamp"),
    ],
)
def test_each_rule_catches_its_defect(contract, make_frame, override, expected_rule):
    result = validate_frame(make_frame({}, override), contract)
    assert len(result.accepted) == 1
    assert len(result.quarantined) == 1
    assert expected_rule in rules_for(result, row=1)


def test_duplicate_keeps_first_and_quarantines_repeat(contract, make_frame):
    df = make_frame({}, {})
    df.loc[1, "tracking_id"] = df.loc[0, "tracking_id"]
    result = validate_frame(df, contract)
    assert list(result.accepted.index) == [0]
    assert rules_for(result, row=1) == {"tracking_id.unique"}


def test_row_breaking_several_rules_lists_all_of_them(contract, make_frame):
    result = validate_frame(make_frame({"carrier": "FEDEX", "weight_kg": "-1"}), contract)
    assert rules_for(result) == {"carrier.allowed", "weight_kg.min"}
    assert len(result.violations) == 2


def test_blank_optional_value_is_not_a_type_error(contract, make_frame):
    result = validate_frame(make_frame({"status": "PICKED_UP", "delivered_at": ""}), contract)
    assert result.quarantined.empty


def test_missing_column_quarantines_everything(contract, make_frame):
    df = make_frame({}, {}).drop(columns=["weight_kg"])
    result = validate_frame(df, contract)
    assert result.accepted.empty
    assert len(result.quarantined) == 2
    assert not result.passed
    assert "weight_kg" in result.feed_checks[0].detail


def test_extra_column_is_reported_but_not_fatal(contract, make_frame):
    df = make_frame({}).assign(driver_phone="99999")
    result = validate_frame(df, contract)
    assert len(result.accepted) == 1
    assert any(c.name == "schema.no_extra_columns" for c in result.feed_checks)


def test_feed_fails_when_reject_rate_exceeds_limit():
    contract = parse_contract(
        {"columns": {"x": {"type": "int", "min": 0}}, "feed": {"max_reject_rate": 0.25}}
    )
    df = pd.DataFrame({"x": ["1", "2", "-1", "-2"]})
    result = validate_frame(df, contract)
    assert result.reject_rate == 0.5
    assert not result.passed
    failing = [c.name for c in result.feed_checks if not c.passed]
    assert failing == ["feed.max_reject_rate"]


def test_feed_fails_below_min_rows(contract, make_frame):
    result = validate_frame(make_frame({}), contract)
    assert not next(c for c in result.feed_checks if c.name == "feed.min_rows").passed


def test_matches_ground_truth_on_synthetic_feed(contract, tmp_path):
    """Every injected defect is caught and no clean row is rejected."""
    path = tmp_path / "feed.csv"
    labelled = write_feed(str(path), FeedSpec(rows=3000, defect_rate=0.08, seed=11))
    result = validate_file(path, contract)
    truth = set(labelled.index[labelled["_defect"] != ""])
    assert set(result.quarantined.index) == truth
    assert result.passed


def test_violation_counts_sorted_descending(contract):
    df = generate_feed(FeedSpec(rows=1500, seed=3)).drop(columns="_defect")
    counts = list(validate_frame(df, contract).violation_counts().values())
    assert counts == sorted(counts, reverse=True)


@pytest.mark.parametrize("missing", [None, float("nan"), pd.NA])
def test_missing_value_in_required_column_is_quarantined(contract, make_frame, missing):
    df = make_frame({}, {})
    df["origin_city"] = df["origin_city"].astype(object)
    df.loc[1, "origin_city"] = missing
    result = validate_frame(df, contract)
    assert list(result.accepted.index) == [0]
    assert rules_for(result, row=1) == {"origin_city.required"}


def test_missing_optional_datetime_is_not_a_type_error(contract, make_frame):
    df = make_frame({"status": "IN_TRANSIT"}, {"status": "DELIVERED"})
    df["delivered_at"] = None
    result = validate_frame(df, contract)
    assert list(result.accepted.index) == [0]
    assert rules_for(result, row=1) == {"delivered_needs_timestamp"}


def test_default_read_csv_gives_same_split_as_read_feed(contract, tmp_path):
    path = tmp_path / "feed.csv"
    write_feed(str(path), FeedSpec(rows=1000, defect_rate=0.08, seed=5))
    strict = validate_file(path, contract)
    loose = validate_frame(pd.read_csv(path), contract)
    assert set(loose.quarantined.index) == set(strict.quarantined.index)
    assert loose.violation_counts() == strict.violation_counts()


@pytest.mark.parametrize(
    "value, ok",
    [("3", True), ("-2", True), ("3.0", True), ("3.5", False), ("0.1", False), ("three", False)],
)
def test_int_column_accepts_only_whole_numbers(value, ok):
    contract = parse_contract({"columns": {"parcels": {"type": "int"}}})
    result = validate_frame(pd.DataFrame({"parcels": [value]}), contract)
    assert (len(result.accepted) == 1) is ok
    if not ok:
        assert result.quarantined.loc[0, "_violations"] == "parcels.type_int"


def test_float_column_still_accepts_decimals():
    contract = parse_contract({"columns": {"w": {"type": "float"}}})
    assert len(validate_frame(pd.DataFrame({"w": ["3.5"]}), contract).accepted) == 1
