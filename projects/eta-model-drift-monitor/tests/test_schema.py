import numpy as np
import pandas as pd
import pytest

from eta_model_drift_monitor.schema import SchemaError, validate


def _row(**overrides):
    row = {
        "shipment_id": "S1",
        "week": 1,
        "distance_km": 120.0,
        "stops": 3,
        "dispatch_hour": 9,
        "region": "north",
        "vehicle_type": "van",
        "actual_hours": 6.5,
    }
    row.update(overrides)
    return row


def test_clean_row_is_valid():
    result = validate(pd.DataFrame([_row()]))
    assert len(result.valid) == 1
    assert result.rejected.empty
    assert result.reject_rate == 0.0


def test_missing_column_is_a_schema_error():
    df = pd.DataFrame([_row()]).drop(columns=["region"])
    with pytest.raises(SchemaError, match="region"):
        validate(df)


def test_label_column_is_optional():
    df = pd.DataFrame([_row()]).drop(columns=["actual_hours"])
    result = validate(df)
    assert len(result.valid) == 1
    assert result.valid["actual_hours"].isna().all()


def test_empty_label_is_allowed():
    result = validate(pd.DataFrame([_row(actual_hours=np.nan)]))
    assert len(result.valid) == 1


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"distance_km": 250_000.0}, "distance_km=250000 outside"),
        ({"distance_km": np.nan}, "distance_km is missing"),
        ({"distance_km": "far"}, "distance_km is missing"),
        ({"stops": 0}, "stops=0 outside"),
        ({"stops": 2.5}, "stops is not a whole number"),
        ({"dispatch_hour": 24}, "dispatch_hour=24 outside"),
        ({"dispatch_hour": 9.5}, "dispatch_hour is not a whole hour"),
        ({"region": None}, "region is missing"),
        ({"vehicle_type": "  "}, "vehicle_type is missing"),
        ({"actual_hours": -1.0}, "actual_hours=-1 outside"),
    ],
)
def test_bad_rows_are_rejected_with_a_reason(overrides, reason):
    result = validate(pd.DataFrame([_row(), _row(shipment_id="S2", **overrides)]))
    assert list(result.valid["shipment_id"]) == ["S1"]
    assert len(result.rejected) == 1
    assert result.rejected.loc[0, "reason"].startswith(reason)
    assert result.reject_rate == 0.5


def test_duplicate_shipment_keeps_first():
    result = validate(pd.DataFrame([_row(), _row(distance_km=999.0)]))
    assert len(result.valid) == 1
    assert result.valid.loc[0, "distance_km"] == 120.0
    assert result.rejected.loc[0, "reason"] == "duplicate shipment_id"


def test_categories_are_normalised_and_unknown_ones_kept():
    result = validate(pd.DataFrame([_row(region=" North "), _row(shipment_id="S2", region="islands")]))
    assert list(result.valid["region"]) == ["north", "islands"]


def test_validate_does_not_modify_input():
    df = pd.DataFrame([_row(region=" North ")])
    validate(df)
    assert df.loc[0, "region"] == " North "
