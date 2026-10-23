import pandas as pd
import pytest

from intermittent_demand_forecaster.backtest import backtest
from intermittent_demand_forecaster.classify import classify_skus
from intermittent_demand_forecaster.generate import generate_demand
from intermittent_demand_forecaster.prepare import prepare_demand


def test_order_lines_without_zero_weeks_classify_like_full_series():
    df = generate_demand(n_skus=8, n_weeks=104, seed=1)
    full = classify_skus(df)
    # Like an order-line extract: zero weeks are absent. Keep every SKU's
    # first week (zero or not) so the series start at the same place.
    first = df.groupby("sku")["week"].transform("min")
    orders_only = df[(df["demand"] > 0) | (df["week"] == first)]
    sparse = classify_skus(orders_only)
    pd.testing.assert_frame_equal(full, sparse)
    assert set(full["demand_class"]) == {"smooth", "erratic", "intermittent", "lumpy"}


def test_order_lines_without_zero_weeks_backtest_like_full_series():
    df = generate_demand(n_skus=4, n_weeks=60, seed=3)
    first = df.groupby("sku")["week"].transform("min")
    orders_only = df[(df["demand"] > 0) | (df["week"] == first)]
    pd.testing.assert_frame_equal(backtest(df, holdout=10), backtest(orders_only, holdout=10))


def test_duplicate_rows_are_summed_and_dates_moved_to_monday():
    raw = pd.DataFrame(
        {
            "sku": ["A", "A", "A"],
            "week": ["2024-01-03", "2024-01-05", "2024-01-15"],  # Wed, Fri, Mon
            "demand": [2, 3, 1],
        }
    )
    out = prepare_demand(raw)
    assert list(out["week"]) == list(pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"]))
    assert list(out["demand"]) == [5.0, 0.0, 1.0]


def test_series_run_to_the_last_week_in_the_data():
    raw = pd.DataFrame(
        {
            "sku": ["OLD", "NEW"],
            "week": pd.to_datetime(["2024-01-01", "2024-01-22"]),
            "demand": [4, 1],
        }
    )
    out = prepare_demand(raw).groupby("sku")["demand"].apply(list)
    assert out["OLD"] == [4.0, 0.0, 0.0, 0.0]  # trailing zeros kept
    assert out["NEW"] == [1.0]  # starts at its own first week


@pytest.mark.parametrize(
    "raw, message",
    [
        (pd.DataFrame({"sku": ["A"], "demand": [1]}), "missing column"),
        (pd.DataFrame({"sku": ["A"], "week": ["2024-01-01"], "demand": [None]}), "missing values"),
        (pd.DataFrame({"sku": ["A"], "week": ["2024-01-01"], "demand": [-2]}), "non-negative"),
    ],
)
def test_bad_input_is_rejected(raw, message):
    with pytest.raises(ValueError, match=message):
        prepare_demand(raw)
