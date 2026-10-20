import numpy as np
import pandas as pd
import pytest

from intermittent_demand_forecaster.backtest import backtest, select_best
from intermittent_demand_forecaster.reorder import (
    REPORT_COLUMNS,
    SERVICE_CHECK_COLUMNS,
    reorder_points,
    service_check,
)


def _frame(demand):
    return pd.DataFrame(
        {
            "sku": "A",
            "week": pd.date_range("2024-01-01", periods=len(demand), freq="W-MON"),
            "demand": demand,
        }
    )


def test_reorder_point_by_hand():
    df = _frame([2, 2, 2, 2])
    best = pd.DataFrame(
        [{"sku": "A", "demand_class": "smooth", "method": "naive", "selected_by": "metric", "rmse": 1.0}]
    )
    out = reorder_points(df, best, lead_time=4, service_level=0.95).iloc[0]
    # forecast 2/week, LTD 8, sigma sqrt(4) * 1 = 2, z(0.95) = 1.6449 -> SS 3.29
    assert out["lead_time_demand"] == 8
    assert out["sigma_lead_time"] == 2
    assert out["safety_stock"] == pytest.approx(3.29, abs=0.01)
    assert out["reorder_point"] == 12  # ceil(11.29)


def test_exact_integer_reorder_point_is_not_rounded_up():
    df = _frame([3, 3, 3])
    best = pd.DataFrame(
        [{"sku": "A", "demand_class": "smooth", "method": "naive", "selected_by": "metric", "rmse": 0.0}]
    )
    assert reorder_points(df, best, lead_time=2, service_level=0.5).iloc[0]["reorder_point"] == 6


def test_higher_service_level_never_lowers_reorder_point(demand):
    best = select_best(backtest(demand, holdout=20))
    low = reorder_points(demand, best, service_level=0.80)
    high = reorder_points(demand, best, service_level=0.98)
    assert list(low.columns) == REPORT_COLUMNS
    assert len(low) == demand["sku"].nunique()
    assert (high["reorder_point"] >= low["reorder_point"]).all()
    assert (high["safety_stock"] >= 0).all()


@pytest.mark.parametrize("kwargs", [{"lead_time": 0}, {"lead_time": 2.5}, {"service_level": 1.0}, {"service_level": 0.3}])
def test_rejects_bad_parameters(demand, kwargs):
    best = select_best(backtest(demand, holdout=20))
    with pytest.raises(ValueError):
        reorder_points(demand, best, **kwargs)


def test_unknown_sku_in_best_is_an_error(demand):
    best = select_best(backtest(demand, holdout=20))
    best.loc[0, "sku"] = "NOT-A-SKU"
    with pytest.raises(ValueError, match="NOT-A-SKU"):
        reorder_points(demand, best)


def test_service_check_by_hand():
    # Demand 2 every week except a spike of 10 in week 6. Naive forecast,
    # rmse 0, lead time 2 -> reorder point is 2 * last week's demand.
    df = _frame([2, 2, 2, 2, 2, 2, 10, 2])
    best = pd.DataFrame(
        [{"sku": "A", "demand_class": "smooth", "method": "naive", "selected_by": "metric", "rmse": 0.0}]
    )
    out = service_check(df, best, lead_time=2, service_level=0.5, holdout=4).iloc[0]
    # origins t = 4, 5, 6 (t = 7 has no full window)
    #   t=4: rop 4, demand 2+2  = 4  covered
    #   t=5: rop 4, demand 2+10 = 12 missed
    #   t=6: rop 4, demand 10+2 = 12 missed
    assert out["windows"] == 3
    assert out["covered"] == 1
    assert out["achieved_service"] == pytest.approx(1 / 3)


def test_service_check_only_uses_history_before_each_origin():
    # A huge last week must not raise the reorder points used for earlier windows.
    base = [1, 1, 1, 1, 1, 1]
    best = pd.DataFrame(
        [{"sku": "A", "demand_class": "smooth", "method": "moving_average", "selected_by": "metric", "rmse": 0.0}]
    )
    plain = service_check(_frame(base + [1]), best, lead_time=1, holdout=3)
    spiked = service_check(_frame(base + [500]), best, lead_time=1, holdout=3)
    assert plain.iloc[0]["covered"] == 3
    assert spiked.iloc[0]["covered"] == 2  # only the spike week itself is missed


def test_service_check_without_full_window_is_nan():
    best = pd.DataFrame(
        [{"sku": "A", "demand_class": "smooth", "method": "naive", "selected_by": "metric", "rmse": 1.0}]
    )
    out = service_check(_frame([1, 2, 3]), best, lead_time=4, holdout=2).iloc[0]
    assert out["windows"] == 0
    assert np.isnan(out["achieved_service"])


def test_service_check_rises_with_target(demand):
    best = select_best(backtest(demand, holdout=20))
    low = service_check(demand, best, service_level=0.6, holdout=20)
    high = service_check(demand, best, service_level=0.98, holdout=20)
    assert list(low.columns) == SERVICE_CHECK_COLUMNS
    assert (high["covered"] >= low["covered"]).all()
