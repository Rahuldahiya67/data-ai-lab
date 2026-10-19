import pandas as pd
import pytest

from intermittent_demand_forecaster.backtest import backtest, select_best
from intermittent_demand_forecaster.reorder import REPORT_COLUMNS, reorder_points


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
