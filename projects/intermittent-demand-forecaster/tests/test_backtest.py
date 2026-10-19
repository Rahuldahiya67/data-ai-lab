import numpy as np
import pytest

from intermittent_demand_forecaster.backtest import (
    backtest,
    rolling_forecasts,
    score,
    select_best,
    summarise,
)
from intermittent_demand_forecaster.methods import METHODS


def test_rolling_forecasts_do_not_see_the_future():
    y = np.array([1, 2, 3, 4, 100.0])
    f = rolling_forecasts(y, "naive", holdout=2)
    # forecast for t=3 sees up to 3, forecast for t=4 sees up to 4; never 100
    assert list(f) == [3, 4]


def test_rolling_forecasts_validates_inputs():
    y = np.arange(5.0)
    with pytest.raises(ValueError):
        rolling_forecasts(y, "naive", holdout=5)
    with pytest.raises(KeyError):
        rolling_forecasts(y, "prophet", holdout=2)


def test_score_by_hand():
    train = np.array([0, 2, 0, 2.0])  # naive errors: 2, 2, 2 -> MAE 2, RMSE 2, mean 1
    actual = np.array([2, 0.0])
    forecast = np.array([1, 1.0])  # errors -1, +1
    s = score(actual, forecast, train)
    assert s["rmse"] == pytest.approx(1.0)
    assert s["mase"] == pytest.approx(0.5)
    assert s["rmsse"] == pytest.approx(0.5)
    assert s["bias"] == pytest.approx(0.0)


def test_score_flat_training_gives_nan():
    s = score(np.array([1.0]), np.array([1.0]), np.array([3, 3, 3.0]))
    assert np.isnan(s["mase"]) and np.isnan(s["rmsse"])


def test_backtest_shape(demand):
    results = backtest(demand, holdout=20)
    assert len(results) == demand["sku"].nunique() * len(METHODS)
    assert set(results.columns) == {"sku", "demand_class", "method", "rmse", "mase", "rmsse", "bias"}


def test_select_best_one_row_per_sku_with_min_metric(demand):
    results = backtest(demand, holdout=20)
    best = select_best(results)
    assert len(best) == demand["sku"].nunique()
    for _, row in best.iterrows():
        sku_rows = results[results["sku"] == row["sku"]]
        assert row["rmsse"] == pytest.approx(sku_rows["rmsse"].min())


def test_intermittent_methods_beat_naive_on_sparse_skus():
    from intermittent_demand_forecaster.generate import generate_demand

    df = generate_demand(n_skus=40, n_weeks=156, seed=11)
    results = backtest(df, holdout=52)
    table = summarise(results)
    for cls in ("intermittent", "lumpy"):
        assert table.loc[cls, "sba"] < table.loc[cls, "naive"]
        assert table.loc[cls, "tsb"] < table.loc[cls, "naive"]


def _two_sku_frame(dead_train: list[int]) -> "pd.DataFrame":
    import pandas as pd

    weeks = pd.date_range("2024-01-01", periods=30, freq="W-MON")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "sku": ["A"] * 30 + ["B"] * 30,
            "week": list(weeks) * 2,
            "demand": dead_train + list(rng.integers(0, 5, 30)),
        }
    )


@pytest.mark.parametrize(
    "a_demand",
    [
        [0] * 25 + [1, 0, 2, 0, 0],  # no demand until the holdout (#8)
        [3] * 25 + [1, 0, 2, 0, 0],  # constant training demand, naive scale 0
    ],
)
def test_select_best_falls_back_when_no_method_can_be_scored(a_demand):
    from intermittent_demand_forecaster.backtest import FALLBACK_METHOD

    results = backtest(_two_sku_frame(a_demand), holdout=5)
    best = select_best(results).set_index("sku")
    assert list(best.index) == ["A", "B"]
    assert best.loc["A", "method"] == FALLBACK_METHOD
    assert best.loc["A", "selected_by"] == "fallback"
    assert best.loc["B", "selected_by"] == "metric"


def test_backtest_skips_short_skus_with_warning(demand):
    import pandas as pd

    short = pd.DataFrame(
        {
            "sku": ["NEW-1"] * 3,
            "week": pd.date_range("2025-06-02", periods=3, freq="W-MON"),
            "demand": [0, 1, 0],
        }
    )
    df = pd.concat([demand.drop(columns="pattern"), short], ignore_index=True)
    with pytest.warns(UserWarning, match="NEW-1"):
        results = backtest(df, holdout=20)
    assert "NEW-1" not in set(results["sku"])
    assert results["sku"].nunique() == demand["sku"].nunique()
