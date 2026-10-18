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
    assert s["mase"] == pytest.approx(0.5)
    assert s["rmsse"] == pytest.approx(0.5)
    assert s["bias"] == pytest.approx(0.0)


def test_score_flat_training_gives_nan():
    s = score(np.array([1.0]), np.array([1.0]), np.array([3, 3, 3.0]))
    assert np.isnan(s["mase"]) and np.isnan(s["rmsse"])


def test_backtest_shape(demand):
    results = backtest(demand, holdout=20)
    assert len(results) == demand["sku"].nunique() * len(METHODS)
    assert set(results.columns) == {"sku", "demand_class", "method", "mase", "rmsse", "bias"}


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
