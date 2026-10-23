import numpy as np
import pytest

from intermittent_demand_forecaster.classify import classify_series
from intermittent_demand_forecaster.generate import PATTERNS, generate_demand


def test_shape_and_columns(demand):
    assert list(demand.columns) == ["sku", "pattern", "week", "demand"]
    assert len(demand) == 12 * 80
    assert demand["sku"].nunique() == 12


def test_demand_is_non_negative_int(demand):
    assert demand["demand"].dtype.kind == "i"
    assert (demand["demand"] >= 0).all()


def test_weeks_are_consecutive_mondays(demand):
    weeks = demand.loc[demand["sku"] == "PART-0001", "week"]
    assert (weeks.dt.dayofweek == 0).all()
    assert (weeks.diff().dropna().dt.days == 7).all()


def test_same_seed_same_data():
    a = generate_demand(n_skus=4, n_weeks=20, seed=1)
    b = generate_demand(n_skus=4, n_weeks=20, seed=1)
    c = generate_demand(n_skus=4, n_weeks=20, seed=2)
    assert a.equals(b)
    assert not a.equals(c)


def test_patterns_spread_evenly():
    df = generate_demand(n_skus=8, n_weeks=10)
    counts = df.drop_duplicates("sku")["pattern"].value_counts()
    assert set(counts.index) == set(PATTERNS)
    assert (counts == 2).all()


def test_generated_patterns_match_classifier_on_long_history():
    # With a long history the classifier should recover the generating pattern.
    df = generate_demand(n_skus=40, n_weeks=520, seed=3)
    agreement = []
    for _, group in df.groupby("sku"):
        stats = classify_series(group["demand"].to_numpy())
        agreement.append(stats.demand_class == group["pattern"].iloc[0])
    assert np.mean(agreement) >= 0.95


@pytest.mark.parametrize("kwargs", [{"n_skus": 0}, {"n_weeks": 0}])
def test_rejects_empty_sizes(kwargs):
    with pytest.raises(ValueError):
        generate_demand(**kwargs)
