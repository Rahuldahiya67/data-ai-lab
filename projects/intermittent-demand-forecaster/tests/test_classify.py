import math

import pandas as pd
import pytest

from intermittent_demand_forecaster.classify import classify_series, classify_skus


def test_smooth():
    s = classify_series([10, 11, 9, 10, 12, 10])
    assert s.adi == 1.0
    assert s.cv2 < 0.49
    assert s.demand_class == "smooth"


def test_erratic():
    assert classify_series([1, 30, 2, 50, 1, 40]).demand_class == "erratic"


def test_intermittent():
    s = classify_series([0, 0, 5, 0, 0, 5, 0, 6, 0, 0])
    assert s.adi == pytest.approx(10 / 3)
    assert s.demand_class == "intermittent"


def test_lumpy():
    assert classify_series([0, 0, 1, 0, 0, 40, 0, 2, 0, 0]).demand_class == "lumpy"


def test_no_demand():
    s = classify_series([0, 0, 0])
    assert s.demand_class == "no_demand"
    assert math.isinf(s.adi)


def test_single_demand_has_zero_cv2():
    s = classify_series([0, 0, 0, 7])
    assert s.cv2 == 0.0
    assert s.demand_class == "intermittent"


def test_rejects_negative_demand():
    with pytest.raises(ValueError):
        classify_series([1, -1, 2])


def test_classify_skus_sorts_by_week():
    df = pd.DataFrame(
        {
            "sku": ["A"] * 4,
            "week": pd.to_datetime(["2024-01-22", "2024-01-01", "2024-01-15", "2024-01-08"]),
            "demand": [0, 3, 0, 3],
        }
    )
    out = classify_skus(df)
    assert out.loc[0, "adi"] == 2.0
    assert out.loc[0, "demand_class"] == "intermittent"
