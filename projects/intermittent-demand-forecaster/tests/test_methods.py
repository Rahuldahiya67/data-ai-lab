import numpy as np
import pytest

from intermittent_demand_forecaster.methods import (
    METHODS,
    croston,
    moving_average,
    naive,
    sba,
    ses,
    tsb,
)


@pytest.mark.parametrize("name", list(METHODS))
def test_empty_and_all_zero_history_forecast_zero(name):
    assert METHODS[name]([]) == 0.0
    assert METHODS[name]([0, 0, 0, 0]) == 0.0


@pytest.mark.parametrize("name", list(METHODS))
def test_constant_demand_forecasts_the_constant(name):
    y = [5.0] * 30
    expected = 5.0 * (1 - 0.1 / 2) if name == "sba" else 5.0
    assert METHODS[name](y) == pytest.approx(expected)


def test_naive_and_moving_average():
    y = [1, 2, 3, 4, 5, 6]
    assert naive(y) == 6
    assert moving_average(y, window=3) == 5
    assert moving_average([2, 4], window=4) == 3


def test_ses_by_hand():
    # level: 10 -> 10 + 0.5 * (20 - 10) = 15 -> 15 + 0.5 * (0 - 15) = 7.5
    assert ses([10, 20, 0], alpha=0.5) == pytest.approx(7.5)


def test_croston_by_hand():
    # demands at t=1 (size 4) and t=4 (size 8), alpha=0.5
    # init: z=4, p=2 ; at t=4: q=3, z=6, p=2.5 -> 6 / 2.5 = 2.4
    y = [0, 4, 0, 0, 8, 0]
    assert croston(y, alpha=0.5) == pytest.approx(2.4)
    assert sba(y, alpha=0.5) == pytest.approx(2.4 * 0.75)


def test_croston_ignores_trailing_zeros_but_tsb_decays():
    base = [0, 5, 0, 5, 0, 5]
    stale = base + [0] * 20
    assert croston(stale) == pytest.approx(croston(base))
    assert tsb(stale) < tsb(base) / 2


def test_tsb_by_hand():
    # y = [3, 0], alpha=beta=0.5. init d=0.5, z=3
    # t0: d=0.75, z=3 ; t1: d=0.375 -> 0.375 * 3 = 1.125
    assert tsb([3, 0], alpha=0.5, beta=0.5) == pytest.approx(1.125)


def test_sba_below_croston_on_intermittent_data():
    rng = np.random.default_rng(0)
    y = np.where(rng.random(100) < 0.3, 5, 0)
    assert 0 < sba(y) < croston(y)


@pytest.mark.parametrize("alpha", [0, -0.1, 1.5])
def test_rejects_bad_alpha(alpha):
    with pytest.raises(ValueError):
        ses([1, 2], alpha=alpha)
    with pytest.raises(ValueError):
        croston([1, 2], alpha=alpha)
