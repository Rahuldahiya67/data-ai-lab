import numpy as np
import pandas as pd
import pytest

from eta_model_drift_monitor.drift import (
    UNSEEN,
    feature_drift,
    numeric_edges,
    prediction_drift,
    psi_categorical,
    psi_numeric,
    status_for,
)
from eta_model_drift_monitor.generator import GeneratorConfig, generate
from eta_model_drift_monitor.schema import validate


@pytest.fixture(scope="module")
def weeks():
    shipments, _ = generate(GeneratorConfig(shipments_per_week=800, seed=21))
    valid = validate(shipments).valid
    return {w: g for w, g in valid.groupby("week")}, valid[valid["week"] <= 8]


def test_identical_distributions_have_zero_psi():
    s = pd.Series(np.arange(1000, dtype=float))
    assert psi_numeric(s, s) == pytest.approx(0.0, abs=1e-12)
    c = pd.Series(["a", "b", "b", "c"] * 50)
    assert psi_categorical(c, c) == pytest.approx(0.0, abs=1e-12)


def test_psi_matches_a_hand_calculation():
    ref = pd.Series(["a"] * 50 + ["b"] * 50)
    cur = pd.Series(["a"] * 80 + ["b"] * 20)
    expected = (0.8 - 0.5) * np.log(0.8 / 0.5) + (0.2 - 0.5) * np.log(0.2 / 0.5)
    assert psi_categorical(ref, cur) == pytest.approx(expected)


def test_psi_grows_with_the_size_of_the_shift():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 5000))
    small = psi_numeric(ref, pd.Series(rng.normal(0.2, 1, 5000)))
    large = psi_numeric(ref, pd.Series(rng.normal(1.0, 1, 5000)))
    assert 0 < small < large
    assert status_for(large) == "alert"


def test_values_beyond_the_reference_range_land_in_the_outer_bins():
    ref = pd.Series(np.linspace(0, 100, 1000))
    cur = pd.Series(np.full(100, 1e6))
    assert psi_numeric(ref, cur) > 1.0


def test_unseen_categories_get_their_own_bin():
    ref = pd.Series(["van"] * 90 + ["bike"] * 10)
    cur = pd.Series(["van"] * 90 + ["drone"] * 10)
    assert psi_categorical(ref, cur) > 0.25
    assert UNSEEN not in ref.values


def test_discrete_feature_edges_are_deduplicated():
    edges = numeric_edges(pd.Series([1, 1, 1, 1, 2, 2, 3] * 20), bins=10)
    assert list(edges) == sorted(set(edges))


def test_empty_inputs_raise_clear_errors():
    with pytest.raises(ValueError, match="reference"):
        psi_numeric(pd.Series([], dtype=float), pd.Series([1.0]))
    with pytest.raises(ValueError, match="current"):
        psi_numeric(pd.Series([1.0, 2.0]), pd.Series([np.nan]))
    with pytest.raises(ValueError, match="bins"):
        numeric_edges(pd.Series([1.0, 2.0]), bins=1)
    with pytest.raises(ValueError, match="current"):
        psi_categorical(pd.Series(["a"]), pd.Series([None]))


@pytest.mark.parametrize("psi, status", [(0.0, "ok"), (0.099, "ok"), (0.10, "warn"), (0.249, "warn"), (0.25, "alert")])
def test_status_thresholds(psi, status):
    assert status_for(psi) == status


def test_quiet_week_is_stable(weeks):
    by_week, ref = weeks
    report = {d.feature: d for d in feature_drift(ref, by_week[10])}
    assert set(report) == {"distance_km", "stops", "dispatch_hour", "region", "vehicle_type"}
    assert all(d.status == "ok" for d in report.values()), report


def test_mix_shift_is_caught_on_vehicle_and_distance(weeks):
    by_week, ref = weeks
    report = {d.feature: d for d in feature_drift(ref, by_week[13])}
    assert report["vehicle_type"].status in {"warn", "alert"}
    assert report["distance_km"].status == "alert"
    assert report["region"].status == "ok"


def test_unit_bug_rows_that_pass_validation_are_out_of_range(weeks):
    # Most metre-valued rows are rejected by validation (see test_schema); the
    # short trips that slip through look like trips longer than any seen in
    # the reference period.
    by_week, ref = weeks
    report = {d.feature: d for d in feature_drift(ref, by_week[17])}
    assert report["distance_km"].out_of_range_share > 0
    assert report["region"].status == "ok"
    quiet = {d.feature: d for d in feature_drift(ref, by_week[10])}
    assert quiet["distance_km"].out_of_range_share < report["distance_km"].out_of_range_share


def test_slowdown_is_invisible_in_the_inputs(weeks):
    by_week, ref = weeks
    report = feature_drift(ref, by_week[21])
    assert all(d.status == "ok" for d in report)


def test_prediction_drift_flags_shifted_predictions():
    rng = np.random.default_rng(1)
    ref = rng.lognormal(1.5, 0.5, 4000)
    assert prediction_drift(ref, rng.lognormal(1.5, 0.5, 4000)).status == "ok"
    shifted = prediction_drift(ref, rng.lognormal(2.0, 0.5, 4000))
    assert shifted.status == "alert"
    assert shifted.feature == "predicted_hours"
