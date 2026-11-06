import numpy as np
import pandas as pd
import pytest

from eta_model_drift_monitor.generator import GeneratorConfig, generate
from eta_model_drift_monitor.model import EtaModel
from eta_model_drift_monitor.performance import error_stats, segment_performance
from eta_model_drift_monitor.schema import validate


@pytest.fixture(scope="module")
def scored():
    shipments, _ = generate(GeneratorConfig(shipments_per_week=800, seed=13))
    valid = validate(shipments).valid
    ref = valid[valid["week"] <= 8]
    model = EtaModel.fit(ref)
    valid = valid.assign(predicted_hours=model.predict(valid))
    return {w: g for w, g in valid.groupby("week")}, valid[valid["week"] <= 8]


def test_error_stats_by_hand():
    s = error_stats(np.array([2.0, 4.0, 6.0]), np.array([1.0, 5.0, 6.0]))
    assert s.rows == 3
    assert s.mae == pytest.approx(2 / 3)
    assert s.bias == pytest.approx(0.0)


def test_error_stats_skips_missing_actuals():
    s = error_stats(np.array([2.0, 4.0]), np.array([1.0, np.nan]))
    assert s.rows == 1 and s.mae == 1.0


def test_error_stats_with_no_actuals_is_empty_not_an_error():
    s = error_stats(np.array([2.0]), np.array([np.nan]))
    assert s.rows == 0 and np.isnan(s.mae)


def test_error_stats_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shapes"):
        error_stats(np.array([1.0, 2.0]), np.array([1.0]))


def test_quiet_week_is_ok_everywhere(scored):
    by_week, ref = scored
    report = segment_performance(ref, by_week[10])
    assert report[0].segment == "overall"
    assert [r.status for r in report] == ["ok"] * 6, [(r.segment, r.mae_ratio) for r in report]


def test_slowdown_alerts_on_the_north_region_only(scored):
    by_week, ref = scored
    report = {r.segment: r for r in segment_performance(ref, by_week[21])}
    north = report["region=north"]
    assert north.status == "alert"
    assert north.current.bias < 0  # deliveries take longer than predicted
    for region in ("south", "east", "west", "central"):
        assert report[f"region={region}"].status != "alert"
        assert report[f"region={region}"].mae_ratio < 1.5
    # Diluted overall: clearly weaker than the north signal.
    assert report["overall"].mae_ratio < north.mae_ratio


def test_mix_shift_does_not_hurt_accuracy(scored):
    by_week, ref = scored
    report = {r.segment: r for r in segment_performance(ref, by_week[13], by="vehicle_type")}
    assert report["vehicle_type=truck"].status == "ok"


def test_small_segments_are_marked_insufficient(scored):
    by_week, ref = scored
    cur = by_week[10].copy()
    cur.loc[cur.index[:5], "region"] = "islands"
    report = {r.segment: r for r in segment_performance(ref, cur)}
    assert report["region=islands"].status == "insufficient"
    assert report["region=islands"].reference is None


def test_unlabelled_week_is_insufficient(scored):
    by_week, ref = scored
    cur = by_week[10].assign(actual_hours=np.nan)
    report = segment_performance(ref, cur)
    assert {r.status for r in report} == {"insufficient"}


def test_missing_prediction_column_is_reported(scored):
    by_week, ref = scored
    with pytest.raises(ValueError, match="current frame is missing columns: predicted_hours"):
        segment_performance(ref, by_week[10].drop(columns=["predicted_hours"]))
