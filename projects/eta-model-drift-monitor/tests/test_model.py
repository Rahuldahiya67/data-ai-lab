import numpy as np
import pandas as pd
import pytest

from eta_model_drift_monitor.generator import GeneratorConfig, generate
from eta_model_drift_monitor.model import EtaModel
from eta_model_drift_monitor.schema import validate


@pytest.fixture(scope="module")
def reference():
    shipments, _ = generate(GeneratorConfig(shipments_per_week=500, monitoring_weeks=0, scenarios=(), seed=5))
    return validate(shipments).valid


@pytest.fixture(scope="module")
def model(reference):
    return EtaModel.fit(reference)


def test_model_beats_a_constant_baseline(model, reference):
    y = reference["actual_hours"].to_numpy()
    mae = np.abs(model.predict(reference) - y).mean()
    baseline = np.abs(y - np.median(y)).mean()
    assert mae < 0.25 * baseline


def test_relative_error_is_small_on_fresh_data(model):
    fresh, _ = generate(GeneratorConfig(shipments_per_week=300, monitoring_weeks=0, scenarios=(), seed=99))
    fresh = validate(fresh).valid
    y = fresh["actual_hours"].to_numpy()
    mape = np.median(np.abs(model.predict(fresh) - y) / y)
    assert mape < 0.12


def test_per_km_coefficients_follow_vehicle_speed(model):
    coef = dict(zip(model.feature_names(), model.coef))
    assert coef["km_per_bike"] > coef["km_per_van"] > coef["km_per_truck"] > 0


def test_fit_ignores_unlabelled_rows(reference):
    unlabelled = reference.head(50).assign(actual_hours=np.nan, distance_km=9999.0)
    with_noise = pd.concat([reference, unlabelled], ignore_index=True)
    a = EtaModel.fit(reference)
    b = EtaModel.fit(with_noise)
    np.testing.assert_allclose(a.coef, b.coef)


def test_fit_needs_enough_labelled_rows(reference):
    with pytest.raises(ValueError, match="at least 20"):
        EtaModel.fit(reference.head(10))
    with pytest.raises(ValueError, match="alpha"):
        EtaModel.fit(reference, alpha=-1)


def test_unseen_categories_still_get_a_distance_based_prediction(model, reference):
    row = reference.head(1).copy()
    near = row.assign(vehicle_type="drone", region="islands", distance_km=10.0)
    far = row.assign(vehicle_type="drone", region="islands", distance_km=400.0)
    p_near, p_far = model.predict(near)[0], model.predict(far)[0]
    assert np.isfinite(p_near) and np.isfinite(p_far)
    assert p_far > p_near + 3


def test_predictions_are_never_negative(model, reference):
    row = reference.head(1).assign(distance_km=0.1, stops=1, vehicle_type="bike")
    assert model.predict(row)[0] >= 0.1


def test_save_and_load_round_trip(model, reference, tmp_path):
    path = tmp_path / "model.json"
    model.save(path)
    loaded = EtaModel.load(path)
    np.testing.assert_allclose(loaded.predict(reference), model.predict(reference), rtol=1e-12)


def test_load_rejects_a_file_with_missing_coefficients(model, tmp_path):
    path = tmp_path / "model.json"
    model.save(path)
    path.write_text(path.read_text().replace('"stops"', '"stopz"'))
    with pytest.raises(ValueError, match="stops"):
        EtaModel.load(path)
