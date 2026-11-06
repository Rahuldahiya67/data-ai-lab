import numpy as np
import pandas as pd
import pytest

from eta_model_drift_monitor.generator import (
    MIX_SHIFT_TRUCK_SHARE,
    SLOWDOWN_FACTOR,
    GeneratorConfig,
    Scenario,
    generate,
    true_hours,
    write_dataset,
)
from eta_model_drift_monitor.schema import validate


@pytest.fixture(scope="module")
def data():
    return generate(GeneratorConfig(shipments_per_week=800, seed=7))


def test_generation_is_deterministic():
    cfg = GeneratorConfig(shipments_per_week=50, seed=3)
    a, ta = generate(cfg)
    b, tb = generate(cfg)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(ta, tb)


def test_every_week_is_present_with_unique_ids(data):
    shipments, _ = data
    assert sorted(shipments["week"].unique()) == list(range(1, 25))
    assert shipments["shipment_id"].is_unique
    assert (shipments.groupby("week").size() == 800).all()


def test_ground_truth_lists_the_planted_scenarios(data):
    _, truth = data
    assert list(truth["kind"]) == ["mix_shift", "unit_bug", "slowdown"]
    assert (truth["start_week"] > 8).all()


def test_reference_weeks_are_almost_all_valid(data):
    shipments, _ = data
    ref = validate(shipments[shipments["week"] <= 8])
    assert ref.reject_rate < 0.01


def test_mix_shift_raises_the_truck_share(data):
    shipments, _ = data
    share = shipments.groupby("week")["vehicle_type"].apply(lambda s: (s == "truck").mean())
    assert share.loc[1:8].mean() == pytest.approx(0.20, abs=0.03)
    assert share.loc[12:14].mean() == pytest.approx(MIX_SHIFT_TRUCK_SHARE, abs=0.03)


def test_unit_bug_inflates_east_distances_only_in_its_window(data):
    shipments, _ = data
    huge = shipments["distance_km"] > 5000
    assert not huge[shipments["week"].between(1, 16)].any()
    bug = shipments["week"].between(17, 18)
    assert huge[bug].sum() > 20
    assert set(shipments.loc[huge, "region"]) == {"east"}


def test_slowdown_lengthens_north_deliveries_only(data):
    shipments, _ = data
    clean = shipments.dropna()
    expected = true_hours(
        clean["distance_km"], clean["stops"], clean["dispatch_hour"], clean["region"], clean["vehicle_type"]
    )
    ratio = clean["actual_hours"] / expected
    in_window = clean["week"].between(20, 24)
    north = clean["region"] == "north"
    assert ratio[in_window & north].median() == pytest.approx(SLOWDOWN_FACTOR, rel=0.05)
    assert ratio[in_window & ~north].median() == pytest.approx(1.0, rel=0.05)
    assert ratio[~in_window].median() == pytest.approx(1.0, rel=0.05)


def test_true_hours_is_vectorised_and_positive():
    hours = true_hours(np.array([10.0, 500.0]), np.array([2, 3]), np.array([9, 14]), ["north", "south"], ["bike", "truck"])
    assert hours.shape == (2,)
    assert (hours > 0).all()
    # Peak dispatch (9) adds delay compared with the same trip at 14:00.
    peak, off_peak = true_hours([10.0, 10.0], [2, 2], [9, 14], ["north", "north"], ["bike", "bike"])
    assert peak > off_peak


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reference_weeks": 0},
        {"shipments_per_week": 0},
        {"missing_rate": 1.0},
        {"scenarios": (Scenario("mix_shift", 3, 5),)},  # inside the reference period
        {"scenarios": (Scenario("mix_shift", 20, 30),)},  # past the last week
        {"scenarios": (Scenario("flood", 10, 11),)},
    ],
)
def test_bad_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        GeneratorConfig(**kwargs)


def test_write_dataset_round_trips(tmp_path):
    cfg = GeneratorConfig(shipments_per_week=20, monitoring_weeks=0, scenarios=())
    ship_path, truth_path = write_dataset(tmp_path, cfg)
    back = pd.read_csv(ship_path)
    assert len(back) == 20 * 8
    assert pd.read_csv(truth_path).empty


def test_unit_bug_shows_up_as_a_reject_rate_spike(data):
    shipments, _ = data
    rates = {w: validate(g).reject_rate for w, g in shipments.groupby("week")}
    quiet = max(rate for w, rate in rates.items() if not 17 <= w <= 18)
    assert min(rates[17], rates[18]) > 3 * quiet
