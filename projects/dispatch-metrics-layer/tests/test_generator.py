from datetime import date

import pytest

from dispatch_metrics_layer.generator import GeneratorConfig, generate, transit_days, write_csvs


def test_same_seed_gives_identical_files(tmp_path):
    a = write_csvs(generate(GeneratorConfig(n_shipments=300, seed=3)), tmp_path / "a")
    b = write_csvs(generate(GeneratorConfig(n_shipments=300, seed=3)), tmp_path / "b")
    for name in a:
        assert a[name].read_bytes() == b[name].read_bytes()


def test_rows_are_internally_consistent():
    cfg = GeneratorConfig(n_shipments=2000, seed=5)
    rows = generate(cfg)["shipments"]
    statuses = {r["status"] for r in rows}
    assert statuses == {"delivered", "in_transit", "cancelled"}
    last_day = date.fromordinal(cfg.start.toordinal() + 7 * cfg.weeks - 1)
    for r in rows:
        dispatch = date.fromisoformat(r["dispatch_date"])
        assert cfg.start <= dispatch <= last_day
        assert date.fromisoformat(r["promised_date"]) > dispatch
        if r["status"] == "delivered":
            assert date.fromisoformat(r["delivered_date"]) >= dispatch
            assert 0 < r["units_delivered"] <= r["units_ordered"]
        else:
            assert r["delivered_date"] == ""
            assert r["units_delivered"] == 0


def test_shipment_ids_are_unique():
    rows = generate(GeneratorConfig(n_shipments=1000))["shipments"]
    assert len({r["shipment_id"] for r in rows}) == 1000


@pytest.mark.parametrize("km, days", [(1, 1), (450, 1), (451, 2), (1415, 4)])
def test_transit_days(km, days):
    assert transit_days(km) == days


@pytest.mark.parametrize(
    "kwargs", [{"n_shipments": 0}, {"weeks": 0}, {"cancel_rate": 1.0}, {"cancel_rate": -0.1}]
)
def test_config_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        GeneratorConfig(**kwargs)
