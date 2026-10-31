from pathlib import Path

import pandas as pd
import pytest

from dispatch_metrics_layer.engine import MetricsEngine
from dispatch_metrics_layer.generator import GeneratorConfig, generate, write_csvs
from dispatch_metrics_layer.spec import load_spec

PROJECT = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT / "specs" / "dispatch.yaml"


@pytest.fixture(scope="session")
def spec():
    return load_spec(SPEC_PATH)


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    write_csvs(generate(GeneratorConfig(n_shipments=1500, seed=11)), out)
    return out


@pytest.fixture(scope="session")
def engine(spec, data_dir):
    return MetricsEngine.from_csv_dir(spec, data_dir)


@pytest.fixture(scope="session")
def frame(data_dir):
    """Shipments joined to carriers and lanes in pandas: the independent reference."""
    ships = pd.read_csv(
        data_dir / "shipments.csv",
        parse_dates=["dispatch_date", "promised_date", "delivered_date"],
    )
    carriers = pd.read_csv(data_dir / "carriers.csv")
    lanes = pd.read_csv(data_dir / "lanes.csv").rename(columns={"distance_km": "lane_km"})
    return ships.merge(carriers, on="carrier_id", how="left").merge(lanes, on="lane_id", how="left")
