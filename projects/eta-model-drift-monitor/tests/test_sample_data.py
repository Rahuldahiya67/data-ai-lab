from pathlib import Path

import pandas as pd

from eta_model_drift_monitor.generator import generate

DATA = Path(__file__).resolve().parents[1] / "data"


def test_committed_sample_matches_the_default_generator():
    shipments, truth = generate()
    on_disk = pd.read_csv(DATA / "shipments.csv")
    pd.testing.assert_frame_equal(on_disk, shipments, check_dtype=False)
    pd.testing.assert_frame_equal(pd.read_csv(DATA / "scenarios.csv"), truth, check_dtype=False)
