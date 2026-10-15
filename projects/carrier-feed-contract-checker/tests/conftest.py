from pathlib import Path

import pandas as pd
import pytest

from carrier_feed_contract_checker.contract import load_contract

CONTRACT_PATH = Path(__file__).resolve().parents[1] / "contracts" / "carrier_shipment_status.yaml"


@pytest.fixture
def contract():
    return load_contract(CONTRACT_PATH)


@pytest.fixture
def clean_row():
    return {
        "tracking_id": "BLU00000001",
        "carrier": "BLUEDART",
        "origin_city": "Delhi",
        "destination_city": "Mumbai",
        "status": "DELIVERED",
        "weight_kg": "12.5",
        "freight_charge_inr": "850.00",
        "picked_up_at": "2026-10-12 08:00:00",
        "delivered_at": "2026-10-13 17:30:00",
    }


@pytest.fixture
def make_frame(clean_row):
    """Build a frame from overrides applied to copies of a clean row."""

    def _make(*overrides: dict) -> pd.DataFrame:
        rows = []
        for i, o in enumerate(overrides):
            row = dict(clean_row, tracking_id=f"BLU{i:08d}")
            row.update(o)
            rows.append(row)
        return pd.DataFrame(rows)

    return _make
