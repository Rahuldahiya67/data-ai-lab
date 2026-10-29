from datetime import date, datetime

import pytest

from delivery_exception_triage_agent.models import ScanEvent, Shipment

AS_OF = datetime(2026, 10, 28, 10, 0)


@pytest.fixture
def as_of():
    return AS_OF


@pytest.fixture
def in_transit_shipment():
    return Shipment(
        shipment_id="SHP-00001",
        carrier="BlueLine",
        origin_hub="Delhi",
        destination_city="Jaipur",
        promised_date=date(2026, 10, 26),
        status="in_transit",
        declared_value_inr=2400.0,
        address_valid=True,
        scans=(
            ScanEvent(datetime(2026, 10, 24, 9, 0), "Delhi", "PICKED_UP"),
            ScanEvent(datetime(2026, 10, 27, 20, 0), "Gurugram", "HUB_IN"),
        ),
    )
