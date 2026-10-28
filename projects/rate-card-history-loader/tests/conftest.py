from datetime import datetime, timedelta

import duckdb
import pytest

from rate_card_history_loader.loader import RateCardLoader

T0 = datetime(2026, 7, 1, 9, 0)


def ev(lsn, op="u", minutes=0, lane="L001-DEL-BOM", rate=10.0, min_charge=300.0, carrier="Gati Express"):
    """Build a raw CDC record the way it arrives on the wire."""
    record = {"lsn": lsn, "op": op, "commit_ts": (T0 + timedelta(minutes=minutes)).isoformat(), "lane_id": lane}
    if op != "d":
        record.update(
            carrier=carrier,
            origin=lane.split("-")[1],
            destination=lane.split("-")[2],
            rate_per_kg=rate,
            min_charge=min_charge,
            currency="INR",
        )
    return record


@pytest.fixture
def loader():
    con = duckdb.connect()
    yield RateCardLoader(con)
    con.close()
