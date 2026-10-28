"""Point-in-time queries over the SCD Type 2 rate dimension."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import duckdb


@dataclass(frozen=True)
class RateVersion:
    lane_id: str
    carrier: str
    rate_per_kg: float
    min_charge: float
    currency: str
    valid_from: datetime
    valid_to: datetime
    source_lsn: int

    def charge(self, weight_kg: float) -> float:
        """Freight charge for a shipment: rate times weight, at least the minimum charge."""
        if weight_kg <= 0:
            raise ValueError(f"weight_kg must be positive, got {weight_kg}")
        return round(max(weight_kg * self.rate_per_kg, self.min_charge), 2)


_COLUMNS = "lane_id, carrier, rate_per_kg, min_charge, currency, valid_from, valid_to, source_lsn"


def rate_at(con: duckdb.DuckDBPyConnection, lane_id: str, at: datetime) -> RateVersion | None:
    """Return the version of ``lane_id`` that was valid at ``at``, or None.

    None means the lane had no contracted rate at that moment: it did not exist
    yet, or it had been withdrawn.
    """
    row = con.execute(
        f"SELECT {_COLUMNS} FROM dim_rate_card "
        "WHERE lane_id = ? AND valid_from <= ? AND ? < valid_to",
        [lane_id, at, at],
    ).fetchone()
    return RateVersion(*row) if row else None


def lane_history(con: duckdb.DuckDBPyConnection, lane_id: str) -> list[RateVersion]:
    """All versions of a lane, oldest first."""
    rows = con.execute(
        f"SELECT {_COLUMNS} FROM dim_rate_card WHERE lane_id = ? ORDER BY valid_from, source_lsn",
        [lane_id],
    ).fetchall()
    return [RateVersion(*row) for row in rows]


def current_rates(con: duckdb.DuckDBPyConnection) -> list[RateVersion]:
    """The open version of every live lane."""
    rows = con.execute(
        f"SELECT {_COLUMNS} FROM dim_rate_card WHERE is_current ORDER BY lane_id"
    ).fetchall()
    return [RateVersion(*row) for row in rows]
