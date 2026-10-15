"""Synthetic carrier shipment feed with defects injected at known rates.

Each generated row carries a hidden ``_defect`` label (empty for clean rows) so
tests can compare what the checker flags with what was actually broken. The
label column is dropped when the feed is written to CSV for normal use.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

CARRIERS = ["BLUEDART", "DELHIVERY", "GATI", "SAFEXPRESS", "VRL"]
STATUSES = ["PICKED_UP", "IN_TRANSIT", "OUT_FOR_DELIVERY", "DELIVERED", "RTO"]
CITIES = [
    "Delhi", "Mumbai", "Bengaluru", "Chennai", "Kolkata",
    "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
]

DEFECT_TYPES = [
    "missing_tracking_id",
    "bad_tracking_format",
    "unknown_status",
    "negative_weight",
    "delivered_before_pickup",
    "duplicate_tracking_id",
    "missing_delivered_at",
]


@dataclass
class FeedSpec:
    """Settings for one synthetic feed."""

    rows: int = 1000
    defect_rate: float = 0.08
    seed: int = 42
    feed_date: datetime = field(default_factory=lambda: datetime(2026, 10, 14))


def _clean_row(rng: random.Random, idx: int, feed_date: datetime) -> dict:
    carrier = rng.choice(CARRIERS)
    origin, destination = rng.sample(CITIES, 2)
    picked_up = feed_date - timedelta(hours=rng.randint(6, 120))
    status = rng.choices(STATUSES, weights=[10, 35, 15, 35, 5])[0]
    delivered = None
    if status == "DELIVERED":
        delivered = picked_up + timedelta(hours=rng.randint(4, 96))
    return {
        "tracking_id": f"{carrier[:3]}{idx:08d}",
        "carrier": carrier,
        "origin_city": origin,
        "destination_city": destination,
        "status": status,
        "weight_kg": round(rng.uniform(0.2, 950.0), 2),
        "freight_charge_inr": round(rng.uniform(80, 42000), 2),
        "picked_up_at": picked_up.strftime("%Y-%m-%d %H:%M:%S"),
        "delivered_at": delivered.strftime("%Y-%m-%d %H:%M:%S") if delivered else "",
        "_defect": "",
    }


def _inject(rng: random.Random, row: dict, defect: str, previous_ids: list[str]) -> dict:
    row = dict(row)
    if defect == "missing_tracking_id":
        row["tracking_id"] = ""
    elif defect == "bad_tracking_format":
        row["tracking_id"] = row["tracking_id"].lower().replace("0", "-", 1)
    elif defect == "unknown_status":
        row["status"] = rng.choice(["DLVRD", "In Transit", "LOST?", "HOLD"])
    elif defect == "negative_weight":
        row["weight_kg"] = -abs(row["weight_kg"])
    elif defect == "delivered_before_pickup":
        picked = datetime.strptime(row["picked_up_at"], "%Y-%m-%d %H:%M:%S")
        row["status"] = "DELIVERED"
        row["delivered_at"] = (picked - timedelta(hours=rng.randint(1, 48))).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    elif defect == "duplicate_tracking_id":
        row["tracking_id"] = rng.choice(previous_ids)
    elif defect == "missing_delivered_at":
        row["status"] = "DELIVERED"
        row["delivered_at"] = ""
    row["_defect"] = defect
    return row


def generate_feed(spec: FeedSpec | None = None) -> pd.DataFrame:
    """Return a synthetic feed as a DataFrame, including the ``_defect`` label."""
    spec = spec or FeedSpec()
    if spec.rows < 1:
        raise ValueError("rows must be at least 1")
    if not 0.0 <= spec.defect_rate <= 1.0:
        raise ValueError("defect_rate must be between 0 and 1")

    rng = random.Random(spec.seed)
    rows: list[dict] = []
    clean_ids: list[str] = []
    for idx in range(spec.rows):
        row = _clean_row(rng, idx, spec.feed_date)
        if rng.random() < spec.defect_rate:
            choices = DEFECT_TYPES if clean_ids else [d for d in DEFECT_TYPES if d != "duplicate_tracking_id"]
            row = _inject(rng, row, rng.choice(choices), clean_ids)
        else:
            clean_ids.append(row["tracking_id"])
        rows.append(row)
    return pd.DataFrame(rows)


def write_feed(path: str, spec: FeedSpec | None = None, keep_labels: bool = False) -> pd.DataFrame:
    """Generate a feed and write it to ``path`` as CSV. Returns the full frame."""
    df = generate_feed(spec)
    out = df if keep_labels else df.drop(columns=["_defect"])
    out.to_csv(path, index=False)
    return df
