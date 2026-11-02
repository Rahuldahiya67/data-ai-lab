"""Synthetic dispatch data: a shipments fact table plus carrier and lane dimensions.

The data is made up but shaped like a real dispatch extract:

* carriers differ in on-time performance (one is clearly worse),
* some shipments are cancelled and some are still in transit (no delivery date),
* some deliveries are short (units_delivered < units_ordered),
* every shipment carries its planned distance and freight cost.

Everything is driven by a seeded ``random.Random`` so the same seed always
produces byte-identical CSV files.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

CARRIERS = [
    # carrier_id, name, mode, probability a delivery is on time
    ("C01", "Northline Freight", "FTL", 0.93),
    ("C02", "Swift Roadways", "FTL", 0.88),
    ("C03", "Metro Parcel", "PARCEL", 0.95),
    ("C04", "Ganga Logistics", "LTL", 0.78),
    ("C05", "Deccan Carriers", "LTL", 0.90),
    ("C06", "Coastal Express", "PARCEL", 0.84),
]

LANES = [
    # lane_id, origin, destination, region, distance_km
    ("L01", "Delhi", "Jaipur", "North", 281),
    ("L02", "Delhi", "Lucknow", "North", 555),
    ("L03", "Delhi", "Chandigarh", "North", 244),
    ("L04", "Mumbai", "Pune", "West", 149),
    ("L05", "Mumbai", "Ahmedabad", "West", 524),
    ("L06", "Mumbai", "Surat", "West", 284),
    ("L07", "Chennai", "Bengaluru", "South", 346),
    ("L08", "Chennai", "Hyderabad", "South", 627),
    ("L09", "Bengaluru", "Kochi", "South", 549),
    ("L10", "Kolkata", "Bhubaneswar", "East", 441),
    ("L11", "Kolkata", "Patna", "East", 584),
    ("L12", "Delhi", "Mumbai", "Cross-region", 1415),
]

COST_PER_KM = {"FTL": 42.0, "LTL": 28.0, "PARCEL": 18.0}

SHIPMENT_COLUMNS = [
    "shipment_id",
    "carrier_id",
    "lane_id",
    "dispatch_date",
    "promised_date",
    "delivered_date",
    "status",
    "units_ordered",
    "units_delivered",
    "distance_km",
    "freight_cost",
]


@dataclass(frozen=True)
class GeneratorConfig:
    n_shipments: int = 4000
    start: date = date(2026, 7, 6)  # a Monday
    weeks: int = 12
    cancel_rate: float = 0.03
    seed: int = 7

    def __post_init__(self) -> None:
        if self.n_shipments < 1:
            raise ValueError("n_shipments must be at least 1")
        if self.weeks < 1:
            raise ValueError("weeks must be at least 1")
        if not 0 <= self.cancel_rate < 1:
            raise ValueError("cancel_rate must be in [0, 1)")


def transit_days(distance_km: int) -> int:
    """Promised transit time: one day per started 450 km, at least one day."""
    return max(1, -(-distance_km // 450))


def generate(config: GeneratorConfig | None = None) -> dict[str, list[dict]]:
    """Return ``{"shipments": [...], "carriers": [...], "lanes": [...]}`` as row dicts."""
    cfg = config or GeneratorConfig()
    rng = random.Random(cfg.seed)
    last_day = cfg.start + timedelta(days=7 * cfg.weeks - 1)
    # Shipments dispatched in the last few days are still on the road.
    snapshot = last_day + timedelta(days=1)

    carriers = [
        {"carrier_id": cid, "carrier_name": name, "mode": mode}
        for cid, name, mode, _ in CARRIERS
    ]
    lanes = [
        {
            "lane_id": lid,
            "origin_city": origin,
            "destination_city": dest,
            "region": region,
            "distance_km": km,
        }
        for lid, origin, dest, region, km in LANES
    ]

    shipments = []
    span_days = (last_day - cfg.start).days + 1
    for i in range(cfg.n_shipments):
        cid, _, mode, p_on_time = rng.choice(CARRIERS)
        lid, _, _, _, km = rng.choice(LANES)
        dispatch = cfg.start + timedelta(days=rng.randrange(span_days))
        promised = dispatch + timedelta(days=transit_days(km))
        units = rng.randint(10, 400)
        cost = round(km * COST_PER_KM[mode] * rng.uniform(0.9, 1.15), 2)

        if rng.random() < cfg.cancel_rate:
            status, delivered, units_delivered = "cancelled", None, 0
        else:
            on_time = rng.random() < p_on_time
            delay = 0 if on_time else rng.choice([1, 1, 2, 3, 5])
            early = rng.random() < 0.15 and on_time
            delivered = promised + timedelta(days=delay - (1 if early else 0))
            delivered = max(delivered, dispatch)
            if delivered >= snapshot:
                status, delivered, units_delivered = "in_transit", None, 0
            else:
                status = "delivered"
                short = rng.random() < 0.06
                units_delivered = units - rng.randint(1, max(1, units // 10)) if short else units

        shipments.append(
            {
                "shipment_id": f"S{i + 1:05d}",
                "carrier_id": cid,
                "lane_id": lid,
                "dispatch_date": dispatch.isoformat(),
                "promised_date": promised.isoformat(),
                "delivered_date": delivered.isoformat() if delivered else "",
                "status": status,
                "units_ordered": units,
                "units_delivered": units_delivered,
                "distance_km": km,
                "freight_cost": cost,
            }
        )
    return {"shipments": shipments, "carriers": carriers, "lanes": lanes}


def write_csvs(tables: dict[str, list[dict]], out_dir: str | Path) -> dict[str, Path]:
    """Write each table to ``<out_dir>/<name>.csv`` and return the paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, rows in tables.items():
        path = out / f"{name}.csv"
        fieldnames = SHIPMENT_COLUMNS if name == "shipments" else list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        paths[name] = path
    return paths


if __name__ == "__main__":  # pragma: no cover
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "data"
    for table, p in write_csvs(generate(), target).items():
        print(f"wrote {table}: {p}")
