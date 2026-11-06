"""Synthetic shipments with planted drift scenarios and a ground-truth file.

The data has a *reference period* (the weeks the model is trained on and the
monitor compares against) followed by *monitoring weeks*. Three kinds of
problem are planted in separate windows, each listed in the ground truth:

``mix_shift``
    The input mix changes: many more long-haul truck loads. Inputs drift, but
    the true relationship between inputs and delivery time does not.
``unit_bug``
    One carrier's feed in the east region starts sending distance in metres.
    Long trips are rejected by validation; short ones slip through looking
    like very long trips.
``slowdown``
    Monsoon in the north: deliveries there take longer for the same inputs.
    The inputs look normal; only the errors show it, once actuals arrive.

Everything is driven by one seed, so the same config always produces the same
rows.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import REGIONS, VEHICLE_TYPES

REGION_WEIGHTS = {"north": 0.25, "south": 0.25, "east": 0.15, "west": 0.20, "central": 0.15}
VEHICLE_WEIGHTS = {"bike": 0.30, "van": 0.50, "truck": 0.20}
# Truck share during a mix shift; the other vehicles share the rest pro rata.
MIX_SHIFT_TRUCK_SHARE = 0.45

# (median km, lognormal sigma, min km, max km) by vehicle
DISTANCE = {
    "bike": (8.0, 0.5, 1.0, 30.0),
    "van": (60.0, 0.6, 5.0, 300.0),
    "truck": (450.0, 0.5, 80.0, 1800.0),
}
STOPS = {"bike": (1, 4), "van": (2, 15), "truck": (1, 6)}
SPEED_KMH = {"bike": 18.0, "van": 35.0, "truck": 48.0}
STOP_HOURS = 0.35
REGION_DELAY_HOURS = {"north": 1.0, "south": 0.5, "east": 1.5, "west": 0.8, "central": 0.3}
PEAK_HOURS = frozenset({8, 9, 10, 17, 18, 19})
PEAK_DELAY_HOURS = 1.2
NOISE_SIGMA = 0.12  # multiplicative log-normal noise on true hours

SLOWDOWN_REGION = "north"
SLOWDOWN_FACTOR = 1.35
UNIT_BUG_REGION = "east"
UNIT_BUG_SHARE = 0.4  # share of that region's rows coming from the broken feed


@dataclass(frozen=True)
class Scenario:
    kind: str  # "mix_shift" | "unit_bug" | "slowdown"
    start_week: int
    end_week: int  # inclusive

    def active(self, week: int) -> bool:
        return self.start_week <= week <= self.end_week


def default_scenarios() -> tuple[Scenario, ...]:
    return (
        Scenario("mix_shift", 12, 14),
        Scenario("unit_bug", 17, 18),
        Scenario("slowdown", 20, 24),
    )


@dataclass(frozen=True)
class GeneratorConfig:
    reference_weeks: int = 8
    monitoring_weeks: int = 16
    shipments_per_week: int = 600
    missing_rate: float = 0.004  # share of rows with one feature blanked
    seed: int = 42
    scenarios: tuple[Scenario, ...] = field(default_factory=default_scenarios)

    def __post_init__(self) -> None:
        if self.reference_weeks < 1 or self.monitoring_weeks < 0:
            raise ValueError("reference_weeks must be >= 1 and monitoring_weeks >= 0")
        if self.shipments_per_week < 1:
            raise ValueError("shipments_per_week must be >= 1")
        if not 0.0 <= self.missing_rate < 1.0:
            raise ValueError("missing_rate must be in [0, 1)")
        last = self.reference_weeks + self.monitoring_weeks
        for s in self.scenarios:
            if s.kind not in {"mix_shift", "unit_bug", "slowdown"}:
                raise ValueError(f"unknown scenario kind: {s.kind!r}")
            if not self.reference_weeks < s.start_week <= s.end_week <= last:
                raise ValueError(
                    f"scenario {s.kind} weeks {s.start_week}-{s.end_week} must fall inside "
                    f"the monitoring weeks {self.reference_weeks + 1}-{last}"
                )

    @property
    def weeks(self) -> range:
        return range(1, self.reference_weeks + self.monitoring_weeks + 1)


def true_hours(distance_km, stops, dispatch_hour, region, vehicle_type) -> np.ndarray:
    """Noise-free delivery time in hours under normal conditions (vectorised)."""
    speed = pd.Series(vehicle_type).map(SPEED_KMH).to_numpy(dtype=float)
    delay = pd.Series(region).map(REGION_DELAY_HOURS).to_numpy(dtype=float)
    peak = np.isin(np.asarray(dispatch_hour), list(PEAK_HOURS)) * PEAK_DELAY_HOURS
    return np.asarray(distance_km, dtype=float) / speed + STOP_HOURS * np.asarray(stops) + delay + peak


def _week(cfg: GeneratorConfig, week: int, rng: np.random.Generator) -> pd.DataFrame:
    active = {s.kind for s in cfg.scenarios if s.active(week)}
    n = cfg.shipments_per_week

    vehicle_weights = dict(VEHICLE_WEIGHTS)
    if "mix_shift" in active:
        rest = 1.0 - VEHICLE_WEIGHTS["truck"]
        vehicle_weights = {
            v: MIX_SHIFT_TRUCK_SHARE if v == "truck" else w / rest * (1.0 - MIX_SHIFT_TRUCK_SHARE)
            for v, w in VEHICLE_WEIGHTS.items()
        }

    region = rng.choice(REGIONS, size=n, p=[REGION_WEIGHTS[r] for r in REGIONS])
    vehicle = rng.choice(VEHICLE_TYPES, size=n, p=[vehicle_weights[v] for v in VEHICLE_TYPES])

    distance = np.empty(n)
    stops = np.empty(n, dtype=int)
    for v in VEHICLE_TYPES:
        mask = vehicle == v
        median, sigma, lo, hi = DISTANCE[v]
        distance[mask] = np.clip(rng.lognormal(np.log(median), sigma, mask.sum()), lo, hi)
        s_lo, s_hi = STOPS[v]
        stops[mask] = rng.integers(s_lo, s_hi + 1, mask.sum())
    distance = np.round(distance, 1)

    # Dispatches cluster in the morning and early evening.
    hours = np.arange(6, 22)
    hour_weights = np.where(np.isin(hours, list(PEAK_HOURS)), 2.0, 1.0)
    dispatch_hour = rng.choice(hours, size=n, p=hour_weights / hour_weights.sum())

    actual = true_hours(distance, stops, dispatch_hour, region, vehicle)
    actual = actual * rng.lognormal(0.0, NOISE_SIGMA, n)
    if "slowdown" in active:
        actual = np.where(region == SLOWDOWN_REGION, actual * SLOWDOWN_FACTOR, actual)

    df = pd.DataFrame(
        {
            "shipment_id": [f"W{week:02d}-{i:05d}" for i in range(n)],
            "week": week,
            "distance_km": distance,
            "stops": stops,
            "dispatch_hour": dispatch_hour,
            "region": region,
            "vehicle_type": vehicle,
            "actual_hours": np.round(actual, 2),
        }
    )

    if "unit_bug" in active:
        broken = (region == UNIT_BUG_REGION) & (rng.random(n) < UNIT_BUG_SHARE)
        df.loc[broken, "distance_km"] = df.loc[broken, "distance_km"] * 1000.0

    if cfg.missing_rate > 0:
        blank = rng.random(n) < cfg.missing_rate
        cols = rng.choice(["distance_km", "region", "vehicle_type"], size=n)
        for col in ("distance_km", "region", "vehicle_type"):
            df.loc[blank & (cols == col), col] = None
    return df


def generate(cfg: GeneratorConfig = GeneratorConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(shipments, ground_truth)``.

    ``shipments`` holds every week, reference and monitoring, with actuals.
    ``ground_truth`` has one row per planted scenario.
    """
    rng = np.random.default_rng(cfg.seed)
    shipments = pd.concat([_week(cfg, w, rng) for w in cfg.weeks], ignore_index=True)
    truth = pd.DataFrame(
        [{"kind": s.kind, "start_week": s.start_week, "end_week": s.end_week} for s in cfg.scenarios],
        columns=["kind", "start_week", "end_week"],
    )
    return shipments, truth


def write_dataset(out_dir: Path, cfg: GeneratorConfig = GeneratorConfig()) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    shipments, truth = generate(cfg)
    ship_path = out_dir / "shipments.csv"
    truth_path = out_dir / "scenarios.csv"
    shipments.to_csv(ship_path, index=False, quoting=csv.QUOTE_MINIMAL)
    truth.to_csv(truth_path, index=False)
    return ship_path, truth_path


if __name__ == "__main__":  # pragma: no cover
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    paths = write_dataset(target)
    print("wrote", *paths)
