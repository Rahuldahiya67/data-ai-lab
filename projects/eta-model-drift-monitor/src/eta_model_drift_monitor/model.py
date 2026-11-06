"""Baseline ETA model: ridge regression in NumPy, saved as plain JSON.

The point of this project is the monitor, not the model, so the model is
deliberately small and transparent. It still has to be a fair model of the
data: travel time depends on distance *per vehicle type* (a truck covers a
kilometre faster than a bike), so distance enters as one column per vehicle.

Categories seen at fit time get a one-hot column. A category the model has
never seen gets all zeros in its one-hot block and, for vehicle type, the
average per-km coefficient, so it still gets a sensible prediction instead of
"distance does not matter". The drift report is where unseen categories are
flagged; the model only has to stay usable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import LABEL

PEAK_HOURS = frozenset({8, 9, 10, 17, 18, 19})


@dataclass
class EtaModel:
    regions: list[str]
    vehicle_types: list[str]
    coef: np.ndarray  # aligned with feature_names()
    alpha: float = 1.0

    def feature_names(self) -> list[str]:
        return (
            ["intercept"]
            + [f"km_per_{v}" for v in self.vehicle_types]
            + ["stops", "peak_dispatch"]
            + [f"region={r}" for r in self.regions]
            + [f"vehicle={v}" for v in self.vehicle_types]
        )

    @staticmethod
    def _design(df: pd.DataFrame, regions: list[str], vehicle_types: list[str], km_fallback: bool) -> np.ndarray:
        n = len(df)
        distance = df["distance_km"].to_numpy(dtype=float)
        vehicle = df["vehicle_type"].to_numpy()
        region = df["region"].to_numpy()
        known_vehicle = np.isin(vehicle, vehicle_types)

        cols = [np.ones(n)]
        for v in vehicle_types:
            col = np.where(vehicle == v, distance, 0.0)
            if km_fallback:
                # Unseen vehicle: spread the distance evenly over the known
                # per-km columns, which applies the average per-km coefficient.
                col = np.where(known_vehicle, col, distance / len(vehicle_types))
            cols.append(col)
        cols.append(df["stops"].to_numpy(dtype=float))
        cols.append(np.isin(df["dispatch_hour"].to_numpy(), list(PEAK_HOURS)).astype(float))
        cols.extend((region == r).astype(float) for r in regions)
        cols.extend((vehicle == v).astype(float) for v in vehicle_types)
        return np.column_stack(cols)

    @classmethod
    def fit(cls, df: pd.DataFrame, alpha: float = 1.0) -> "EtaModel":
        """Fit on validated rows that have an actual delivery time."""
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        train = df[df[LABEL].notna()]
        if len(train) < 20:
            raise ValueError(f"need at least 20 labelled rows to fit, got {len(train)}")
        regions = sorted(train["region"].unique().tolist())
        vehicle_types = sorted(train["vehicle_type"].unique().tolist())
        X = cls._design(train, regions, vehicle_types, km_fallback=False)
        y = train[LABEL].to_numpy(dtype=float)
        penalty = alpha * np.eye(X.shape[1])
        penalty[0, 0] = 0.0  # do not shrink the intercept
        coef = np.linalg.solve(X.T @ X + penalty, X.T @ y)
        return cls(regions=regions, vehicle_types=vehicle_types, coef=coef, alpha=alpha)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._design(df, self.regions, self.vehicle_types, km_fallback=True)
        # A delivery cannot take negative time; clip the rare extrapolation.
        return np.clip(X @ self.coef, 0.1, None)

    def to_dict(self) -> dict[str, object]:
        return {
            "regions": self.regions,
            "vehicle_types": self.vehicle_types,
            "alpha": self.alpha,
            "coef": dict(zip(self.feature_names(), self.coef.tolist())),
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "EtaModel":
        raw = json.loads(path.read_text())
        model = cls(regions=raw["regions"], vehicle_types=raw["vehicle_types"], coef=np.zeros(0), alpha=raw["alpha"])
        names = model.feature_names()
        missing = [n for n in names if n not in raw["coef"]]
        if missing:
            raise ValueError(f"model file is missing coefficients: {', '.join(missing)}")
        model.coef = np.array([raw["coef"][n] for n in names], dtype=float)
        return model
