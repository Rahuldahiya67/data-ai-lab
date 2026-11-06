"""Shipment schema and row-level validation.

Every batch the monitor sees (the reference period or one production week) goes
through ``validate`` first. Rows that cannot be scored are not silently dropped:
they come back in a rejected frame with a reason, because a sudden rise in
rejects is itself a sign that an upstream feed has changed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REGIONS = ("north", "south", "east", "west", "central")
VEHICLE_TYPES = ("bike", "van", "truck")

NUMERIC_FEATURES = ("distance_km", "stops", "dispatch_hour")
CATEGORICAL_FEATURES = ("region", "vehicle_type")
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

REQUIRED_COLUMNS = ("shipment_id", "week") + FEATURES
# actual_hours is optional per row: it is empty until the delivery completes.
LABEL = "actual_hours"

# Hard limits for a single domestic shipment. Anything outside is a data error,
# not an unusual shipment. Distance above 5,000 km is the typical symptom of a
# feed that switched to metres.
LIMITS = {
    "distance_km": (0.1, 5000.0),
    "stops": (1, 60),
    "dispatch_hour": (0, 23),
    "actual_hours": (0.1, 24.0 * 14),
}


class SchemaError(ValueError):
    """The batch as a whole is unusable (for example, a column is missing)."""


@dataclass(frozen=True)
class ValidationResult:
    valid: pd.DataFrame
    rejected: pd.DataFrame  # original columns plus a ``reason`` column

    @property
    def reject_rate(self) -> float:
        total = len(self.valid) + len(self.rejected)
        return 0.0 if total == 0 else len(self.rejected) / total


def _reason_for(row: pd.Series) -> str | None:
    for col in NUMERIC_FEATURES:
        value = row[col]
        if pd.isna(value):
            return f"{col} is missing"
        lo, hi = LIMITS[col]
        if not lo <= value <= hi:
            return f"{col}={value:g} outside [{lo:g}, {hi:g}]"
    if row["dispatch_hour"] != int(row["dispatch_hour"]):
        return "dispatch_hour is not a whole hour"
    if row["stops"] != int(row["stops"]):
        return "stops is not a whole number"
    for col in CATEGORICAL_FEATURES:
        if pd.isna(row[col]) or str(row[col]).strip() == "":
            return f"{col} is missing"
    label = row.get(LABEL)
    if label is not None and not pd.isna(label):
        lo, hi = LIMITS[LABEL]
        if not lo <= label <= hi:
            return f"{LABEL}={label:g} outside [{lo:g}, {hi:g}]"
    return None


def validate(df: pd.DataFrame) -> ValidationResult:
    """Split a batch into rows the model can score and rejected rows with a reason.

    Categorical values are normalised to lower case with surrounding spaces
    removed. Unknown categories are *kept*: they are drift, not bad data, and
    the drift report counts them.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"batch is missing required columns: {', '.join(missing)}")

    work = df.copy()
    if LABEL not in work.columns:
        work[LABEL] = np.nan
    for col in NUMERIC_FEATURES + (LABEL,):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        work[col] = work[col].where(work[col].isna(), work[col].astype(str).str.strip().str.lower())

    duplicated = work["shipment_id"].duplicated(keep="first")
    reasons = work.apply(_reason_for, axis=1)
    reasons = reasons.where(~duplicated, "duplicate shipment_id")

    bad = reasons.notna()
    rejected = work[bad].assign(reason=reasons[bad]).reset_index(drop=True)
    valid = work[~bad].reset_index(drop=True)
    valid["stops"] = valid["stops"].astype(int)
    valid["dispatch_hour"] = valid["dispatch_hour"].astype(int)
    return ValidationResult(valid=valid, rejected=rejected)
