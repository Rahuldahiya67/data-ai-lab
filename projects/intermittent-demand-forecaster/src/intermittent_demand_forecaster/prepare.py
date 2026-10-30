"""Turn raw SKU-week demand rows into complete weekly series.

Classification and forecasting both need one row per SKU per week, with the
weeks without orders present as zeros. Order-line extracts rarely look like
that: they hold only weeks with orders, sometimes several rows per week, and
dates that are not always the Monday of the week. :func:`prepare_demand`
fixes all of this in one place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("sku", "week", "demand")


def _examples(values: pd.Series, limit: int = 3) -> str:
    """Up to ``limit`` distinct offending values with their row labels, for error messages."""
    shown = [f"{v!r} (row {i})" for i, v in values.drop_duplicates().head(limit).items()]
    more = len(values) - len(shown)
    return ", ".join(shown) + (f" and {more} more" if more > 0 else "")


def prepare_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Return a complete weekly series per SKU with columns sku, week, demand.

    - dates are moved to the Monday of their week,
    - rows for the same SKU and week are summed,
    - every week from a SKU's first row to the last week in the data is
      present, with zero demand where there were no rows. Trailing zeros
      matter: a part that stopped selling must look like one.

    Raises ``ValueError`` on missing columns, missing values, demand that is
    not a finite non-negative number, or a week that is not a date. The
    message names the offending values and their rows.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing column(s): {', '.join(missing)}")
    data = df.loc[:, list(REQUIRED_COLUMNS)].copy()
    if data.isna().any().any():
        bad = data.columns[data.isna().any()].tolist()
        raise ValueError(f"missing values in column(s): {', '.join(bad)}")
    demand = pd.to_numeric(data["demand"], errors="coerce")
    if demand.isna().any():
        raise ValueError(f"demand must be numeric, got {_examples(data['demand'][demand.isna()])}")
    data["demand"] = demand.astype(float)
    if not np.isfinite(data["demand"]).all():
        raise ValueError(f"demand must be finite, got {_examples(data['demand'][~np.isfinite(data['demand'])])}")
    if (data["demand"] < 0).any():
        raise ValueError("demand must be non-negative")
    if data.empty:
        return data.astype({"demand": float})

    weeks = pd.to_datetime(data["week"], errors="coerce")
    if weeks.isna().any():
        raise ValueError(f"week must be a date, could not parse {_examples(data['week'][weeks.isna()])}")
    data["week"] = weeks.dt.to_period("W-SUN").dt.start_time
    data = data.groupby(["sku", "week"], as_index=False, sort=True)["demand"].sum()

    last_week = data["week"].max()
    frames = []
    for sku, group in data.groupby("sku", sort=True):
        weeks = pd.date_range(group["week"].min(), last_week, freq="W-MON")
        series = group.set_index("week")["demand"].reindex(weeks, fill_value=0.0)
        frames.append(
            pd.DataFrame({"sku": sku, "week": weeks, "demand": series.to_numpy()})
        )
    return pd.concat(frames, ignore_index=True)
