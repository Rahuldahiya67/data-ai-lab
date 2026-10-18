"""Rolling-origin backtest and per-SKU method selection.

For each SKU the last ``holdout`` weeks are forecast one step at a time: the
forecast for week t only sees weeks before t. Errors are scaled by the
in-sample one-step naive error of the training part, so they can be compared
across SKUs with very different volumes.

    MASE   mean absolute error / mean absolute naive error (training part)
    RMSSE  root mean squared error / root mean squared naive error
    bias   mean (forecast - actual) / mean training demand

Selection uses RMSSE, not MASE. For intermittent series the median of demand
is often zero, and MAE-based metrics reward a forecast of zero, which would
leave the warehouse with no stock. Squared error targets the mean, which is
what stocking decisions need.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .classify import classify_series
from .methods import METHODS

SELECTION_METRIC = "rmsse"


def rolling_forecasts(y: np.ndarray, method: str, holdout: int) -> np.ndarray:
    """One-step-ahead forecasts for the last ``holdout`` periods of ``y``."""
    if method not in METHODS:
        raise KeyError(f"unknown method: {method}")
    if not 0 < holdout < len(y):
        raise ValueError("holdout must be between 1 and len(y) - 1")
    fn = METHODS[method]
    start = len(y) - holdout
    return np.array([fn(y[:t]) for t in range(start, len(y))])


def score(actual: np.ndarray, forecast: np.ndarray, train: np.ndarray) -> dict[str, float]:
    """Scaled error metrics. Scales of zero (flat training data) give NaN."""
    err = forecast - actual
    naive_err = np.diff(train)
    mae_scale = np.abs(naive_err).mean() if len(naive_err) else 0.0
    mse_scale = (naive_err**2).mean() if len(naive_err) else 0.0
    level = train.mean() if len(train) else 0.0

    def ratio(num: float, den: float) -> float:
        return float(num / den) if den > 0 else float("nan")

    return {
        "mase": ratio(np.abs(err).mean(), mae_scale),
        "rmsse": ratio(np.sqrt((err**2).mean()), np.sqrt(mse_scale)),
        "bias": ratio(err.mean(), level),
    }


def backtest(
    df: pd.DataFrame,
    holdout: int = 26,
    methods: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Score every method on every SKU.

    ``df`` is long format with columns sku, week, demand. Returns one row per
    (sku, method) with the demand class (computed on the training part only,
    so the holdout does not leak into it) and the metrics from :func:`score`.
    """
    methods = list(methods) if methods is not None else list(METHODS)
    rows = []
    for sku, group in df.sort_values("week").groupby("sku", sort=True):
        y = group["demand"].to_numpy(dtype=float)
        train, actual = y[:-holdout], y[-holdout:]
        demand_class = classify_series(train).demand_class
        for method in methods:
            forecast = rolling_forecasts(y, method, holdout)
            rows.append(
                {"sku": sku, "demand_class": demand_class, "method": method}
                | score(actual, forecast, train)
            )
    return pd.DataFrame(rows)


def select_best(results: pd.DataFrame, metric: str = SELECTION_METRIC) -> pd.DataFrame:
    """Pick the method with the lowest ``metric`` for each SKU.

    Ties go to the method listed first in the results.
    """
    ordered = results.reset_index(drop=True)
    idx = ordered.groupby("sku", sort=True)[metric].idxmin()
    return ordered.loc[idx].reset_index(drop=True)


def summarise(results: pd.DataFrame, metric: str = SELECTION_METRIC) -> pd.DataFrame:
    """Mean of ``metric`` per demand class and method (lower is better)."""
    return (
        results.pivot_table(index="demand_class", columns="method", values=metric, aggfunc="mean")
        .round(3)
    )
