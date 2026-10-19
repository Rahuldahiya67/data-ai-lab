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

import warnings
from typing import Iterable

import numpy as np
import pandas as pd

from .classify import classify_series
from .methods import METHODS

SELECTION_METRIC = "rmsse"
# Used when no method can be scored for a SKU (no demand or constant demand in
# the training window). TSB decays towards zero when demand stops, so it is the
# safest default for new and dormant parts.
FALLBACK_METHOD = "tsb"
# The naive scale needs at least two training weeks.
MIN_TRAIN_WEEKS = 2


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

    SKUs with fewer than ``holdout + MIN_TRAIN_WEEKS`` weeks cannot be scored;
    they are left out and named in a warning.
    """
    if holdout < 1:
        raise ValueError("holdout must be at least 1")
    methods = list(methods) if methods is not None else list(METHODS)
    rows = []
    too_short = []
    for sku, group in df.sort_values("week").groupby("sku", sort=True):
        y = group["demand"].to_numpy(dtype=float)
        if len(y) < holdout + MIN_TRAIN_WEEKS:
            too_short.append(str(sku))
            continue
        train, actual = y[:-holdout], y[-holdout:]
        demand_class = classify_series(train).demand_class
        for method in methods:
            forecast = rolling_forecasts(y, method, holdout)
            rows.append(
                {"sku": sku, "demand_class": demand_class, "method": method}
                | score(actual, forecast, train)
            )
    if too_short:
        warnings.warn(
            f"skipped {len(too_short)} SKU(s) with fewer than {holdout + MIN_TRAIN_WEEKS} "
            f"weeks of history: {', '.join(too_short)}",
            stacklevel=2,
        )
    columns = ["sku", "demand_class", "method", "mase", "rmsse", "bias"]
    return pd.DataFrame(rows, columns=columns)


def select_best(results: pd.DataFrame, metric: str = SELECTION_METRIC) -> pd.DataFrame:
    """Pick the method with the lowest ``metric`` for each SKU.

    Ties go to the method listed first in the results. SKUs where no method
    has a finite score get :data:`FALLBACK_METHOD`. The ``selected_by`` column
    says which rule applied ("metric" or "fallback").
    """
    ordered = results.reset_index(drop=True)
    scored = ordered[np.isfinite(ordered[metric])]
    idx = scored.groupby("sku", sort=True)[metric].idxmin()
    best = ordered.loc[idx].assign(selected_by="metric")

    unscored = sorted(set(ordered["sku"]) - set(best["sku"]))
    fallback = ordered[ordered["sku"].isin(unscored) & (ordered["method"] == FALLBACK_METHOD)]
    missing = set(unscored) - set(fallback["sku"])
    if missing:
        raise ValueError(
            f"no finite {metric} and no {FALLBACK_METHOD} row for: {', '.join(sorted(missing))}"
        )
    fallback = fallback.assign(selected_by="fallback")
    return pd.concat([best, fallback]).sort_values("sku").reset_index(drop=True)


def summarise(results: pd.DataFrame, metric: str = SELECTION_METRIC) -> pd.DataFrame:
    """Mean of ``metric`` per demand class and method (lower is better)."""
    return (
        results.pivot_table(index="demand_class", columns="method", values=metric, aggfunc="mean")
        .round(3)
    )
