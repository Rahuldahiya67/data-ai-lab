"""Reorder points from the selected forecasting method.

For each SKU, with lead time L weeks and target cycle service level s:

    forecast            the selected method's one-step forecast on the full history
    lead-time demand    L * forecast
    sigma (lead time)   sqrt(L) * RMSE of that method in the backtest
    safety stock        z(s) * sigma, where z is the standard normal quantile
    reorder point       ceil(lead-time demand + safety stock)

Using the backtest RMSE ties safety stock to how wrong the chosen method
actually was on recent weeks, rather than to the raw demand variance. The
sqrt(L) step assumes one-step errors are independent and the normal quantile
is an approximation; for very sparse parts both are rough, which is why the
report also shows the demand class.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from .backtest import check_whole_number
from .methods import METHODS
from .prepare import prepare_demand

def _check_policy(lead_time, service_level) -> tuple[int, float]:
    """Validate the stocking policy and return (lead time, z for the service level)."""
    lead_time = check_whole_number("lead_time", lead_time)
    if isinstance(service_level, bool) or not isinstance(service_level, (int, float, np.floating)):
        raise ValueError(f"service_level must be a number, got {service_level!r}")
    if not 0.5 <= service_level < 1:  # also rejects NaN
        raise ValueError(f"service_level must be in [0.5, 1), got {service_level}")
    return lead_time, NormalDist().inv_cdf(service_level)


REPORT_COLUMNS = [
    "sku",
    "demand_class",
    "method",
    "selected_by",
    "forecast_per_week",
    "lead_time_weeks",
    "lead_time_demand",
    "sigma_lead_time",
    "safety_stock",
    "reorder_point",
]


def reorder_points(
    df: pd.DataFrame,
    best: pd.DataFrame,
    lead_time: int = 4,
    service_level: float = 0.95,
) -> pd.DataFrame:
    """Reorder point per SKU.

    ``df`` is the demand data (any shape :func:`prepare_demand` accepts) and
    ``best`` is the output of :func:`select_best`. Returns one row per SKU in
    ``best``, columns :data:`REPORT_COLUMNS`.
    """
    lead_time, z = _check_policy(lead_time, service_level)

    series = {sku: g["demand"].to_numpy() for sku, g in prepare_demand(df).groupby("sku")}
    rows = []
    for row in best.itertuples(index=False):
        if row.sku not in series:
            raise ValueError(f"no demand data for SKU {row.sku}")
        forecast = METHODS[row.method](series[row.sku])
        ltd = lead_time * forecast
        sigma = math.sqrt(lead_time) * row.rmse
        safety = z * sigma
        rows.append(
            {
                "sku": row.sku,
                "demand_class": row.demand_class,
                "method": row.method,
                "selected_by": row.selected_by,
                "forecast_per_week": round(forecast, 3),
                "lead_time_weeks": int(lead_time),
                "lead_time_demand": round(ltd, 2),
                "sigma_lead_time": round(sigma, 2),
                "safety_stock": round(safety, 2),
                "reorder_point": int(math.ceil(ltd + safety - 1e-9)),
            }
        )
    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


SERVICE_CHECK_COLUMNS = ["sku", "demand_class", "method", "windows", "covered", "achieved_service"]


def service_check(
    df: pd.DataFrame,
    best: pd.DataFrame,
    lead_time: int = 4,
    service_level: float = 0.95,
    holdout: int = 26,
) -> pd.DataFrame:
    """How often the reorder point would have covered lead-time demand.

    For every origin week t in the last ``holdout`` weeks that still has a
    full lead time after it, the reorder point is rebuilt from the selected
    method's forecast on weeks before t (same formula as
    :func:`reorder_points`) and compared with the actual demand in weeks
    t .. t + lead_time - 1. ``achieved_service`` is the share of those
    windows where demand did not exceed the reorder point; compare it with
    the target ``service_level``.

    The RMSE used for safety stock comes from the same holdout, so this is an
    optimistic check of the normal approximation, not a clean out-of-sample
    test. SKUs with no complete window get NaN.
    """
    lead_time, z = _check_policy(lead_time, service_level)
    holdout = check_whole_number("holdout", holdout)

    series = {sku: g["demand"].to_numpy() for sku, g in prepare_demand(df).groupby("sku")}
    rows = []
    for row in best.itertuples(index=False):
        if row.sku not in series:
            raise ValueError(f"no demand data for SKU {row.sku}")
        y = series[row.sku]
        safety = z * math.sqrt(lead_time) * row.rmse
        first = max(len(y) - holdout, 1)
        windows = covered = 0
        for t in range(first, len(y) - lead_time + 1):
            rop = math.ceil(lead_time * METHODS[row.method](y[:t]) + safety - 1e-9)
            windows += 1
            covered += int(y[t : t + lead_time].sum() <= rop)
        rows.append(
            {
                "sku": row.sku,
                "demand_class": row.demand_class,
                "method": row.method,
                "windows": windows,
                "covered": covered,
                "achieved_service": covered / windows if windows else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=SERVICE_CHECK_COLUMNS)
