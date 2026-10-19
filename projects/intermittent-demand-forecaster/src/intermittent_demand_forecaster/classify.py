"""Demand pattern classification (Syntetos-Boylan-Croston scheme).

Two statistics describe a demand history:

    ADI  average demand interval = number of periods / number of periods with demand
    CV2  squared coefficient of variation of the non-zero demand sizes

The standard cut-offs (Syntetos, Boylan and Croston, 2005) split SKUs into
four classes:

                  CV2 < 0.49     CV2 >= 0.49
    ADI < 1.32    smooth         erratic
    ADI >= 1.32   intermittent   lumpy
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .prepare import prepare_demand

ADI_CUTOFF = 1.32
CV2_CUTOFF = 0.49


@dataclass(frozen=True)
class DemandStats:
    n_periods: int
    n_demand_periods: int
    adi: float
    cv2: float
    demand_class: str


def classify_series(demand: np.ndarray | pd.Series) -> DemandStats:
    """Compute ADI, CV2 and the demand class for one SKU's history.

    A history with no demand at all is classed as ``no_demand``. With a single
    demand period CV2 is 0 (there is no variation to measure yet). CV2 uses
    the population standard deviation (ddof=0) of the non-zero sizes.
    """
    y = np.asarray(demand, dtype=float)
    if y.ndim != 1:
        raise ValueError("demand must be one-dimensional")
    if np.any(y < 0):
        raise ValueError("demand must be non-negative")

    n = len(y)
    sizes = y[y > 0]
    k = len(sizes)
    if k == 0:
        return DemandStats(n, 0, float("inf"), float("nan"), "no_demand")

    adi = n / k
    mean = sizes.mean()
    cv2 = float((sizes.std() / mean) ** 2)

    if adi < ADI_CUTOFF:
        demand_class = "smooth" if cv2 < CV2_CUTOFF else "erratic"
    else:
        demand_class = "intermittent" if cv2 < CV2_CUTOFF else "lumpy"
    return DemandStats(n, k, float(adi), cv2, demand_class)


def classify_skus(df: pd.DataFrame) -> pd.DataFrame:
    """Classify every SKU in a long-format frame with columns sku, week, demand.

    The frame goes through :func:`prepare_demand` first, so weeks without
    rows count as zero demand.
    """
    rows = []
    for sku, group in prepare_demand(df).groupby("sku", sort=True):
        stats = classify_series(group["demand"].to_numpy())
        rows.append(
            {
                "sku": sku,
                "n_periods": stats.n_periods,
                "n_demand_periods": stats.n_demand_periods,
                "adi": stats.adi,
                "cv2": stats.cv2,
                "demand_class": stats.demand_class,
            }
        )
    return pd.DataFrame(rows)
