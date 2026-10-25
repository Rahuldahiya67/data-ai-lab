"""One-step-ahead forecasting methods for intermittent demand.

Every method takes a 1-D demand history (oldest first) and returns a single
float: the forecast for the next period. An empty history forecasts 0.

Croston, SBA and TSB separate *when* demand happens from *how much* is
ordered, which is what makes them suitable for spare parts:

    Croston  smooth the non-zero sizes (z) and the intervals between them (p),
             forecast z / p. Updated only in periods with demand.
    SBA      Croston multiplied by (1 - alpha / 2), which removes Croston's
             known positive bias (Syntetos and Boylan, 2005).
    TSB      smooth the probability of demand (d) every period and the size
             (z) in periods with demand, forecast d * z. Because d decays in
             empty periods, TSB reacts to parts that are going obsolete
             (Teunter, Syntetos and Babai, 2011).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

DEFAULT_ALPHA = 0.1
DEFAULT_WINDOW = 4

Forecaster = Callable[[np.ndarray], float]


def _as_history(history) -> np.ndarray:
    """Validate a demand history.

    Missing values and negative demand are rejected rather than forecast: with
    them, Croston and TSB silently treat the value as "no demand" while SES and
    the moving average return NaN or a pulled-down level, so the methods would
    no longer be compared on the same data.
    """
    y = np.asarray(history, dtype=float)
    if y.ndim != 1:
        raise ValueError("history must be one-dimensional")
    bad = np.flatnonzero(~np.isfinite(y))
    if len(bad):
        raise ValueError(f"history has missing or infinite values at position(s) {bad[:5].tolist()}")
    neg = np.flatnonzero(y < 0)
    if len(neg):
        raise ValueError(
            f"history has negative demand at position(s) {neg[:5].tolist()}; net returns out before forecasting"
        )
    return y


def naive(history) -> float:
    """Last observed value."""
    y = _as_history(history)
    return float(y[-1]) if len(y) else 0.0


def moving_average(history, window: int = DEFAULT_WINDOW) -> float:
    """Mean of the last ``window`` periods (or all of them if fewer)."""
    if window < 1:
        raise ValueError("window must be at least 1")
    y = _as_history(history)
    return float(y[-window:].mean()) if len(y) else 0.0


def ses(history, alpha: float = DEFAULT_ALPHA) -> float:
    """Simple exponential smoothing, level initialised to the first value."""
    _check_alpha(alpha)
    y = _as_history(history)
    if not len(y):
        return 0.0
    level = y[0]
    for value in y[1:]:
        level += alpha * (value - level)
    return float(level)


def _croston_components(y: np.ndarray, alpha: float) -> tuple[float, float] | None:
    """Return smoothed (size, interval) or None if there is no demand."""
    demand_idx = np.flatnonzero(y > 0)
    if not len(demand_idx):
        return None
    first = demand_idx[0]
    z = y[first]
    p = float(first + 1)  # periods from the start of history to the first demand
    last = first
    for t in demand_idx[1:]:
        q = t - last
        z += alpha * (y[t] - z)
        p += alpha * (q - p)
        last = t
    return float(z), float(p)


def croston(history, alpha: float = DEFAULT_ALPHA) -> float:
    _check_alpha(alpha)
    comps = _croston_components(_as_history(history), alpha)
    if comps is None:
        return 0.0
    z, p = comps
    return z / p


def sba(history, alpha: float = DEFAULT_ALPHA) -> float:
    return (1 - alpha / 2) * croston(history, alpha)


def tsb(history, alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_ALPHA) -> float:
    """Teunter-Syntetos-Babai. ``alpha`` smooths size, ``beta`` smooths probability."""
    _check_alpha(alpha)
    _check_alpha(beta, "beta")
    y = _as_history(history)
    demand_idx = np.flatnonzero(y > 0)
    if not len(demand_idx):
        return 0.0
    z = y[demand_idx[0]]
    d = len(demand_idx) / len(y)
    for value in y:
        if value > 0:
            d += beta * (1 - d)
            z += alpha * (value - z)
        else:
            d += beta * (0 - d)
    return float(d * z)


def _check_alpha(value: float, name: str = "alpha") -> None:
    if not 0 < value <= 1:
        raise ValueError(f"{name} must be in (0, 1], got {value}")


METHODS: dict[str, Forecaster] = {
    "naive": naive,
    "moving_average": moving_average,
    "ses": ses,
    "croston": croston,
    "sba": sba,
    "tsb": tsb,
}
