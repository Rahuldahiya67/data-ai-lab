"""Synthetic weekly demand for a spare-parts catalogue.

Each SKU is assigned one of the four Syntetos-Boylan demand patterns and its
weekly demand is drawn from a two-part process: a Bernoulli "does anyone order
this week" draw, then an order size. The pattern controls how often demand
occurs and how variable the order sizes are:

    smooth        frequent demand, stable sizes
    erratic       frequent demand, highly variable sizes
    intermittent  sparse demand, stable sizes
    lumpy         sparse demand, highly variable sizes
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PATTERNS = ("smooth", "erratic", "intermittent", "lumpy")


@dataclass(frozen=True)
class PatternSpec:
    """Parameters of the demand process for one pattern."""

    occurrence_prob: float  # probability that a given week has any demand
    size_mean: float  # mean order size when demand occurs
    variable_sizes: bool  # True -> lognormal sizes with high CV, False -> near-normal sizes


PATTERN_SPECS: dict[str, PatternSpec] = {
    "smooth": PatternSpec(occurrence_prob=0.95, size_mean=20.0, variable_sizes=False),
    "erratic": PatternSpec(occurrence_prob=0.95, size_mean=20.0, variable_sizes=True),
    "intermittent": PatternSpec(occurrence_prob=0.30, size_mean=6.0, variable_sizes=False),
    "lumpy": PatternSpec(occurrence_prob=0.30, size_mean=6.0, variable_sizes=True),
}

# Lognormal sigma for variable sizes. CV^2 of a lognormal is exp(sigma^2) - 1,
# so sigma = 1.0 gives CV^2 of about 1.7, well above the 0.49 cut-off.
_VARIABLE_SIGMA = 1.0
# Relative standard deviation for stable sizes (CV^2 = 0.0625).
_STABLE_CV = 0.25


def _order_sizes(rng: np.random.Generator, spec: PatternSpec, n: int) -> np.ndarray:
    if spec.variable_sizes:
        # Choose mu so that the lognormal mean equals size_mean.
        mu = np.log(spec.size_mean) - _VARIABLE_SIGMA**2 / 2
        sizes = rng.lognormal(mean=mu, sigma=_VARIABLE_SIGMA, size=n)
    else:
        sizes = rng.normal(loc=spec.size_mean, scale=_STABLE_CV * spec.size_mean, size=n)
    return np.maximum(np.rint(sizes), 1).astype(int)


def generate_demand(
    n_skus: int = 40,
    n_weeks: int = 104,
    seed: int = 42,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """Return weekly demand in long format.

    Columns: sku, pattern (the pattern used to generate it), week (Monday of
    the week), demand (non-negative int). SKUs are spread evenly across the
    four patterns. Output is fully determined by ``seed``.
    """
    if n_skus < 1:
        raise ValueError("n_skus must be at least 1")
    if n_weeks < 1:
        raise ValueError("n_weeks must be at least 1")

    rng = np.random.default_rng(seed)
    weeks = pd.date_range(start=start, periods=n_weeks, freq="W-MON")
    frames = []
    for i in range(n_skus):
        pattern = PATTERNS[i % len(PATTERNS)]
        spec = PATTERN_SPECS[pattern]
        occurs = rng.random(n_weeks) < spec.occurrence_prob
        demand = np.zeros(n_weeks, dtype=int)
        demand[occurs] = _order_sizes(rng, spec, int(occurs.sum()))
        frames.append(
            pd.DataFrame(
                {
                    "sku": f"PART-{i + 1:04d}",
                    "pattern": pattern,
                    "week": weeks,
                    "demand": demand,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
