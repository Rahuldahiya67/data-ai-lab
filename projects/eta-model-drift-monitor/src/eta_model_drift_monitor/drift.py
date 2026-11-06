"""Input and prediction drift with the Population Stability Index (PSI).

PSI compares how a variable is spread across bins in the reference period
and in the current batch:

    PSI = sum over bins of (cur% - ref%) * ln(cur% / ref%)

Common rule of thumb, used here as defaults: below 0.10 is stable, 0.10 to
0.25 is a moderate shift worth watching, above 0.25 is a major shift.

Numeric bins are quantiles of the *reference* data, so each reference bin
holds about the same share and the cut points stay fixed week to week. The
outer bins are open-ended, so values beyond anything seen in the reference
period still land in a bin (and push PSI up, which is what we want).
Categorical bins are the reference categories plus one ``<unseen>`` bin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES

UNSEEN = "<unseen>"
EPSILON = 1e-4  # floor for empty bins so the log term stays finite
WARN_PSI = 0.10
ALERT_PSI = 0.25


def status_for(psi: float, warn: float = WARN_PSI, alert: float = ALERT_PSI) -> str:
    if psi >= alert:
        return "alert"
    if psi >= warn:
        return "warn"
    return "ok"


def _psi_from_shares(ref_share: np.ndarray, cur_share: np.ndarray) -> float:
    ref = np.maximum(ref_share, EPSILON)
    cur = np.maximum(cur_share, EPSILON)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def numeric_edges(reference: pd.Series, bins: int = 10) -> np.ndarray:
    """Inner cut points from reference quantiles (duplicates removed)."""
    values = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError("reference has no numeric values to bin")
    if bins < 2:
        raise ValueError("bins must be >= 2")
    qs = np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1])
    return np.unique(qs)


def _numeric_shares(values: pd.Series, edges: np.ndarray) -> np.ndarray:
    v = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    counts = np.bincount(np.searchsorted(edges, v, side="right"), minlength=len(edges) + 1)
    return counts / max(v.size, 1)


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    edges = numeric_edges(reference, bins)
    if pd.to_numeric(current, errors="coerce").dropna().empty:
        raise ValueError("current batch has no numeric values")
    return _psi_from_shares(_numeric_shares(reference, edges), _numeric_shares(current, edges))


def _categorical_shares(values: pd.Series, categories: list[str]) -> np.ndarray:
    v = values.dropna().astype(str)
    mapped = v.where(v.isin(categories), UNSEEN)
    counts = mapped.value_counts()
    return np.array([counts.get(c, 0) for c in categories + [UNSEEN]], dtype=float) / max(len(v), 1)


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    categories = sorted(reference.dropna().astype(str).unique().tolist())
    if not categories:
        raise ValueError("reference has no categories")
    if current.dropna().empty:
        raise ValueError("current batch has no categorical values")
    return _psi_from_shares(_categorical_shares(reference, categories), _categorical_shares(current, categories))


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    psi: float
    status: str
    unseen_share: float  # categorical only; 0.0 for numeric features
    out_of_range_share: float  # numeric only: share outside the reference min..max


def feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric: tuple[str, ...] = NUMERIC_FEATURES,
    categorical: tuple[str, ...] = CATEGORICAL_FEATURES,
    bins: int = 10,
) -> list[FeatureDrift]:
    """PSI and supporting shares for every feature, in a fixed order."""
    report: list[FeatureDrift] = []
    for col in numeric:
        psi = psi_numeric(reference[col], current[col], bins)
        lo, hi = reference[col].min(), reference[col].max()
        cur = current[col].dropna()
        out = float(((cur < lo) | (cur > hi)).mean()) if len(cur) else 0.0
        report.append(FeatureDrift(col, psi, status_for(psi), 0.0, out))
    for col in categorical:
        psi = psi_categorical(reference[col], current[col])
        known = set(reference[col].dropna().astype(str))
        cur = current[col].dropna().astype(str)
        unseen = float((~cur.isin(known)).mean()) if len(cur) else 0.0
        report.append(FeatureDrift(col, psi, status_for(psi), unseen, 0.0))
    return report


def prediction_drift(reference_pred: np.ndarray, current_pred: np.ndarray, bins: int = 10) -> FeatureDrift:
    """PSI on the model's own output: a cheap early signal that needs no labels."""
    psi = psi_numeric(pd.Series(reference_pred), pd.Series(current_pred), bins)
    lo, hi = float(np.min(reference_pred)), float(np.max(reference_pred))
    out = float(((current_pred < lo) | (current_pred > hi)).mean()) if len(current_pred) else 0.0
    return FeatureDrift("predicted_hours", psi, status_for(psi), 0.0, out)
