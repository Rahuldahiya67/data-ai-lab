"""Error monitoring once actual delivery times arrive.

Drift checks only say the inputs look different. Whether the model is
actually worse is only known once deliveries complete, usually days after the
prediction. This module scores the rows that have an actual, overall and per
segment (for example per region), and compares each against the same
segment's error in the reference period.

A regional problem is easy to miss overall: if one region in five gets 35 %
slower, the overall error moves much less than that region's error. That is
why every segment is compared with its own baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import LABEL

WARN_RATIO = 1.25  # current MAE / reference MAE
ALERT_RATIO = 1.5
MIN_ROWS = 30  # fewer labelled rows than this: not enough evidence


@dataclass(frozen=True)
class ErrorStats:
    rows: int
    mae: float
    bias: float  # mean(predicted - actual); negative means deliveries take longer than promised
    p90_abs_error: float


@dataclass(frozen=True)
class SegmentPerformance:
    segment: str  # "overall" or "<column>=<value>"
    current: ErrorStats
    reference: ErrorStats | None
    mae_ratio: float | None
    status: str  # "ok" | "warn" | "alert" | "insufficient"


def error_stats(predicted: np.ndarray, actual: np.ndarray) -> ErrorStats:
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if predicted.shape != actual.shape:
        raise ValueError(f"predicted and actual have different shapes: {predicted.shape} vs {actual.shape}")
    keep = ~np.isnan(actual)
    err = predicted[keep] - actual[keep]
    if err.size == 0:
        return ErrorStats(0, float("nan"), float("nan"), float("nan"))
    return ErrorStats(
        rows=int(err.size),
        mae=float(np.mean(np.abs(err))),
        bias=float(np.mean(err)),
        p90_abs_error=float(np.quantile(np.abs(err), 0.9)),
    )


def _status(current: ErrorStats, reference: ErrorStats | None, min_rows: int) -> tuple[float | None, str]:
    if reference is None or reference.rows < min_rows or current.rows < min_rows:
        return None, "insufficient"
    ratio = current.mae / reference.mae if reference.mae > 0 else float("inf")
    if ratio >= ALERT_RATIO:
        return ratio, "alert"
    if ratio >= WARN_RATIO:
        return ratio, "warn"
    return ratio, "ok"


def segment_performance(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    by: str = "region",
    prediction_col: str = "predicted_hours",
    min_rows: int = MIN_ROWS,
) -> list[SegmentPerformance]:
    """Overall row first, then one row per segment value seen in either period."""
    for name, df in (("reference", reference), ("current", current)):
        missing = [c for c in (prediction_col, LABEL, by) if c not in df.columns]
        if missing:
            raise ValueError(f"{name} frame is missing columns: {', '.join(missing)}")

    def stats(df: pd.DataFrame) -> ErrorStats:
        return error_stats(df[prediction_col].to_numpy(), df[LABEL].to_numpy())

    out: list[SegmentPerformance] = []
    ref_all, cur_all = stats(reference), stats(current)
    ratio, status = _status(cur_all, ref_all, min_rows)
    out.append(SegmentPerformance("overall", cur_all, ref_all, ratio, status))

    values = sorted(set(reference[by].dropna().astype(str)) | set(current[by].dropna().astype(str)))
    for value in values:
        ref_seg = reference[reference[by].astype(str) == value]
        cur_seg = current[current[by].astype(str) == value]
        ref_stats = stats(ref_seg) if len(ref_seg) else None
        cur_stats = stats(cur_seg)
        ratio, status = _status(cur_stats, ref_stats, min_rows)
        out.append(SegmentPerformance(f"{by}={value}", cur_stats, ref_stats, ratio, status))
    return out
