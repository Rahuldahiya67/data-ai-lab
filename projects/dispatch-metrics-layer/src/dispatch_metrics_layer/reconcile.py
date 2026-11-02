"""Reconcile competing definitions of the same KPI on the same data.

Two reports that both say "on-time rate" can disagree for two separate reasons:

1. **Different definitions.** One divides by delivered shipments, another by
   every dispatched shipment (so in-transit shipments count as not on time),
   a third weighs shipments by units. Each is a metric in the spec; the report
   puts them side by side, overall and per group, with the gap to a baseline.
2. **Different aggregation.** Averaging per-group rates (say, one rate per
   carrier, then the mean of those) gives a small carrier the same weight as a
   large one. The layer always computes ``SUM(numerator) / SUM(denominator)``
   at the requested grain; the report shows how far the unweighted average of
   group rates is from that pooled value.

The report never computes a metric itself: every number comes from
``MetricsEngine.query``, so it uses exactly the SQL the layer serves.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from statistics import fmean
from typing import Any

from .compiler import MetricRequest, RequestError
from .engine import MetricsEngine


@dataclass(frozen=True)
class GroupRow:
    group: Any
    values: dict[str, float | None]


@dataclass(frozen=True)
class AggregationCheck:
    metric: str
    pooled: float | None
    average_of_groups: float | None
    groups_used: int

    @property
    def gap(self) -> float | None:
        if self.pooled is None or self.average_of_groups is None:
            return None
        return self.average_of_groups - self.pooled


@dataclass(frozen=True)
class ReconciliationReport:
    metrics: tuple[str, ...]
    baseline: str
    overall: dict[str, float | None]
    by: str | None = None
    groups: list[GroupRow] = field(default_factory=list)
    aggregation: list[AggregationCheck] = field(default_factory=list)

    def gap_to_baseline(self, metric: str, values: dict[str, float | None] | None = None) -> float | None:
        values = self.overall if values is None else values
        a, b = values.get(metric), values.get(self.baseline)
        if a is None or b is None:
            return None
        return a - b

    @property
    def spread(self) -> float | None:
        """Largest minus smallest overall value across the definitions."""
        present = [v for v in self.overall.values() if v is not None]
        return max(present) - min(present) if len(present) > 1 else None


def reconcile(
    engine: MetricsEngine,
    metrics: list[str] | tuple[str, ...],
    by: str | None = None,
    base: MetricRequest | None = None,
) -> ReconciliationReport:
    """Compare ``metrics`` (first one is the baseline), overall and per ``by`` group.

    ``base`` carries filters and the date range; its metrics, dimensions and
    grain are replaced. Only ratio and derived metrics get an aggregation
    check, since a simple count or sum has no rate to average.
    """
    metrics = tuple(metrics)
    if len(metrics) < 2:
        raise RequestError("reconcile needs at least two metrics: a baseline and one to compare")
    base = base or MetricRequest(metrics)
    request = replace(base, metrics=metrics, dimensions=(), grain=None)
    (overall,) = engine.query(request).records() or [dict.fromkeys(metrics)]
    overall = {m: overall[m] for m in metrics}

    groups: list[GroupRow] = []
    checks: list[AggregationCheck] = []
    if by is not None:
        rows = engine.query(replace(request, dimensions=(by,))).records()
        groups = [GroupRow(r[by], {m: r[m] for m in metrics}) for r in rows]
        for m in metrics:
            if engine.spec.metrics[m].type == "simple":
                continue
            present = [g.values[m] for g in groups if g.values[m] is not None]
            checks.append(
                AggregationCheck(
                    metric=m,
                    pooled=overall[m],
                    average_of_groups=fmean(present) if present else None,
                    groups_used=len(present),
                )
            )
    return ReconciliationReport(
        metrics=metrics, baseline=metrics[0], overall=overall, by=by, groups=groups, aggregation=checks
    )
