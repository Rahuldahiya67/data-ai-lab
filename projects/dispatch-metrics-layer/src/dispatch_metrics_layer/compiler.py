"""Compile a metric request into one parameterised DuckDB query.

The query has two levels:

1. ``agg``: one pass over the fact table (plus only the joins the request needs)
   that computes each required *measure* at the requested grain.
2. the outer SELECT turns measures into *metrics*. A ratio metric is always
   ``SUM(numerator) / SUM(denominator)`` at the requested grain, never an
   average of per-row or per-group ratios, so "on-time rate by region" and
   "on-time rate overall" are always consistent with each other.

Filter values are bound as query parameters, never pasted into the SQL text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .expressions import qualify
from .spec import Spec

GRAINS = {"day", "week", "month"}
OPERATORS = {"=", "!=", "<", "<=", ">", ">=", "in", "not_in"}
FACT_ALIAS = "f"
PERIOD_COLUMN = "period"


class RequestError(ValueError):
    """Raised when a metric request does not fit the spec."""


@dataclass(frozen=True)
class Filter:
    dimension: str
    op: str
    value: Any


@dataclass(frozen=True)
class MetricRequest:
    metrics: tuple[str, ...]
    dimensions: tuple[str, ...] = ()
    filters: tuple[Filter, ...] = ()
    grain: str | None = None
    start: date | None = None  # inclusive
    end: date | None = None  # exclusive

    def __post_init__(self) -> None:
        # Accept lists from callers but store tuples so requests stay hashable.
        object.__setattr__(self, "metrics", tuple(self.metrics))
        object.__setattr__(self, "dimensions", tuple(self.dimensions))
        object.__setattr__(self, "filters", tuple(self.filters))


@dataclass(frozen=True)
class CompiledQuery:
    sql: str
    params: list[Any] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)


def _join_alias(table: str) -> str:
    return f"j_{table}"


def _column_ref(spec: Spec, table: str, column: str) -> str:
    alias = FACT_ALIAS if table == spec.fact_table else _join_alias(table)
    return f"{alias}.{column}"


def validate_request(spec: Spec, request: MetricRequest) -> None:
    problems = []
    if not request.metrics:
        problems.append("request at least one metric")
    for name in request.metrics:
        if name not in spec.metrics:
            problems.append(f"unknown metric {name!r} (known: {sorted(spec.metrics)})")
    if len(set(request.metrics)) != len(request.metrics):
        problems.append("a metric is requested more than once")
    for name in request.dimensions:
        if name not in spec.dimensions:
            problems.append(f"unknown dimension {name!r} (known: {sorted(spec.dimensions)})")
    if len(set(request.dimensions)) != len(request.dimensions):
        problems.append("a dimension is requested more than once")
    if PERIOD_COLUMN in request.dimensions:
        problems.append(f"{PERIOD_COLUMN!r} is reserved for the time grain")
    if request.grain is not None and request.grain not in GRAINS:
        problems.append(f"grain must be one of {sorted(GRAINS)}, got {request.grain!r}")
    if request.start and request.end and request.start >= request.end:
        problems.append(f"start {request.start} must be before end {request.end}")
    for flt in request.filters:
        if flt.dimension not in spec.dimensions:
            problems.append(f"filter on unknown dimension {flt.dimension!r}")
        if flt.op not in OPERATORS:
            problems.append(f"filter operator must be one of {sorted(OPERATORS)}, got {flt.op!r}")
        elif flt.op in {"in", "not_in"}:
            if not isinstance(flt.value, (list, tuple, set, frozenset)) or not flt.value:
                problems.append(f"filter {flt.dimension} {flt.op} needs a non-empty list of values")
        elif isinstance(flt.value, (list, tuple, set, frozenset)):
            problems.append(f"filter {flt.dimension} {flt.op} needs a single value, not a list")
    if problems:
        raise RequestError("; ".join(problems))


def _measure_sql(spec: Spec, name: str, fact_columns: set[str]) -> str:
    measure = spec.measures[name]
    expr = qualify(measure.expr, fact_columns, FACT_ALIAS) if measure.expr else None
    if measure.agg == "count":
        base = f"COUNT({expr})" if expr else "COUNT(*)"
    elif measure.agg == "count_distinct":
        base = f"COUNT(DISTINCT {expr})"
    else:
        base = f"{measure.agg.upper()}({expr})"
    if measure.filter:
        base += f" FILTER (WHERE {qualify(measure.filter, fact_columns, FACT_ALIAS)})"
    if measure.agg == "sum":
        # A group whose rows are all filtered out has nothing to add up: 0, not NULL.
        base = f"COALESCE({base}, 0)"
    return base


def compile_request(spec: Spec, request: MetricRequest, fact_columns: set[str]) -> CompiledQuery:
    """Build the SQL for ``request``. ``fact_columns`` are the fact table's column names."""
    validate_request(spec, request)
    fact = spec.fact
    params: list[Any] = []

    needed_tables = {spec.dimensions[d].table for d in request.dimensions}
    needed_tables |= {spec.dimensions[f.dimension].table for f in request.filters}
    needed_tables.discard(spec.fact_table)

    group_cols: list[tuple[str, str]] = []  # (sql expression, output name)
    if request.grain:
        time_ref = f"{FACT_ALIAS}.{fact.time_column}"
        group_cols.append((f"CAST(date_trunc('{request.grain}', {time_ref}) AS DATE)", PERIOD_COLUMN))
    for dim_name in request.dimensions:
        dim = spec.dimensions[dim_name]
        group_cols.append((_column_ref(spec, dim.table, dim.column), dim_name))

    measure_names: list[str] = []
    for metric_name in request.metrics:
        for m in spec.metrics[metric_name].measures:
            if m not in measure_names:
                measure_names.append(m)

    select_parts = [f"{expr} AS {name}" for expr, name in group_cols]
    select_parts += [f"{_measure_sql(spec, m, fact_columns)} AS ms__{m}" for m in measure_names]

    from_sql = f"{fact.name} AS {FACT_ALIAS}"
    for table in sorted(needed_tables):
        join = fact.joins[table]
        from_sql += (
            f"\n  LEFT JOIN {table} AS {_join_alias(table)}"
            f" ON {FACT_ALIAS}.{join.key} = {_join_alias(table)}.{join.key}"
        )

    where = []
    if request.start:
        where.append(f"{FACT_ALIAS}.{fact.time_column} >= ?")
        params.append(request.start)
    if request.end:
        where.append(f"{FACT_ALIAS}.{fact.time_column} < ?")
        params.append(request.end)
    for flt in request.filters:
        dim = spec.dimensions[flt.dimension]
        ref = _column_ref(spec, dim.table, dim.column)
        if flt.op in {"in", "not_in"}:
            values = sorted(flt.value, key=str)
            marks = ", ".join("?" for _ in values)
            keyword = "IN" if flt.op == "in" else "NOT IN"
            where.append(f"{ref} {keyword} ({marks})")
            params.extend(values)
        else:
            where.append(f"{ref} {flt.op} ?")
            params.append(flt.value)

    inner = "SELECT\n    " + ",\n    ".join(select_parts) + f"\n  FROM {from_sql}"
    if where:
        inner += "\n  WHERE " + "\n    AND ".join(where)
    if group_cols:
        inner += "\n  GROUP BY " + ", ".join(str(i + 1) for i in range(len(group_cols)))

    outer_parts = [name for _, name in group_cols]
    for metric_name in request.metrics:
        metric = spec.metrics[metric_name]
        if metric.type == "simple":
            outer_parts.append(f"ms__{metric.measure} AS {metric_name}")
        else:
            outer_parts.append(
                f"CAST(ms__{metric.numerator} AS DOUBLE)"
                f" / NULLIF(ms__{metric.denominator}, 0) AS {metric_name}"
            )

    sql = f"WITH agg AS (\n  {inner}\n)\nSELECT " + ", ".join(outer_parts) + "\nFROM agg"
    if group_cols:
        sql += "\nORDER BY " + ", ".join(f"{name} NULLS LAST" for _, name in group_cols)
    columns = [name for _, name in group_cols] + list(request.metrics)
    return CompiledQuery(sql=sql, params=params, columns=columns)
