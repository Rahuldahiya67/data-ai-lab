"""Load and validate the metrics spec (YAML).

A spec has five parts::

    fact_table: shipments
    tables:      # primary keys, the fact table's time column, many-to-one joins
    dimensions:  # name -> table.column that reports can group or filter by
    measures:    # name -> aggregation over the fact table, optional row filter
    metrics:     # name -> simple (one measure) or ratio (sum / sum)

Validation is done in two steps. ``load_spec`` checks the structure and the
references between parts (no data needed). ``check_columns`` checks that every
column the spec mentions exists in the actual tables. Both collect every problem
they find and raise one ``SpecError`` listing them, so a broken spec is fixed in
one pass instead of one error at a time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
AGGREGATIONS = {"count", "sum", "count_distinct", "min", "max"}
METRIC_TYPES = {"simple", "ratio"}
FORMATS = {"number", "percent", "currency"}


class SpecError(ValueError):
    """Raised when a spec is invalid. ``problems`` lists every issue found."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("invalid metrics spec:\n  - " + "\n  - ".join(self.problems))


@dataclass(frozen=True)
class Join:
    target: str
    key: str
    relationship: str


@dataclass(frozen=True)
class Table:
    name: str
    primary_key: str
    time_column: str | None = None
    joins: dict[str, Join] = field(default_factory=dict)


@dataclass(frozen=True)
class Dimension:
    name: str
    table: str
    column: str
    description: str = ""


@dataclass(frozen=True)
class Measure:
    name: str
    agg: str
    expr: str | None = None
    filter: str | None = None
    description: str = ""


@dataclass(frozen=True)
class Metric:
    name: str
    type: str
    measure: str | None = None
    numerator: str | None = None
    denominator: str | None = None
    format: str = "number"
    description: str = ""

    @property
    def measures(self) -> list[str]:
        if self.type == "simple":
            return [self.measure]  # type: ignore[list-item]
        return [self.numerator, self.denominator]  # type: ignore[list-item]


@dataclass(frozen=True)
class Spec:
    fact_table: str
    tables: dict[str, Table]
    dimensions: dict[str, Dimension]
    measures: dict[str, Measure]
    metrics: dict[str, Metric]

    @property
    def fact(self) -> Table:
        return self.tables[self.fact_table]


def _as_mapping(value: Any, where: str, problems: list[str]) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        problems.append(f"{where} must be a mapping, got {type(value).__name__}")
        return {}
    return value


def _check_keys(raw: dict, allowed: set[str], where: str, problems: list[str]) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        problems.append(f"{where}: unknown keys {unknown} (allowed: {sorted(allowed)})")


def _check_name(name: Any, where: str, problems: list[str]) -> bool:
    if not isinstance(name, str) or not NAME_RE.match(name):
        problems.append(
            f"{where}: name {name!r} must be lower_snake_case (letters, digits, underscores)"
        )
        return False
    return True


def _opt_str(raw: dict, key: str, where: str, problems: list[str]) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        problems.append(f"{where}.{key} must be a non-empty string")
        return None
    return value.strip()


def parse_spec(raw: Any) -> Spec:
    """Build a ``Spec`` from an already-parsed YAML document and validate references."""
    problems: list[str] = []
    raw = _as_mapping(raw, "spec", problems)
    _check_keys(raw, {"fact_table", "tables", "dimensions", "measures", "metrics"}, "spec", problems)

    fact_table = raw.get("fact_table")
    if not isinstance(fact_table, str):
        problems.append("spec.fact_table is required and must be a table name")
        fact_table = ""

    tables: dict[str, Table] = {}
    for name, body in _as_mapping(raw.get("tables"), "tables", problems).items():
        where = f"tables.{name}"
        if not _check_name(name, where, problems):
            continue
        body = _as_mapping(body, where, problems)
        _check_keys(body, {"primary_key", "time_column", "joins"}, where, problems)
        pk = _opt_str(body, "primary_key", where, problems)
        if pk is None:
            problems.append(f"{where}.primary_key is required")
            continue
        joins = {}
        for target, jbody in _as_mapping(body.get("joins"), f"{where}.joins", problems).items():
            jwhere = f"{where}.joins.{target}"
            jbody = _as_mapping(jbody, jwhere, problems)
            if True in jbody:
                problems.append(
                    f"{jwhere}: YAML reads a bare 'on' key as the boolean true; "
                    "name the join column with 'key' instead"
                )
                jbody = {k: v for k, v in jbody.items() if k is not True}
            _check_keys(jbody, {"key", "relationship"}, jwhere, problems)
            key = _opt_str(jbody, "key", jwhere, problems)
            rel = jbody.get("relationship")
            if key is None:
                problems.append(f"{jwhere}.key is required")
                continue
            if rel != "many_to_one":
                problems.append(
                    f"{jwhere}: relationship must be 'many_to_one', got {rel!r}; "
                    "other relationships would duplicate fact rows and inflate sums"
                )
                continue
            joins[target] = Join(target=target, key=key, relationship=rel)
        tables[name] = Table(
            name=name,
            primary_key=pk,
            time_column=_opt_str(body, "time_column", where, problems),
            joins=joins,
        )

    if fact_table and fact_table not in tables:
        problems.append(f"spec.fact_table {fact_table!r} is not defined under tables")
    fact = tables.get(fact_table)
    if fact is not None:
        if fact.time_column is None:
            problems.append(f"tables.{fact_table}.time_column is required on the fact table")
        for target in fact.joins:
            if target not in tables:
                problems.append(f"tables.{fact_table}.joins: unknown table {target!r}")
            elif target == fact_table:
                problems.append(f"tables.{fact_table}.joins: the fact table cannot join itself")
    for name, table in tables.items():
        if name != fact_table and table.joins:
            problems.append(
                f"tables.{name}.joins: only the fact table may declare joins "
                "(one hop from the fact table)"
            )
    reachable = {fact_table, *(fact.joins if fact else {})}

    dimensions: dict[str, Dimension] = {}
    for name, body in _as_mapping(raw.get("dimensions"), "dimensions", problems).items():
        where = f"dimensions.{name}"
        if not _check_name(name, where, problems):
            continue
        body = _as_mapping(body, where, problems)
        _check_keys(body, {"table", "column", "description"}, where, problems)
        table = _opt_str(body, "table", where, problems)
        column = _opt_str(body, "column", where, problems)
        if table is None or column is None:
            problems.append(f"{where}: table and column are required")
            continue
        if table not in tables:
            problems.append(f"{where}: unknown table {table!r}")
        elif table not in reachable:
            problems.append(f"{where}: table {table!r} is not joined to the fact table")
        dimensions[name] = Dimension(name, table, column, body.get("description", "") or "")

    measures: dict[str, Measure] = {}
    for name, body in _as_mapping(raw.get("measures"), "measures", problems).items():
        where = f"measures.{name}"
        if not _check_name(name, where, problems):
            continue
        body = _as_mapping(body, where, problems)
        _check_keys(body, {"agg", "expr", "filter", "description"}, where, problems)
        agg = body.get("agg")
        if agg not in AGGREGATIONS:
            problems.append(f"{where}.agg must be one of {sorted(AGGREGATIONS)}, got {agg!r}")
            continue
        expr = _opt_str(body, "expr", where, problems)
        if agg != "count" and expr is None:
            problems.append(f"{where}: agg {agg!r} needs an expr")
            continue
        measures[name] = Measure(
            name=name,
            agg=agg,
            expr=expr,
            filter=_opt_str(body, "filter", where, problems),
            description=body.get("description", "") or "",
        )

    metrics: dict[str, Metric] = {}
    for name, body in _as_mapping(raw.get("metrics"), "metrics", problems).items():
        where = f"metrics.{name}"
        if not _check_name(name, where, problems):
            continue
        body = _as_mapping(body, where, problems)
        _check_keys(
            body,
            {"type", "measure", "numerator", "denominator", "format", "description"},
            where,
            problems,
        )
        mtype = body.get("type")
        if mtype not in METRIC_TYPES:
            problems.append(f"{where}.type must be one of {sorted(METRIC_TYPES)}, got {mtype!r}")
            continue
        fmt = body.get("format", "number")
        if fmt not in FORMATS:
            problems.append(f"{where}.format must be one of {sorted(FORMATS)}, got {fmt!r}")
        needed = ["measure"] if mtype == "simple" else ["numerator", "denominator"]
        refs = {}
        for key in needed:
            ref = body.get(key)
            if ref is None:
                problems.append(f"{where}.{key} is required for a {mtype} metric")
            elif ref not in measures:
                problems.append(f"{where}.{key}: unknown measure {ref!r}")
            refs[key] = ref
        metrics[name] = Metric(
            name=name,
            type=mtype,
            format=fmt if fmt in FORMATS else "number",
            description=body.get("description", "") or "",
            **refs,
        )

    clash = sorted(set(dimensions) & set(metrics))
    if clash:
        problems.append(f"names used for both a dimension and a metric: {clash}")
    if not metrics:
        problems.append("spec defines no metrics")

    if problems:
        raise SpecError(problems)
    return Spec(fact_table, tables, dimensions, measures, metrics)


def load_spec(path: str | Path) -> Spec:
    """Read a YAML spec file and validate it."""
    with Path(path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return parse_spec(raw)


def check_columns(spec: Spec, schema: dict[str, set[str]]) -> None:
    """Check every column the spec references against ``schema`` (table -> column names).

    Raises ``SpecError`` listing all missing tables, missing columns and unsafe
    measure expressions or filters.
    """
    from .expressions import check_fragment

    problems: list[str] = []

    def need(table: str, column: str, where: str) -> None:
        if table not in schema:
            problems.append(f"{where}: table {table!r} was not loaded")
        elif column not in schema[table]:
            problems.append(f"{where}: column {table}.{column} does not exist")

    for name, table in spec.tables.items():
        need(name, table.primary_key, f"tables.{name}.primary_key")
        if table.time_column:
            need(name, table.time_column, f"tables.{name}.time_column")
        for target, join in table.joins.items():
            need(name, join.key, f"tables.{name}.joins.{target}")
            need(target, join.key, f"tables.{name}.joins.{target}")
            if join.key != spec.tables[target].primary_key:
                problems.append(
                    f"tables.{name}.joins.{target}: joins must use the target's primary key "
                    f"{spec.tables[target].primary_key!r}, got {join.key!r}"
                )
    for name, dim in spec.dimensions.items():
        need(dim.table, dim.column, f"dimensions.{name}")

    fact_columns = schema.get(spec.fact_table, set())
    for name, measure in spec.measures.items():
        for key in ("expr", "filter"):
            fragment = getattr(measure, key)
            if fragment is None:
                continue
            for problem in check_fragment(fragment, fact_columns):
                problems.append(f"measures.{name}.{key}: {problem}")

    if problems:
        raise SpecError(problems)
