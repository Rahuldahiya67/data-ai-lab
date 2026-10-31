"""Run metric requests against DuckDB.

``MetricsEngine`` owns a DuckDB connection with the spec's tables loaded. Before
it answers any request it checks the data, not just the spec:

* every column the spec references exists (``spec.check_columns``),
* every join key is unique in its dimension table. A duplicated ``lane_id`` in
  the lanes table would turn each matching shipment into two rows and silently
  double freight cost, so it is a hard error (``FanOutError``),
* fact rows whose key has no match in a dimension table are counted and exposed
  as ``orphans``; they still count in totals and show up with a NULL dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from .compiler import CompiledQuery, MetricRequest, compile_request
from .spec import Spec, check_columns


class FanOutError(ValueError):
    """A join key is not unique in its dimension table."""


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    query: CompiledQuery

    def records(self) -> list[dict[str, Any]]:
        return [dict(zip(self.columns, row)) for row in self.rows]


class MetricsEngine:
    def __init__(self, spec: Spec, conn: duckdb.DuckDBPyConnection):
        self.spec = spec
        self.conn = conn
        self.schema = self._read_schema()
        check_columns(spec, self.schema)
        self.orphans = self._check_joins()

    @classmethod
    def from_csv_dir(cls, spec: Spec, data_dir: str | Path) -> "MetricsEngine":
        """Load ``<data_dir>/<table>.csv`` for every table in the spec into an in-memory DuckDB."""
        data_dir = Path(data_dir)
        conn = duckdb.connect()
        for name in spec.tables:
            path = data_dir / f"{name}.csv"
            if not path.is_file():
                raise FileNotFoundError(f"missing data file for table {name!r}: {path}")
            conn.execute(
                f"CREATE TABLE {name} AS SELECT * FROM read_csv(?, header = true)",
                [str(path)],
            )
        return cls(spec, conn)

    def _read_schema(self) -> dict[str, set[str]]:
        rows = self.conn.execute(
            "SELECT table_name, column_name FROM information_schema.columns"
        ).fetchall()
        schema: dict[str, set[str]] = {}
        for table, column in rows:
            schema.setdefault(table, set()).add(column)
        return schema

    def _check_joins(self) -> dict[str, int]:
        fact = self.spec.fact
        orphans = {}
        for target, join in fact.joins.items():
            dupes = self.conn.execute(
                f"SELECT {join.key}, COUNT(*) AS n FROM {target} "
                f"GROUP BY 1 HAVING COUNT(*) > 1 ORDER BY 1 LIMIT 5"
            ).fetchall()
            if dupes:
                sample = ", ".join(f"{k!r} x{n}" for k, n in dupes)
                raise FanOutError(
                    f"join key {target}.{join.key} is not unique ({sample}); joining it "
                    f"would duplicate {fact.name} rows and inflate every sum"
                )
            (missing,) = self.conn.execute(
                f"SELECT COUNT(*) FROM {fact.name} AS f "
                f"WHERE f.{join.key} IS NOT NULL AND NOT EXISTS "
                f"(SELECT 1 FROM {target} AS t WHERE t.{join.key} = f.{join.key})"
            ).fetchone()
            orphans[target] = missing
        return orphans

    def compile(self, request: MetricRequest) -> CompiledQuery:
        return compile_request(self.spec, request, self.schema[self.spec.fact_table])

    def query(self, request: MetricRequest) -> QueryResult:
        compiled = self.compile(request)
        rows = self.conn.execute(compiled.sql, compiled.params).fetchall()
        return QueryResult(columns=compiled.columns, rows=rows, query=compiled)
