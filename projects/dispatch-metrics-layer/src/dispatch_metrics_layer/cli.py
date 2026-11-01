"""Command line interface: ``dispatch-metrics``.

Subcommands::

    dispatch-metrics list                       # metrics and dimensions in the spec
    dispatch-metrics check                      # validate spec against the data
    dispatch-metrics query -m on_time_rate,otif_rate --by carrier --grain month \\
        --where "region in North,South" --start 2026-08-01 --format table
    dispatch-metrics reconcile -m on_time_rate,on_time_rate_dispatched,on_time_rate_units \\
        --by carrier                            # compare competing definitions
    dispatch-metrics generate --out data        # write the synthetic CSVs

``--spec`` and ``--data`` default to ``specs/dispatch.yaml`` and ``data/``
relative to the current directory. Exit codes: 0 success, 1 invalid spec or
data, 2 invalid request or arguments.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from .compiler import OPERATORS, Filter, MetricRequest, RequestError
from .engine import DuplicateKeyError, FanOutError, MetricsEngine, QueryResult
from .expressions import quote_ident
from .generator import GeneratorConfig, generate, write_csvs
from .reconcile import ReconciliationReport, reconcile
from .spec import Spec, SpecError, load_spec

DEFAULT_SPEC = Path("specs/dispatch.yaml")
DEFAULT_DATA = Path("data")

# "carrier = Metro Parcel", "status!=cancelled", "region in North,South"
_LIST_RE = re.compile(r"^\s*([a-z][a-z0-9_]*)\s+(in|not_in)\s+(.+?)\s*$")
_CMP_RE = re.compile(r"^\s*([a-z][a-z0-9_]*)\s*(!=|<=|>=|=|<|>)\s*(.*?)\s*$")


class UsageError(ValueError):
    """Bad command-line input (exit code 2)."""


def parse_where(text: str) -> Filter:
    """Parse one ``--where`` expression into a ``Filter``.

    List operators split their values on commas: ``region in North,South``.
    Values are passed to DuckDB as strings; DuckDB casts them when the column
    is numeric or a date.
    """
    m = _LIST_RE.match(text)
    if m:
        dim, op, values = m.groups()
        items = [v.strip() for v in values.split(",") if v.strip()]
        if not items:
            raise UsageError(f"--where {text!r}: {op} needs at least one value")
        return Filter(dim, op, tuple(items))
    m = _CMP_RE.match(text)
    if m:
        dim, op, value = m.groups()
        if value == "":
            raise UsageError(f"--where {text!r}: missing value after {op!r}")
        return Filter(dim, op, value)
    ops = ", ".join(sorted(OPERATORS))
    raise UsageError(
        f"--where {text!r}: expected 'dimension OP value' (OP one of {ops}; "
        "in and not_in take comma-separated values)"
    )


def _parse_date(text: str, flag: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise UsageError(f"{flag} {text!r}: expected a date as YYYY-MM-DD") from None


def _split(values: Sequence[str] | None) -> list[str]:
    out: list[str] = []
    for value in values or []:
        out.extend(v.strip() for v in value.split(",") if v.strip())
    return out


def format_value(value: Any, fmt: str) -> str:
    """Human-readable cell for the table output. Raw values go to csv and json."""
    if value is None:
        return "-"
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return value
    if fmt == "percent":
        return f"{value * 100:.1f}%"
    if fmt == "currency":
        return f"{value:,.2f}"
    if isinstance(value, float) and not value.is_integer():
        return f"{value:,.3f}"
    return f"{int(value):,}"


def render_table(result: QueryResult, spec: Spec) -> str:
    formats = {name: spec.metrics[name].format for name in result.query.columns if name in spec.metrics}
    header = list(result.columns)
    body = [
        [format_value(v, formats.get(col, "number")) for col, v in zip(header, row)]
        for row in result.rows
    ]
    out = _columns(header, body, [col in formats for col in header])
    if not body:
        out.append("(no rows)")
    return "\n".join(out)


def format_gap(value: float | None, fmt: str) -> str:
    """A signed difference: percentage points for percent metrics."""
    if value is None:
        return "-"
    if fmt == "percent":
        points = round(value * 100, 1) + 0.0  # + 0.0 turns -0.0 into 0.0
        return f"{points:+.1f} pts"
    if fmt == "currency":
        return f"{value:+,.2f}"
    return f"{value:+,.3f}"


def _columns(header: list[str], body: list[list[str]], numeric: list[bool]) -> list[str]:
    widths = [max(len(h), *(len(r[i]) for r in body)) if body else len(h) for i, h in enumerate(header)]

    def line(cells: list[str]) -> str:
        return "  ".join(
            c.rjust(w) if num else c.ljust(w) for c, w, num in zip(cells, widths, numeric)
        ).rstrip()

    return [line(header), line(["-" * w for w in widths])] + [line(r) for r in body]


def render_reconciliation(report: ReconciliationReport, spec: Spec) -> str:
    fmt = {m: spec.metrics[m].format for m in report.metrics}
    base = report.baseline
    out = ["Overall"]
    body = [
        [m, format_value(report.overall[m], fmt[m]), "baseline" if m == base else format_gap(report.gap_to_baseline(m), fmt[m])]
        for m in report.metrics
    ]
    out += _columns(["definition", "value", f"vs {base}"], body, [False, True, True])
    if spec.metrics[base].description:
        out.append(f"baseline: {spec.metrics[base].description}")
    if report.spread is not None:
        out.append(f"spread across definitions: {format_gap(report.spread, fmt[base]).lstrip('+')}")
    if report.by is not None:
        out += ["", f"By {report.by}"]
        body = [
            [format_value(g.group, "number")] + [format_value(g.values[m], fmt[m]) for m in report.metrics]
            for g in report.groups
        ]
        out += _columns([report.by, *report.metrics], body, [False] + [True] * len(report.metrics))
        if not body:
            out.append("(no rows)")
        if report.aggregation:
            out += ["", f"Pooled rate vs unweighted average of {report.by} rates"]
            body = [
                [
                    c.metric,
                    format_value(c.pooled, fmt[c.metric]),
                    format_value(c.average_of_groups, fmt[c.metric]),
                    format_gap(c.gap, fmt[c.metric]),
                ]
                for c in report.aggregation
            ]
            out += _columns(["metric", "pooled", "avg of groups", "gap"], body, [False, True, True, True])
    return "\n".join(out)


def _jsonable(value: Any) -> Any:
    return value.isoformat() if isinstance(value, date) else value


def render(result: QueryResult, spec: Spec, fmt: str) -> str:
    if fmt == "json":
        return json.dumps([{k: _jsonable(v) for k, v in r.items()} for r in result.records()], indent=2)
    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(result.columns)
        writer.writerows([_jsonable(v) for v in row] for row in result.rows)
        return buf.getvalue().rstrip("\n")
    return render_table(result, spec)


def build_request(args: argparse.Namespace) -> MetricRequest:
    metrics = _split(args.metrics)
    if not metrics:
        raise UsageError("--metrics needs at least one metric name")
    return MetricRequest(
        metrics=metrics,
        dimensions=_split(args.by),
        filters=[parse_where(w) for w in args.where or []],
        grain=args.grain,
        start=_parse_date(args.start, "--start") if args.start else None,
        end=_parse_date(args.end, "--end") if args.end else None,
    )


def cmd_list(args: argparse.Namespace, out) -> int:
    spec = load_spec(args.spec)
    print("Metrics", file=out)
    for name, metric in sorted(spec.metrics.items()):
        if metric.type == "derived":
            how = f"= {metric.expr}"
        elif metric.type == "ratio":
            how = f"= {metric.numerator} / {metric.denominator}"
        else:
            how = f"= {metric.measure}"
        print(f"  {name:<26} {metric.type:<8} {metric.format:<8} {how}", file=out)
        if metric.description:
            print(f"  {'':<26} {metric.description}", file=out)
    print("Dimensions", file=out)
    for name, dim in sorted(spec.dimensions.items()):
        print(f"  {name:<26} {dim.table}.{dim.column}", file=out)
    return 0


def cmd_check(args: argparse.Namespace, out) -> int:
    spec = load_spec(args.spec)
    engine = MetricsEngine.from_csv_dir(spec, args.data)
    for table in spec.tables:
        (n,) = engine.conn.execute(f"SELECT COUNT(*) FROM {quote_ident(table)}").fetchone()
        print(f"{table}: {n:,} rows", file=out)
    print(
        f"spec OK: {len(spec.metrics)} metrics, {len(spec.dimensions)} dimensions, "
        f"{len(spec.measures)} measures",
        file=out,
    )
    for table, n in engine.orphans.items():
        if n:
            print(
                f"warning: {n:,} {spec.fact_table} rows have no match in {table}; "
                "they count in totals and group under an empty value",
                file=out,
            )
    return 0


def cmd_query(args: argparse.Namespace, out) -> int:
    spec = load_spec(args.spec)
    request = build_request(args)
    engine = MetricsEngine.from_csv_dir(spec, args.data)
    if args.sql:
        compiled = engine.compile(request)
        print(compiled.sql, file=out)
        print(f"-- params: {[_jsonable(p) for p in compiled.params]}", file=out)
        return 0
    print(render(engine.query(request), spec, args.format), file=out)
    return 0


def cmd_reconcile(args: argparse.Namespace, out) -> int:
    spec = load_spec(args.spec)
    by = _split(args.by)
    if len(by) > 1:
        raise UsageError("--by takes one dimension for reconcile")
    # Filters and dates as for query; reconcile sets the grouping itself.
    base = build_request(argparse.Namespace(**{**vars(args), "by": None, "grain": None}))
    engine = MetricsEngine.from_csv_dir(spec, args.data)
    report = reconcile(engine, base.metrics, by=by[0] if by else None, base=base)
    print(render_reconciliation(report, spec), file=out)
    return 0


def cmd_generate(args: argparse.Namespace, out) -> int:
    try:
        config = GeneratorConfig(n_shipments=args.shipments, seed=args.seed)
    except ValueError as exc:
        raise UsageError(str(exc)) from None
    tables = generate(config)
    for name, path in write_csvs(tables, args.out).items():
        print(f"wrote {path} ({len(tables[name]):,} rows)", file=out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dispatch-metrics",
        description="Dispatch KPIs defined once in YAML, compiled to DuckDB SQL.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, data: bool = True) -> None:
        p.add_argument("--spec", type=Path, default=DEFAULT_SPEC, help="metrics spec (YAML)")
        if data:
            p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="folder of <table>.csv files")

    p = sub.add_parser("list", help="list metrics and dimensions")
    common(p, data=False)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("check", help="validate the spec against the data")
    common(p)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("query", help="compute metrics")
    common(p)
    p.add_argument("-m", "--metrics", action="append", required=True, help="comma-separated metric names")
    p.add_argument("--by", action="append", help="comma-separated dimensions to group by")
    p.add_argument("--where", action="append", help="filter, e.g. 'region in North,South' (repeatable)")
    p.add_argument("--grain", choices=["day", "week", "month"], help="time grain on the fact time column")
    p.add_argument("--start", help="first date included (YYYY-MM-DD)")
    p.add_argument("--end", help="first date excluded (YYYY-MM-DD)")
    p.add_argument("--format", choices=["table", "csv", "json"], default="table")
    p.add_argument("--sql", action="store_true", help="print the compiled SQL instead of running it")
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("reconcile", help="compare competing definitions of a KPI")
    common(p)
    p.add_argument("-m", "--metrics", action="append", required=True,
                   help="comma-separated metrics; the first is the baseline")
    p.add_argument("--by", action="append", help="one dimension for the per-group view")
    p.add_argument("--where", action="append", help="filter, as for query (repeatable)")
    p.add_argument("--start", help="first date included (YYYY-MM-DD)")
    p.add_argument("--end", help="first date excluded (YYYY-MM-DD)")
    p.set_defaults(func=cmd_reconcile)

    p = sub.add_parser("generate", help="write the synthetic dispatch CSVs")
    p.add_argument("--out", type=Path, default=DEFAULT_DATA)
    p.add_argument("--shipments", type=int, default=GeneratorConfig().n_shipments)
    p.add_argument("--seed", type=int, default=GeneratorConfig().seed)
    p.set_defaults(func=cmd_generate)
    return parser


def main(argv: Sequence[str] | None = None, out=None, err=None) -> int:
    out = out or sys.stdout
    err = err or sys.stderr
    args = build_parser().parse_args(argv)
    try:
        return args.func(args, out)
    except (SpecError, FanOutError, DuplicateKeyError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=err)
        return 1
    except (RequestError, UsageError) as exc:
        print(f"error: {exc}", file=err)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
