"""Command line interface.

    rate-card-history generate [--out data]
    rate-card-history load data/cdc_batches --db rates.duckdb
    rate-card-history rate-at L001-CCU-BLR 2026-07-15T10:00:00 --db rates.duckdb
    rate-card-history history L001-CCU-BLR --db rates.duckdb
    rate-card-history rerate data/invoices.csv --db rates.duckdb [--out report.csv]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import duckdb

from .events import read_jsonl
from .generator import generate_stream, write_stream
from .invoices import generate_invoices, write_invoices
from .loader import RateCardLoader
from .lookup import lane_history, rate_at
from .rerate import read_invoices, rerate, write_report


def _connect(db: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db)


def cmd_generate(args: argparse.Namespace) -> int:
    out = Path(args.out)
    stream = generate_stream(seed=args.seed)
    batches = write_stream(stream, out / "cdc_batches")
    n = write_invoices(generate_invoices(stream.history, seed=args.seed + 14), out / "invoices.csv")
    print(f"wrote {len(batches)} CDC batches ({len(stream.history)} changes) to {out / 'cdc_batches'}")
    print(f"wrote {n} invoices to {out / 'invoices.csv'}")
    return 0


def cmd_load(args: argparse.Namespace) -> int:
    paths = sorted(Path(args.batch_dir).glob("*.jsonl"))
    if not paths:
        print(f"no .jsonl batches found in {args.batch_dir}", file=sys.stderr)
        return 2
    con = _connect(args.db)
    loader = RateCardLoader(con)
    header = f"{'batch':<14}{'recv':>6}{'rej':>5}{'dup':>5}{'seen':>6}{'no-op':>7}{'open':>6}{'close':>7}{'wmark':>7}"
    print(header)
    for path in paths:
        r = loader.load_batch(list(read_jsonl(path)), path.stem)
        print(
            f"{r.batch_id:<14}{r.received:>6}{r.rejected:>5}{r.duplicates:>5}{r.below_watermark:>6}"
            f"{r.no_op:>7}{r.versions_opened:>6}{r.versions_closed:>7}{r.watermark_after:>7}"
        )
    versions, current = con.execute(
        "SELECT count(*), count(*) FILTER (WHERE is_current) FROM dim_rate_card"
    ).fetchone()
    quarantined = con.execute("SELECT count(*) FROM quarantine").fetchone()[0]
    print(f"dim_rate_card: {versions} versions, {current} current lanes; quarantine: {quarantined} rows")
    con.close()
    return 0


def _parse_when(value: str) -> datetime:
    try:
        when = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not an ISO-8601 timestamp: {value!r}") from exc
    if when.tzinfo is not None:
        raise argparse.ArgumentTypeError("use naive UTC timestamps (no offset)")
    return when


def cmd_rate_at(args: argparse.Namespace) -> int:
    con = _connect(args.db)
    version = rate_at(con, args.lane_id, args.at)
    con.close()
    if version is None:
        print(f"{args.lane_id}: no contracted rate at {args.at.isoformat()}")
        return 1
    print(
        f"{version.lane_id} at {args.at.isoformat()}: {version.rate_per_kg:.2f} {version.currency}/kg, "
        f"min {version.min_charge:.2f}, carrier {version.carrier} "
        f"(valid from {version.valid_from.isoformat()}, lsn {version.source_lsn})"
    )
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    con = _connect(args.db)
    versions = lane_history(con, args.lane_id)
    con.close()
    if not versions:
        print(f"{args.lane_id}: no history")
        return 1
    for v in versions:
        valid_to = "open" if v.valid_to.year == 9999 else v.valid_to.isoformat()
        print(f"{v.valid_from.isoformat()}  ->  {valid_to:<19}  {v.rate_per_kg:>7.2f}/kg  min {v.min_charge:>6.2f}")
    return 0


def cmd_rerate(args: argparse.Namespace) -> int:
    invoices, invalid = read_invoices(Path(args.invoices))
    con = _connect(args.db)
    report = rerate(con, invoices, tolerance=args.tolerance, invalid=invalid)
    con.close()
    s = report.summary()
    print(f"invoices: {len(invoices)} valid, {s['invalid']} invalid")
    for status in ("ok", "overbilled", "underbilled", "no_rate", "currency_mismatch"):
        print(f"  {status:<18}{s[status]:>5}")
    print(f"overbilled total:  {s['overbilled_total']:.2f}")
    print(f"underbilled total: {s['underbilled_total']:.2f}")
    print(f"billed with no contracted rate: {s['no_rate_billed_total']:.2f}")
    for line_no, reason in report.invalid[:10]:
        print(f"  invalid line {line_no}: {reason}")
    if args.out:
        write_report(report, Path(args.out))
        print(f"wrote lines needing attention to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rate-card-history", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="write synthetic CDC batches and invoices")
    p.add_argument("--out", default="data")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("load", help="load CDC batch files into the dimension, in file name order")
    p.add_argument("batch_dir")
    p.add_argument("--db", required=True)
    p.set_defaults(func=cmd_load)

    p = sub.add_parser("rate-at", help="show the rate valid for a lane at a moment")
    p.add_argument("lane_id")
    p.add_argument("at", type=_parse_when)
    p.add_argument("--db", required=True)
    p.set_defaults(func=cmd_rate_at)

    p = sub.add_parser("history", help="show every version of a lane")
    p.add_argument("lane_id")
    p.add_argument("--db", required=True)
    p.set_defaults(func=cmd_history)

    p = sub.add_parser("rerate", help="re-rate an invoice CSV against the rate valid on each ship date")
    p.add_argument("invoices")
    p.add_argument("--db", required=True)
    p.add_argument("--tolerance", type=float, default=0.01)
    p.add_argument("--out")
    p.set_defaults(func=cmd_rerate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
