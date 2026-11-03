"""Command line entry point: ``stall-monitor generate | run | score``."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import timedelta
from pathlib import Path
from typing import Sequence

from .generator import GeneratorConfig, write_dataset
from .monitor import Alert, AlertKind, StallMonitor
from .scoring import format_table, sweep
from .stream import read_events

ALERT_FIELDS = ("kind", "shipment_id", "stage", "location", "last_event_time",
                "watermark", "raised_at_ingest", "dead_letters")


def _finite_hours(value: str) -> float:
    try:
        hours = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number of hours: {value}") from None
    if not math.isfinite(hours):
        raise argparse.ArgumentTypeError(f"hours must be a finite number: {value}")
    return hours


def _non_negative_hours(value: str) -> float:
    hours = _finite_hours(value)
    if hours < 0:
        raise argparse.ArgumentTypeError(f"hours cannot be negative: {value}")
    return hours


def _positive_hours(value: str) -> float:
    hours = _finite_hours(value)
    if hours <= 0:
        raise argparse.ArgumentTypeError(f"hours must be positive: {value}")
    return hours


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stall-monitor",
        description="Flag stalled shipments in a carrier tracking event stream.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="write a synthetic stream and its planted stalls")
    gen.add_argument("--out", type=Path, default=Path("data"))
    gen.add_argument("--shipments", type=int, default=GeneratorConfig.shipments)
    gen.add_argument("--days", type=int, default=GeneratorConfig.days)
    gen.add_argument("--seed", type=int, default=GeneratorConfig.seed)

    run = sub.add_parser("run", help="replay a stream and print alerts and stream counters")
    run.add_argument("events", type=Path)
    run.add_argument("--lateness", type=_non_negative_hours, default=4.0,
                     help="allowed lateness in hours (default 4)")
    run.add_argument("--retention", type=_positive_hours, default=24.0,
                     help="hours to keep delivered shipments after their last scan (default 24)")
    run.add_argument("--alerts-out", type=Path, help="also write every alert to this CSV")

    score = sub.add_parser("score", help="score alerts against planted stalls at several lateness values")
    score.add_argument("events", type=Path)
    score.add_argument("truth", type=Path)
    score.add_argument("--lateness", type=_non_negative_hours, nargs="+", default=[0.0, 2.0, 4.0, 6.0])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "generate":
            config = GeneratorConfig(shipments=args.shipments, days=args.days, seed=args.seed)
            n_events, n_stalls = write_dataset(args.out, config)
            print(f"wrote {n_events} events and {n_stalls} planted stalls to {args.out}")
        elif args.command == "run":
            _run(args)
        else:
            print(format_table(sweep(args.events, args.truth, args.lateness)))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _run(args: argparse.Namespace) -> None:
    monitor = StallMonitor(allowed_lateness=timedelta(hours=args.lateness),
                           state_retention=timedelta(hours=args.retention))
    alerts = monitor.run(read_events(args.events))
    by_kind = {kind: sum(1 for a in alerts if a.kind is kind) for kind in AlertKind}
    stats = monitor.stats
    print(f"events: {stats.accepted} accepted, {stats.duplicates} duplicates, "
          f"{stats.invalid} invalid, {stats.too_late} too late, {stats.out_of_order} out of order")
    print("alerts: " + ", ".join(f"{n} {kind.value.lower()}" for kind, n in by_kind.items()))
    open_stalls = monitor.open_stalls()
    print(f"open stalls at end of stream: {len(open_stalls)}")
    print(f"shipments held in state: {len(monitor.shipments)} ({stats.evicted} delivered and released)")
    with_dlq = [a for a in alerts if a.kind is AlertKind.STALLED and a.dead_letters]
    if with_dlq:
        ids = ", ".join(sorted({a.shipment_id for a in with_dlq}))
        print(f"stall alerts with dead-lettered scans (check the dead-letter list first): {ids}")
    if args.alerts_out:
        _write_alerts(args.alerts_out, alerts)
        print(f"wrote {len(alerts)} alerts to {args.alerts_out}")


def _write_alerts(path: Path, alerts: Sequence[Alert]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(ALERT_FIELDS)
        for a in alerts:
            writer.writerow([
                a.kind.value, a.shipment_id, a.stage.value, a.location,
                a.last_event_time.isoformat(),
                a.watermark.isoformat() if a.watermark else "",
                a.raised_at_ingest.isoformat(), a.dead_letters,
            ])


if __name__ == "__main__":
    raise SystemExit(main())
