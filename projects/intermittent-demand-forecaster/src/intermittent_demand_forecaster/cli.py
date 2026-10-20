"""Command line interface.

    intermittent-demand generate --out data/demand.csv
    intermittent-demand backtest --input data/demand.csv --out results.csv
    intermittent-demand reorder  --input data/demand.csv --lead-time 4 --service-level 0.95 --out reorder.csv
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

from .backtest import backtest, select_best, summarise
from .generate import generate_demand
from .reorder import reorder_points, service_check


def _read(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["week"])


def _write(df: pd.DataFrame, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


def _run_backtest(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = _read(args.input)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = backtest(df, holdout=args.holdout)
    for w in caught:
        print(f"warning: {w.message}", file=sys.stderr)
    if results.empty:
        raise SystemExit("no SKU has enough history for this holdout")
    return df, results, select_best(results)


def cmd_generate(args) -> int:
    df = generate_demand(n_skus=args.skus, n_weeks=args.weeks, seed=args.seed)
    _write(df, args.out)
    print(f"wrote {len(df)} rows ({args.skus} SKUs x {args.weeks} weeks) to {args.out}")
    return 0


def cmd_backtest(args) -> int:
    _, results, best = _run_backtest(args)
    if args.out:
        _write(results, args.out)
        print(f"wrote {len(results)} rows to {args.out}")
    print(f"\nMean RMSSE by demand class (holdout {args.holdout} weeks, lower is better):")
    print(summarise(results).to_string())
    print("\nSelected method per SKU:")
    counts = best.groupby(["demand_class", "method"]).size().unstack(fill_value=0)
    print(counts.to_string())
    return 0


def cmd_reorder(args) -> int:
    df, _, best = _run_backtest(args)
    report = reorder_points(df, best, lead_time=args.lead_time, service_level=args.service_level)
    if args.out:
        _write(report, args.out)
        print(f"wrote {len(report)} rows to {args.out}")
    print(
        f"\nReorder points: lead time {args.lead_time} weeks, "
        f"service level {args.service_level:.0%}"
    )
    by_class = report.groupby("demand_class").agg(
        skus=("sku", "count"),
        lead_time_demand=("lead_time_demand", "sum"),
        safety_stock=("safety_stock", "sum"),
        reorder_point=("reorder_point", "sum"),
    )
    check = service_check(
        df, best, lead_time=args.lead_time, service_level=args.service_level, holdout=args.holdout
    )
    pooled = check.groupby("demand_class")[["covered", "windows"]].sum()
    by_class["achieved_service"] = pooled["covered"] / pooled["windows"]
    print(by_class.round({"lead_time_demand": 1, "safety_stock": 1, "achieved_service": 3}).to_string())
    print(
        f"\nachieved_service: share of {args.lead_time}-week windows in the last {args.holdout} weeks "
        "where demand stayed at or below the reorder point (backtest check)"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="intermittent-demand",
        description="Classify spare-parts demand, pick a forecasting method per SKU and set reorder points.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="write synthetic weekly demand to CSV")
    gen.add_argument("--skus", type=int, default=40)
    gen.add_argument("--weeks", type=int, default=104)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--out", required=True)
    gen.set_defaults(func=cmd_generate)

    for name, func, help_text in (
        ("backtest", cmd_backtest, "score every method on every SKU"),
        ("reorder", cmd_reorder, "reorder point per SKU from the selected method"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--input", required=True, help="CSV with columns sku, week, demand")
        p.add_argument("--holdout", type=int, default=26, help="weeks held out for scoring")
        p.add_argument("--out", help="write the full table to this CSV")
        p.set_defaults(func=func)
        if name == "reorder":
            p.add_argument("--lead-time", type=int, default=4, help="replenishment lead time in weeks")
            p.add_argument("--service-level", type=float, default=0.95, help="target cycle service level")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
