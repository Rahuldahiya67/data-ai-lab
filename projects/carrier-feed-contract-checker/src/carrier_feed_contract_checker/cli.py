"""Command line entry point: generate a synthetic feed, or check a feed against a contract.

Exit codes for ``check``: 0 feed passed, 1 feed failed its checks, 2 the contract
or feed could not be read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .contract import ContractError, load_contract
from .generate import GeneratorSpec, write_feed
from .report import build_report, render_markdown
from .validate import validate_file

EXIT_PASS, EXIT_FAIL, EXIT_ERROR = 0, 1, 2


def _generate(args: argparse.Namespace) -> int:
    spec = GeneratorSpec(rows=args.rows, defect_rate=args.defect_rate, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df = write_feed(args.out, spec, keep_labels=args.keep_labels)
    defects = int((df["_defect"] != "").sum())
    print(f"Wrote {len(df)} rows to {args.out} ({defects} with injected defects)")
    return EXIT_PASS


def _check(args: argparse.Namespace) -> int:
    try:
        contract = load_contract(args.contract)
        result = validate_file(args.feed, contract)
    except (ContractError, OSError, ValueError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.accepted.to_csv(out / "accepted.csv", index=False)
    result.quarantined.to_csv(out / "quarantine.csv", index=False)
    report = build_report(result, contract, Path(args.feed).name, sample_size=args.sample)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")

    rows = report["rows"]
    print(
        f"{report['verdict']}: {rows['accepted']} accepted, {rows['quarantined']} quarantined "
        f"of {rows['total']} rows ({rows['reject_rate']:.1%})"
    )
    for c in result.feed_checks:
        if not c.passed:
            print(f"  failed {c.name}: {c.detail}")
    print(f"Outputs written to {out}/")
    return EXIT_PASS if result.passed else EXIT_FAIL


def _non_negative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a whole number: {value!r}") from exc
    if number < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {number}")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="carrier-feed-check",
        description="Check carrier shipment feeds against a YAML data contract.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="write a synthetic carrier feed")
    gen.add_argument("--out", required=True, help="CSV file to write")
    gen.add_argument("--rows", type=int, default=1000)
    gen.add_argument("--defect-rate", type=float, default=0.08)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--keep-labels", action="store_true", help="keep the _defect ground-truth column")
    gen.set_defaults(func=_generate)

    chk = sub.add_parser("check", help="validate a feed and write accepted, quarantine and report files")
    chk.add_argument("feed", help="CSV feed to check")
    chk.add_argument("--contract", required=True, help="YAML contract file")
    chk.add_argument("--out-dir", default="out", help="directory for outputs (default: out)")
    chk.add_argument("--sample", type=_non_negative_int, default=10, help="quarantined rows to show in the report")
    chk.set_defaults(func=_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ValueError as exc:  # e.g. invalid generator settings
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
