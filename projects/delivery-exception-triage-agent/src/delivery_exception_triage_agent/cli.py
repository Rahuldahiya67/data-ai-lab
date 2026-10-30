"""Command line entry points.

``triage-eval``  run the baseline planner over a dataset and write the evaluation report.
``triage-run``   triage one ticket and print its trace step by step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import DEFAULT_MAX_STEPS, triage
from .dataset import DatasetError, load_shipments, load_tickets
from .evaluate import evaluate
from .planner import RuleBasedPlanner
from .report import format_resolution, write_report

PLANNERS = {"rule-based": RuleBasedPlanner}


def _load(data: Path):
    shipments_path, tickets_path = data / "shipments.json", data / "tickets.jsonl"
    for p in (shipments_path, tickets_path):
        if not p.is_file():
            raise DatasetError(f"{p} not found (run triage-generate --out {data} first)")
    return load_shipments(shipments_path), load_tickets(tickets_path)


def eval_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a triage planner on labelled tickets.")
    parser.add_argument("--data", type=Path, default=Path("data"), help="folder with shipments.json and tickets.jsonl")
    parser.add_argument("--out", type=Path, default=Path("reports"), help="folder for evaluation.json and .md")
    parser.add_argument("--planner", choices=sorted(PLANNERS), default="rule-based")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")
    try:
        shipments, tickets = _load(args.data)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report = evaluate(shipments, tickets, PLANNERS[args.planner](), max_steps=args.max_steps)
    json_path, md_path = write_report(report, args.planner, args.out)
    s = report.summary()
    print(
        f"{s['tickets']} tickets | action accuracy {s['action_accuracy']:.1%} | "
        f"resolution accuracy {s['resolution_accuracy']:.1%} | "
        f"tool precision {s['tool_precision']:.1%} | tool recall {s['tool_recall']:.1%} | "
        f"policy blocks {s['policy_blocks']} | tool errors {s['tool_errors']}"
    )
    print(f"wrote {json_path} and {md_path}")
    return 0


def run_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Triage one ticket and print the trace.")
    parser.add_argument("ticket_id")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--planner", choices=sorted(PLANNERS), default="rule-based")
    args = parser.parse_args(argv)
    try:
        shipments, tickets = _load(args.data)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    match = next((t for t in tickets if t.ticket.ticket_id == args.ticket_id), None)
    if match is None:
        print(f"error: no ticket {args.ticket_id!r} in {args.data}", file=sys.stderr)
        return 2
    result = triage(match.ticket, shipments, PLANNERS[args.planner]())
    print(f"{match.ticket.ticket_id} ({match.ticket.shipment_id}): {match.ticket.message}")
    for i, step in enumerate(result.steps, 1):
        outcome = f"ERROR {step.error}" if step.error else json.dumps(step.observation, default=str)
        print(f"  {i}. {step.call.name}({json.dumps(step.call.args)}) -> {outcome}")
    print(f"resolution: {format_resolution(result.resolution)}")
    print(f"expected:   {format_resolution(match.expected)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(eval_main())
