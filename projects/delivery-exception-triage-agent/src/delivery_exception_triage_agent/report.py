"""Write an evaluation report as JSON (for tooling) and Markdown (for people)."""

from __future__ import annotations

import json
from pathlib import Path

from .evaluate import EvalReport, TicketOutcome
from .models import Resolution


def format_resolution(r: Resolution) -> str:
    if r.action == "issue_refund" and r.amount_inr is not None:
        return f"issue_refund ({r.amount_inr:,.2f} INR)"
    if r.action == "schedule_redelivery" and r.redelivery_date is not None:
        return f"schedule_redelivery ({r.redelivery_date.isoformat()})"
    if r.action == "escalate" and r.reason:
        return f"escalate ({r.reason})"
    return r.action


def _cell(text: str) -> str:
    """Make free text safe inside a Markdown table cell."""
    return " ".join(str(text).split()).replace("|", "\\|")


def _outcome_dict(o: TicketOutcome) -> dict:
    return {
        "ticket_id": o.ticket_id,
        "scenario": o.scenario,
        "message": o.message,
        "expected": format_resolution(o.expected),
        "actual": format_resolution(o.actual),
        "tools_called": list(o.tools_called),
        "tool_errors": o.tool_errors,
        "blocked": o.blocked,
        "planner_error": o.planner_error,
    }


def to_json(report: EvalReport, planner_name: str) -> dict:
    return {
        "planner": planner_name,
        "summary": report.summary(),
        "by_scenario": {k: {"correct": ok, "tickets": n} for k, (ok, n) in report.by_scenario().items()},
        "confusion": [
            {"expected": e, "actual": a, "count": c} for (e, a), c in sorted(report.confusion().items())
        ],
        "mistakes": [_outcome_dict(o) for o in report.mistakes()],
    }


def to_markdown(report: EvalReport, planner_name: str, max_mistakes: int = 20) -> str:
    s = report.summary()
    lines = [
        f"# Triage evaluation: {planner_name}",
        "",
        f"{s['tickets']} tickets, one shared action ledger, tickets in file order.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Action accuracy | {s['action_accuracy']:.1%} |",
        f"| Resolution accuracy (action + amount/date/reason) | {s['resolution_accuracy']:.1%} |",
        f"| Tool-call precision | {s['tool_precision']:.1%} |",
        f"| Tool-call recall | {s['tool_recall']:.1%} |",
        f"| Tool errors | {s['tool_errors']} |",
        f"| Policy blocks | {s['policy_blocks']} |",
        f"| Planner errors | {s['planner_errors']} |",
        f"| Step budget exhausted | {s['step_budget_exhausted']} |",
        "",
        "## By scenario",
        "",
        "| Scenario | Correct | Tickets | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for name, (ok, n) in sorted(report.by_scenario().items()):
        lines.append(f"| {name} | {ok} | {n} | {ok / n:.0%} |")

    off_diagonal = sorted(((e, a), c) for (e, a), c in report.confusion().items() if e != a)
    lines += ["", "## Wrong actions", ""]
    if off_diagonal:
        lines += ["| Expected | Actual | Count |", "|---|---|---:|"]
        lines += [f"| {e} | {a} | {c} |" for (e, a), c in off_diagonal]
    else:
        lines.append("None.")

    mistakes = report.mistakes()
    lines += ["", f"## Mistakes ({len(mistakes)})", ""]
    if mistakes:
        lines += ["| Ticket | Scenario | Message | Expected | Got |", "|---|---|---|---|---|"]
        for o in mistakes[:max_mistakes]:
            lines.append(
                f"| {o.ticket_id} | {o.scenario} | {_cell(o.message)} | "
                f"{format_resolution(o.expected)} | {format_resolution(o.actual)} |"
            )
        if len(mistakes) > max_mistakes:
            lines.append(f"\n{len(mistakes) - max_mistakes} more in the JSON report.")
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def write_report(report: EvalReport, planner_name: str, out_dir: Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "evaluation.json"
    md_path = out_dir / "evaluation.md"
    json_path.write_text(json.dumps(to_json(report, planner_name), indent=2) + "\n", encoding="utf-8")
    md_path.write_text(to_markdown(report, planner_name), encoding="utf-8")
    return json_path, md_path
