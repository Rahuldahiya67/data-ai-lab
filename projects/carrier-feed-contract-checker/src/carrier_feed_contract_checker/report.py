"""Turn a validation result into a feed quality report (JSON-ready dict and Markdown)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .contract import Contract
from .validate import ValidationResult

# Row 0 of the frame is line 2 of the CSV file (line 1 is the header).
CSV_HEADER_LINES = 1


def csv_line(row: int) -> int:
    """Line number in the source CSV for a 0-based frame row."""
    return row + CSV_HEADER_LINES + 1


def build_report(
    result: ValidationResult,
    contract: Contract,
    feed_name: str,
    sample_size: int = 10,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Summarise ``result`` as a plain dict that can be written as JSON."""
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 0:
        # A negative value would reach DataFrame.head(), which then returns
        # every row except the last ones instead of a sample.
        raise ValueError(f"sample_size must be a whole number >= 0, got {sample_size!r}")
    generated_at = generated_at or datetime.now(timezone.utc)
    rule_columns: dict[str, str] = {}
    for v in result.violations:
        rule_columns.setdefault(v.rule, v.column)

    samples = []
    for idx, row in result.quarantined.head(sample_size).iterrows():
        samples.append(
            {
                "csv_line": csv_line(int(idx)),
                "violations": row["_violations"].split("; "),
            }
        )

    return {
        "feed": feed_name,
        "contract": {"name": contract.name, "version": contract.version},
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "verdict": "PASS" if result.passed else "FAIL",
        "rows": {
            "total": result.total_rows,
            "accepted": len(result.accepted),
            "quarantined": len(result.quarantined),
            "reject_rate": round(result.reject_rate, 4),
        },
        "feed_checks": [
            {"name": c.name, "passed": c.passed, "detail": c.detail} for c in result.feed_checks
        ],
        "violations_by_rule": [
            {"rule": rule, "column": rule_columns.get(rule, ""), "count": count}
            for rule, count in result.violation_counts().items()
        ],
        "sample_quarantined_rows": samples,
    }


def _cell(value: Any) -> str:
    """Make a value safe inside a Markdown table cell.

    Feed check details quote column names straight from the carrier's header,
    so a ``|`` or a line break there would otherwise split or end the table row.
    """
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def render_markdown(report: dict[str, Any]) -> str:
    """Render a report dict as a short Markdown page for the carrier or the ops team."""
    rows = report["rows"]
    lines = [
        f"# Feed quality report: {report['feed']}",
        "",
        f"- Contract: `{report['contract']['name']}` v{report['contract']['version']}",
        f"- Checked at: {report['generated_at']}",
        f"- Verdict: **{report['verdict']}**",
        f"- Rows: {rows['total']} total, {rows['accepted']} accepted, "
        f"{rows['quarantined']} quarantined ({rows['reject_rate']:.1%})",
        "",
        "## Feed checks",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for c in report["feed_checks"]:
        lines.append(f"| {_cell(c['name'])} | {'pass' if c['passed'] else 'FAIL'} | {_cell(c['detail'])} |")

    lines += ["", "## Violations by rule", ""]
    if report["violations_by_rule"]:
        lines += ["| Rule | Column | Rows |", "|---|---|---:|"]
        for v in report["violations_by_rule"]:
            lines.append(f"| {_cell(v['rule'])} | {_cell(v['column'])} | {v['count']} |")
    else:
        lines.append("No row-level violations.")

    if report["sample_quarantined_rows"]:
        lines += [
            "",
            "## Sample quarantined rows",
            "",
            "Line numbers refer to the source CSV, counting the header as line 1.",
            "",
            "| CSV line | Rules broken |",
            "|---:|---|",
        ]
        for s in report["sample_quarantined_rows"]:
            lines.append(f"| {s['csv_line']} | {_cell(', '.join(s['violations']))} |")
    return "\n".join(lines) + "\n"
