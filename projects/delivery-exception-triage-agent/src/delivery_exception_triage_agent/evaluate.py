"""Run a planner over labelled tickets and score it against the ground truth.

Metrics, all computed from the agent trace and the ledger:

- action accuracy: the final action matches the expected one;
- resolution accuracy: the action and its details match (refund amount, redelivery
  date, escalation reason);
- tool-call precision and recall against the expected tool sequence, counted as
  multisets (order is not scored; calls that failed validation still count as calls);
- tool errors, policy blocks, planner errors and step-budget exhaustions;
- per-scenario accuracy and an expected-vs-actual confusion table.

Tickets run in file order against one shared ledger, the same way a support desk
works through a queue, so cross-ticket guardrails (no second compensation) apply.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from .agent import DEFAULT_MAX_STEPS, TriageResult, triage
from .models import LabelledTicket, Resolution, Shipment
from .planner import Planner
from .tools import ActionLedger

AMOUNT_TOLERANCE_INR = 0.01


def resolution_matches(expected: Resolution, actual: Resolution) -> bool:
    """True when the action and the detail that matters for that action agree."""
    if expected.action != actual.action:
        return False
    if expected.action == "issue_refund":
        if expected.amount_inr is None or actual.amount_inr is None:
            return False
        return abs(expected.amount_inr - actual.amount_inr) <= AMOUNT_TOLERANCE_INR
    if expected.action == "schedule_redelivery":
        return expected.redelivery_date == actual.redelivery_date
    if expected.action == "escalate":
        return expected.reason == actual.reason
    return True


@dataclass(frozen=True)
class TicketOutcome:
    ticket_id: str
    scenario: str
    message: str
    expected: Resolution
    actual: Resolution
    expected_tools: tuple[str, ...]
    tools_called: tuple[str, ...]
    tool_errors: int
    blocked: bool
    planner_error: str | None

    @property
    def action_correct(self) -> bool:
        return self.expected.action == self.actual.action

    @property
    def resolution_correct(self) -> bool:
        return resolution_matches(self.expected, self.actual)


@dataclass
class EvalReport:
    outcomes: list[TicketOutcome] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.outcomes)

    def _rate(self, hits: int) -> float:
        return hits / self.n if self.n else 0.0

    @property
    def action_accuracy(self) -> float:
        return self._rate(sum(o.action_correct for o in self.outcomes))

    @property
    def resolution_accuracy(self) -> float:
        return self._rate(sum(o.resolution_correct for o in self.outcomes))

    def _tool_overlap(self) -> tuple[int, int, int]:
        matched = called = expected = 0
        for o in self.outcomes:
            c, e = Counter(o.tools_called), Counter(o.expected_tools)
            matched += sum((c & e).values())
            called += sum(c.values())
            expected += sum(e.values())
        return matched, called, expected

    @property
    def tool_precision(self) -> float:
        matched, called, _ = self._tool_overlap()
        return matched / called if called else 0.0

    @property
    def tool_recall(self) -> float:
        matched, _, expected = self._tool_overlap()
        return matched / expected if expected else 0.0

    @property
    def tool_errors(self) -> int:
        return sum(o.tool_errors for o in self.outcomes)

    @property
    def policy_blocks(self) -> int:
        return sum(o.blocked for o in self.outcomes)

    @property
    def planner_errors(self) -> int:
        return sum(o.planner_error is not None for o in self.outcomes)

    @property
    def budget_exhausted(self) -> int:
        return sum(o.actual.reason == "step_budget_exhausted" for o in self.outcomes)

    def by_scenario(self) -> dict[str, tuple[int, int]]:
        """scenario -> (resolutions correct, tickets), in first-seen order."""
        out: dict[str, list[int]] = {}
        for o in self.outcomes:
            row = out.setdefault(o.scenario, [0, 0])
            row[0] += o.resolution_correct
            row[1] += 1
        return {k: (v[0], v[1]) for k, v in out.items()}

    def confusion(self) -> Counter:
        """(expected action, actual action) -> count."""
        return Counter((o.expected.action, o.actual.action) for o in self.outcomes)

    def mistakes(self) -> list[TicketOutcome]:
        return [o for o in self.outcomes if not o.resolution_correct]

    def summary(self) -> dict:
        return {
            "tickets": self.n,
            "action_accuracy": round(self.action_accuracy, 4),
            "resolution_accuracy": round(self.resolution_accuracy, 4),
            "tool_precision": round(self.tool_precision, 4),
            "tool_recall": round(self.tool_recall, 4),
            "tool_errors": self.tool_errors,
            "policy_blocks": self.policy_blocks,
            "planner_errors": self.planner_errors,
            "step_budget_exhausted": self.budget_exhausted,
        }


def _outcome(t: LabelledTicket, r: TriageResult) -> TicketOutcome:
    return TicketOutcome(
        ticket_id=t.ticket.ticket_id,
        scenario=t.scenario,
        message=t.ticket.message,
        expected=t.expected,
        actual=r.resolution,
        expected_tools=t.expected_tools,
        tools_called=r.tools_called,
        tool_errors=r.tool_errors,
        blocked=bool(r.blocked_actions),
        planner_error=r.planner_error,
    )


def evaluate(
    shipments: dict[str, Shipment],
    tickets: Iterable[LabelledTicket],
    planner: Planner,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> EvalReport:
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError(f"max_steps must be a positive integer, got {max_steps!r}")
    ledger = ActionLedger()
    report = EvalReport()
    for t in tickets:
        result = triage(t.ticket, shipments, planner, ledger, max_steps=max_steps)
        report.outcomes.append(_outcome(t, result))
    return report
