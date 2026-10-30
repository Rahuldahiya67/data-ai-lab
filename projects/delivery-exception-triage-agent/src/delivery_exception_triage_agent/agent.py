"""The agent loop: ask the planner for a call, run it, feed the result back.

The loop stops when an action tool runs. Two safety nets keep every ticket
resolved: a blocked action is turned into an escalation carrying the violation,
and a run that hits the step budget is escalated as ``step_budget_exhausted``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .models import Resolution, Shipment, Ticket, ToolCall
from .planner import Planner, Step
from .policy_rules import ESCALATION_REASONS
from .tools import ACTION_TOOLS, ActionLedger, ToolError, Toolbox

DEFAULT_MAX_STEPS = 8


@dataclass
class TriageResult:
    ticket_id: str
    resolution: Resolution
    steps: list[Step] = field(default_factory=list)
    tool_errors: int = 0
    blocked_actions: list[list[str]] = field(default_factory=list)

    @property
    def tools_called(self) -> tuple[str, ...]:
        return tuple(s.call.name for s in self.steps)


def _resolution_from(call: ToolCall) -> Resolution:
    a = call.args
    if call.name == "issue_refund":
        return Resolution("issue_refund", amount_inr=float(a["amount_inr"]))
    if call.name == "schedule_redelivery":
        return Resolution("schedule_redelivery", redelivery_date=date.fromisoformat(a["date"]))
    if call.name == "escalate":
        return Resolution("escalate", reason=a["reason"])
    return Resolution(call.name)


def triage(
    ticket: Ticket,
    shipments: dict[str, Shipment],
    planner: Planner,
    ledger: ActionLedger | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> TriageResult:
    box = Toolbox(shipments, ticket, ledger)
    result = TriageResult(ticket.ticket_id, Resolution("escalate", reason="step_budget_exhausted"))

    for _ in range(max_steps):
        call = planner.next_call(ticket, list(result.steps))
        try:
            obs = box.execute(call)
        except ToolError as exc:
            result.tool_errors += 1
            result.steps.append(Step(call, error=str(exc)))
            continue
        result.steps.append(Step(call, observation=obs))
        if call.name not in ACTION_TOOLS:
            continue
        if obs.get("executed"):
            result.resolution = _resolution_from(call)
            return result
        # Blocked by policy: escalate with the most specific reason we have.
        violations = obs["violations"]
        result.blocked_actions.append(violations)
        reason = next((v for v in violations if v in ESCALATION_REASONS), "policy_block")
        fallback = ToolCall("escalate", {"ticket_id": ticket.ticket_id, "reason": reason})
        result.steps.append(Step(fallback, observation=box.execute(fallback)))
        result.resolution = Resolution("escalate", reason=reason)
        return result

    # Out of steps: escalate, and keep the escalation in the trace so it agrees with the ledger.
    fallback = ToolCall("escalate", {"ticket_id": ticket.ticket_id, "reason": "step_budget_exhausted"})
    result.steps.append(Step(fallback, observation=box.execute(fallback)))
    return result
