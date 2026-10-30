"""The tool layer the agent calls.

Every call goes through ``Toolbox.execute``, which checks the tool name and the
arguments against a typed spec before running anything. A bad call raises
``ToolError``; the agent loop turns that into an observation for the planner, the
way a tool-calling API returns an error message instead of crashing.

Read tools return facts about the shipment. Action tools resolve the ticket; they
are checked against the policy guardrails first and are recorded in a ledger
that persists across tickets, so the same shipment cannot be compensated twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable

from .models import Shipment, Ticket, ToolCall
from .policy_rules import ESCALATION_REASONS


class ToolError(Exception):
    """A tool call was rejected before it ran (unknown tool or bad arguments)."""


# Argument kinds: "str", "number", "date" (ISO string), or a tuple of allowed strings.
@dataclass(frozen=True)
class ToolSpec:
    name: str
    params: dict[str, Any]
    kind: str  # "read" or "action"
    description: str


TOOL_SPECS: dict[str, ToolSpec] = {
    s.name: s
    for s in (
        ToolSpec("get_shipment", {"shipment_id": "str"}, "read", "Shipment status, promised date and declared value."),
        ToolSpec("get_tracking_events", {"shipment_id": "str"}, "read", "Scan history and hours since the last scan."),
        ToolSpec("check_address", {"shipment_id": "str"}, "read", "Whether the delivery address passed validation."),
        ToolSpec("send_eta_update", {"ticket_id": "str", "eta_date": "date"}, "action", "Reply with a new ETA."),
        ToolSpec("create_reship", {"shipment_id": "str"}, "action", "Send a replacement for a lost shipment."),
        ToolSpec("schedule_redelivery", {"shipment_id": "str", "date": "date"}, "action", "Book a new delivery attempt."),
        ToolSpec("request_address", {"ticket_id": "str"}, "action", "Ask the customer for a corrected address."),
        ToolSpec(
            "issue_refund",
            {"shipment_id": "str", "amount_inr": "number", "reason": ("damaged",)},
            "action",
            "Refund the customer.",
        ),
        ToolSpec("escalate", {"ticket_id": "str", "reason": ESCALATION_REASONS}, "action", "Hand over to a supervisor."),
        ToolSpec("close_ticket", {"ticket_id": "str"}, "action", "Close with no further action."),
    )
}

ACTION_TOOLS = frozenset(n for n, s in TOOL_SPECS.items() if s.kind == "action")


def validate_call(call: ToolCall) -> dict[str, Any]:
    """Check a call against its spec and return the arguments with dates parsed."""
    spec = TOOL_SPECS.get(call.name)
    if spec is None:
        raise ToolError(f"unknown tool {call.name!r}")
    if not isinstance(call.args, dict):
        raise ToolError(f"{call.name}: arguments must be an object")
    missing = sorted(set(spec.params) - set(call.args))
    extra = sorted(set(call.args) - set(spec.params))
    if missing:
        raise ToolError(f"{call.name}: missing argument(s) {', '.join(missing)}")
    if extra:
        raise ToolError(f"{call.name}: unexpected argument(s) {', '.join(extra)}")
    parsed: dict[str, Any] = {}
    for key, kind in spec.params.items():
        value = call.args[key]
        if kind == "str":
            if not isinstance(value, str) or not value.strip():
                raise ToolError(f"{call.name}: {key} must be a non-empty string")
            parsed[key] = value
        elif kind == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
                raise ToolError(f"{call.name}: {key} must be a number")
            parsed[key] = float(value)
        elif kind == "date":
            if not isinstance(value, str):
                raise ToolError(f"{call.name}: {key} must be an ISO date string")
            try:
                parsed[key] = date.fromisoformat(value)
            except ValueError:
                raise ToolError(f"{call.name}: {key} is not an ISO date: {value!r}") from None
        else:  # tuple of allowed values
            if value not in kind:
                raise ToolError(f"{call.name}: {key} must be one of {', '.join(kind)}")
            parsed[key] = value
    return parsed


@dataclass(frozen=True)
class ExecutedAction:
    ticket_id: str
    name: str
    args: dict[str, Any]


@dataclass
class ActionLedger:
    """Every action that actually ran, across all tickets in a session."""

    actions: list[ExecutedAction] = field(default_factory=list)

    def for_shipment(self, shipment_id: str) -> list[ExecutedAction]:
        return [a for a in self.actions if a.args.get("shipment_id") == shipment_id]


class Toolbox:
    """Tools bound to one ticket, over a shared shipment table and action ledger."""

    def __init__(
        self,
        shipments: dict[str, Shipment],
        ticket: Ticket,
        ledger: ActionLedger | None = None,
        policy: Callable[..., list[str]] | None = None,
    ):
        self.shipments = shipments
        self.ticket = ticket
        self.now: datetime = ticket.created_at
        self.ledger = ledger if ledger is not None else ActionLedger()
        if policy is None:
            from .policy import check_action

            policy = check_action
        self.policy = policy

    # -- helpers -----------------------------------------------------------
    def _shipment(self, call_name: str, shipment_id: str) -> Shipment:
        s = self.shipments.get(shipment_id)
        if s is None:
            raise ToolError(f"{call_name}: no shipment {shipment_id!r}")
        return s

    def _check_ticket(self, call_name: str, ticket_id: str) -> None:
        if ticket_id != self.ticket.ticket_id:
            raise ToolError(f"{call_name}: ticket_id {ticket_id!r} is not the ticket being handled")

    def _check_own_shipment(self, call_name: str, shipment_id: str) -> None:
        # Read tools may look up any shipment; actions may only touch the ticket's own.
        if shipment_id != self.ticket.shipment_id:
            raise ToolError(
                f"{call_name}: shipment_id {shipment_id!r} does not belong to ticket {self.ticket.ticket_id} "
                f"(expected {self.ticket.shipment_id!r})"
            )

    # -- entry point -------------------------------------------------------
    def execute(self, call: ToolCall) -> dict[str, Any]:
        args = validate_call(call)
        if call.name not in ACTION_TOOLS:
            return getattr(self, f"_tool_{call.name}")(args)

        if "ticket_id" in args:
            self._check_ticket(call.name, args["ticket_id"])
        shipment = None
        if "shipment_id" in args:
            self._check_own_shipment(call.name, args["shipment_id"])
            shipment = self._shipment(call.name, args["shipment_id"])
        else:
            shipment = self.shipments.get(self.ticket.shipment_id)

        violations = [] if call.name == "escalate" else self.policy(call.name, args, shipment, self)
        if violations:
            return {"executed": False, "blocked": True, "violations": violations}
        self.ledger.actions.append(ExecutedAction(self.ticket.ticket_id, call.name, dict(call.args)))
        return {"executed": True, "action": call.name}

    # -- read tools --------------------------------------------------------
    def _tool_get_shipment(self, args):
        s = self.shipments.get(args["shipment_id"])
        if s is None:
            return {"found": False, "shipment_id": args["shipment_id"]}
        return {
            "found": True,
            "shipment_id": s.shipment_id,
            "status": s.status,
            "carrier": s.carrier,
            "destination_city": s.destination_city,
            "promised_date": s.promised_date.isoformat(),
            "declared_value_inr": s.declared_value_inr,
        }

    def _tool_get_tracking_events(self, args):
        s = self._shipment("get_tracking_events", args["shipment_id"])
        last = s.scans[-1].at if s.scans else None
        hours = None if last is None else round((self.now - last).total_seconds() / 3600, 1)
        return {
            "shipment_id": s.shipment_id,
            "events": [{"at": e.at.isoformat(), "location": e.location, "code": e.code} for e in s.scans],
            "hours_since_last_scan": hours,
        }

    def _tool_check_address(self, args):
        s = self._shipment("check_address", args["shipment_id"])
        return {"shipment_id": s.shipment_id, "address_valid": s.address_valid}
