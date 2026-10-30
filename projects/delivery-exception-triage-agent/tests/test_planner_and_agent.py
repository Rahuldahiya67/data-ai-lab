from collections import Counter
from dataclasses import replace

import pytest

from delivery_exception_triage_agent.agent import triage
from delivery_exception_triage_agent.generator import generate
from delivery_exception_triage_agent.models import Ticket, ToolCall
from delivery_exception_triage_agent.planner import RuleBasedPlanner, detect_intent
from delivery_exception_triage_agent.tools import ActionLedger


@pytest.mark.parametrize(
    "message, intent",
    [
        ("Parcel arrived damaged, the box was crushed.", "damaged"),
        ("Tracking says delivered but I never received it.", "not_received"),
        ("I didn't get my parcel even though the app says delivered", "not_received"),
        ("Got it, thanks for the quick delivery.", "thanks"),
        ("Where is my parcel?", "status"),
    ],
)
def test_detect_intent(message, intent):
    assert detect_intent(message) == intent


@pytest.fixture
def ticket(as_of):
    return Ticket("TKT-00001", "SHP-00001", as_of, "Where is my parcel SHP-00001?")


def test_moving_late_parcel_gets_an_eta_for_tomorrow(in_transit_shipment, ticket):
    result = triage(ticket, {"SHP-00001": in_transit_shipment}, RuleBasedPlanner())
    assert result.resolution.action == "send_eta_update"
    assert result.tools_called == ("get_shipment", "get_tracking_events", "send_eta_update")
    assert result.steps[-1].call.args["eta_date"] == "2026-10-29"


def test_unknown_shipment_is_escalated(ticket):
    result = triage(ticket, {}, RuleBasedPlanner())
    assert result.resolution.action == "escalate"
    assert result.resolution.reason == "shipment_not_found"


class ScriptedPlanner:
    """Replays fixed calls; stands in for a model that makes mistakes."""

    def __init__(self, calls):
        self.calls = list(calls)

    def next_call(self, ticket, history):
        return self.calls[min(len(history), len(self.calls) - 1)]


def test_tool_errors_are_fed_back_and_the_run_continues(in_transit_shipment, ticket):
    planner = ScriptedPlanner(
        [
            ToolCall("get_shipment", {"id": "SHP-00001"}),  # wrong argument name
            ToolCall("get_shipment", {"shipment_id": "SHP-00001"}),
            ToolCall("close_ticket", {"ticket_id": "TKT-00001"}),
        ]
    )
    result = triage(ticket, {"SHP-00001": in_transit_shipment}, planner)
    assert result.tool_errors == 1
    assert "missing argument" in result.steps[0].error
    assert result.resolution.action == "close_ticket"


def test_blocked_action_becomes_an_escalation(in_transit_shipment, ticket):
    big = replace(in_transit_shipment, status="delivered", declared_value_inr=20000.0)
    planner = ScriptedPlanner(
        [ToolCall("issue_refund", {"shipment_id": "SHP-00001", "amount_inr": 20000, "reason": "damaged"})]
    )
    ledger = ActionLedger()
    result = triage(ticket, {"SHP-00001": big}, planner, ledger)
    assert result.blocked_actions == [["refund_above_limit"]]
    assert result.resolution.action == "escalate" and result.resolution.reason == "refund_above_limit"
    assert [a.name for a in ledger.actions] == ["escalate"]


def test_step_budget_escalates_a_planner_that_loops(in_transit_shipment, ticket):
    planner = ScriptedPlanner([ToolCall("get_shipment", {"shipment_id": "SHP-00001"})])
    ledger = ActionLedger()
    result = triage(ticket, {"SHP-00001": in_transit_shipment}, planner, ledger, max_steps=4)
    # Four planner steps plus the fallback escalation, which must appear in the trace.
    assert len(result.steps) == 5
    assert result.tools_called == ("get_shipment",) * 4 + ("escalate",)
    assert result.steps[-1].observation == {"executed": True, "action": "escalate"}
    assert result.resolution.reason == "step_budget_exhausted"
    assert [a.args["reason"] for a in ledger.actions] == ["step_budget_exhausted"]


def test_trace_ends_with_the_ledgered_action_for_every_generated_ticket():
    shipments, tickets = generate(n_tickets=60, seed=3)
    ledger = ActionLedger()
    for t in tickets:
        r = triage(t.ticket, shipments, RuleBasedPlanner(), ledger)
        assert r.tools_called[-1] == ledger.actions[-1].name == r.resolution.action


def test_rule_based_planner_on_generated_tickets():
    shipments, tickets = generate(n_tickets=300, seed=7)
    ledger = ActionLedger()
    correct = Counter()
    total = Counter()
    for t in tickets:
        r = triage(t.ticket, shipments, RuleBasedPlanner(), ledger)
        total[t.scenario] += 1
        correct[t.scenario] += r.resolution.action == t.expected.action
        assert r.tool_errors == 0
    # Every ticket is resolved with exactly one executed action.
    assert len(ledger.actions) == len(tickets)
    # Status-driven scenarios do not depend on wording, so the baseline gets them all.
    for s in ("on_time_inquiry", "late_moving", "late_stuck", "failed_bad_address",
              "failed_customer_absent", "unknown_shipment"):
        assert correct[s] == total[s], s
    assert sum(correct.values()) / len(tickets) > 0.85
