from dataclasses import replace
from datetime import date

import pytest

from delivery_exception_triage_agent.evaluate import evaluate, resolution_matches
from delivery_exception_triage_agent.generator import generate
from delivery_exception_triage_agent.models import LabelledTicket, Resolution, Ticket, ToolCall
from delivery_exception_triage_agent.planner import RuleBasedPlanner


@pytest.mark.parametrize(
    "expected, actual, ok",
    [
        (Resolution("issue_refund", amount_inr=1200.0), Resolution("issue_refund", amount_inr=1200.004), True),
        (Resolution("issue_refund", amount_inr=1200.0), Resolution("issue_refund", amount_inr=1100.0), False),
        (Resolution("issue_refund", amount_inr=1200.0), Resolution("issue_refund"), False),
        (Resolution("escalate", reason="delivery_dispute"), Resolution("escalate", reason="policy_block"), False),
        (Resolution("escalate", reason="delivery_dispute"), Resolution("escalate", reason="delivery_dispute"), True),
        (
            Resolution("schedule_redelivery", redelivery_date=date(2026, 10, 29)),
            Resolution("schedule_redelivery", redelivery_date=date(2026, 10, 30)),
            False,
        ),
        (Resolution("close_ticket"), Resolution("send_eta_update"), False),
        (Resolution("close_ticket"), Resolution("close_ticket"), True),
    ],
)
def test_resolution_matches(expected, actual, ok):
    assert resolution_matches(expected, actual) is ok


def _labelled(ticket, action, tools, **kw):
    return LabelledTicket(ticket, "s", Resolution(action, **kw), tuple(tools))


def test_metrics_on_a_hand_built_batch(in_transit_shipment, as_of):
    shipments = {"SHP-00001": in_transit_shipment}
    t1 = Ticket("TKT-00001", "SHP-00001", as_of, "Where is SHP-00001?")
    t2 = Ticket("TKT-00002", "SHP-00001", as_of, "Got it, thanks")
    tickets = [
        # Planner sends an ETA, as expected, using the expected three tools.
        _labelled(t1, "send_eta_update", ["get_shipment", "get_tracking_events", "send_eta_update"]),
        # Labelled as close_ticket with two tools; the planner sends an ETA with three.
        _labelled(t2, "close_ticket", ["get_shipment", "close_ticket"]),
    ]
    report = evaluate(shipments, tickets, RuleBasedPlanner())
    assert report.n == 2
    assert report.action_accuracy == 0.5
    assert report.resolution_accuracy == 0.5
    # 4 of 6 calls were expected; 4 of 5 expected calls were made.
    assert report.tool_precision == pytest.approx(4 / 6)
    assert report.tool_recall == pytest.approx(4 / 5)
    assert report.confusion()[("close_ticket", "send_eta_update")] == 1
    assert [m.ticket_id for m in report.mistakes()] == ["TKT-00002"]
    assert report.by_scenario() == {"s": (1, 2)}


class AlwaysRefund:
    def next_call(self, ticket, history):
        return ToolCall("issue_refund", {"shipment_id": ticket.shipment_id, "amount_inr": 100, "reason": "damaged"})


def test_blocked_actions_and_errors_are_counted(in_transit_shipment, as_of):
    t = Ticket("TKT-00001", "SHP-00001", as_of, "Where is SHP-00001?")
    report = evaluate({"SHP-00001": in_transit_shipment}, [_labelled(t, "send_eta_update", [])], AlwaysRefund())
    s = report.summary()
    assert s["policy_blocks"] == 1 and s["tool_errors"] == 0 and s["planner_errors"] == 0
    assert report.outcomes[0].actual == Resolution("escalate", reason="policy_block")


def test_empty_batch_reports_zeros():
    report = evaluate({}, [], RuleBasedPlanner())
    assert report.summary()["tickets"] == 0
    assert report.action_accuracy == 0.0 and report.tool_precision == 0.0


@pytest.mark.parametrize("bad", [0, -1, True, 2.5])
def test_max_steps_must_be_a_positive_integer(bad):
    with pytest.raises(ValueError, match="max_steps"):
        evaluate({}, [], RuleBasedPlanner(), max_steps=bad)


def test_baseline_on_the_generated_set():
    shipments, tickets = generate(n_tickets=300, seed=7)
    report = evaluate(shipments, tickets, RuleBasedPlanner())
    s = report.summary()
    assert s["tickets"] == 300
    assert s["tool_errors"] == 0 and s["planner_errors"] == 0 and s["step_budget_exhausted"] == 0
    # The baseline follows the expected tool sequence whenever it gets the intent right.
    assert s["tool_recall"] > 0.9
    assert s["resolution_accuracy"] <= s["action_accuracy"]
    # Wording misses only affect the message-driven scenarios.
    for scenario, (ok, n) in report.by_scenario().items():
        if scenario.startswith(("on_time", "late_", "failed_", "unknown")):
            assert ok == n, scenario


def test_shared_ledger_blocks_a_second_compensation_in_the_batch(in_transit_shipment, as_of):
    delivered = replace(in_transit_shipment, status="delivered", declared_value_inr=900.0)
    msg = "Parcel SHP-00001 arrived damaged"
    tickets = [
        _labelled(Ticket(f"TKT-0000{i}", "SHP-00001", as_of, msg), "issue_refund", [], amount_inr=900.0)
        for i in (1, 2)
    ]
    report = evaluate({"SHP-00001": delivered}, tickets, RuleBasedPlanner())
    assert [o.actual.action for o in report.outcomes] == ["issue_refund", "escalate"]
    assert report.outcomes[1].actual.reason == "policy_block"
