from dataclasses import replace
from datetime import datetime

import pytest

from delivery_exception_triage_agent.models import ScanEvent, Ticket, ToolCall
from delivery_exception_triage_agent.tools import ActionLedger, ToolError, Toolbox, validate_call


@pytest.fixture
def ticket(as_of):
    return Ticket("TKT-00001", "SHP-00001", as_of, "Where is my parcel SHP-00001?")


@pytest.fixture
def box(in_transit_shipment, ticket):
    return Toolbox({"SHP-00001": in_transit_shipment}, ticket)


def call(name, **args):
    return ToolCall(name, args)


# -- argument validation ------------------------------------------------------

@pytest.mark.parametrize(
    "bad, message",
    [
        (call("cancel_order", shipment_id="SHP-00001"), "unknown tool"),
        (call("get_shipment"), "missing argument"),
        (call("get_shipment", shipment_id="SHP-00001", force=True), "unexpected argument"),
        (call("get_shipment", shipment_id=""), "non-empty string"),
        (call("issue_refund", shipment_id="S", amount_inr="100", reason="damaged"), "must be a number"),
        (call("issue_refund", shipment_id="S", amount_inr=True, reason="damaged"), "must be a number"),
        (call("issue_refund", shipment_id="S", amount_inr=float("nan"), reason="damaged"), "must be a number"),
        (call("issue_refund", shipment_id="S", amount_inr=10, reason="goodwill"), "must be one of"),
        (call("schedule_redelivery", shipment_id="S", date="29-10-2026"), "not an ISO date"),
        (call("escalate", ticket_id="T", reason="angry"), "must be one of"),
    ],
)
def test_validate_call_rejects_bad_calls(bad, message):
    with pytest.raises(ToolError, match=message):
        validate_call(bad)


def test_validate_call_parses_dates_and_numbers():
    args = validate_call(call("issue_refund", shipment_id="S", amount_inr=100, reason="damaged"))
    assert args["amount_inr"] == 100.0 and isinstance(args["amount_inr"], float)
    args = validate_call(call("schedule_redelivery", shipment_id="S", date="2026-10-29"))
    assert str(args["date"]) == "2026-10-29"


# -- read tools ----------------------------------------------------------------

def test_get_shipment_reports_unknown_ids_as_not_found(box):
    assert box.execute(call("get_shipment", shipment_id="SHP-99999")) == {"found": False, "shipment_id": "SHP-99999"}
    obs = box.execute(call("get_shipment", shipment_id="SHP-00001"))
    assert obs["found"] and obs["status"] == "in_transit" and obs["promised_date"] == "2026-10-26"


def test_tracking_reports_hours_since_last_scan(box):
    obs = box.execute(call("get_tracking_events", shipment_id="SHP-00001"))
    assert obs["hours_since_last_scan"] == 14.0  # 27 Oct 20:00 -> 28 Oct 10:00
    assert [e["code"] for e in obs["events"]] == ["PICKED_UP", "HUB_IN"]


def test_tracking_on_unknown_shipment_is_a_tool_error(box):
    with pytest.raises(ToolError, match="no shipment"):
        box.execute(call("get_tracking_events", shipment_id="SHP-99999"))


# -- action tools and guardrails ----------------------------------------------------

def test_action_on_another_ticket_is_rejected(box):
    with pytest.raises(ToolError, match="not the ticket being handled"):
        box.execute(call("close_ticket", ticket_id="TKT-00002"))


def test_eta_update_runs_and_is_recorded(box):
    obs = box.execute(call("send_eta_update", ticket_id="TKT-00001", eta_date="2026-10-29"))
    assert obs == {"executed": True, "action": "send_eta_update"}
    assert [a.name for a in box.ledger.actions] == ["send_eta_update"]


def test_eta_in_the_past_is_blocked(box):
    obs = box.execute(call("send_eta_update", ticket_id="TKT-00001", eta_date="2026-10-27"))
    assert obs["blocked"] and obs["violations"] == ["eta_in_the_past"]
    assert box.ledger.actions == []


def test_reship_is_blocked_while_the_parcel_is_still_moving(box):
    obs = box.execute(call("create_reship", shipment_id="SHP-00001"))
    assert obs["violations"] == ["reship_before_stuck_threshold"]


def test_reship_allowed_once_stuck_and_only_once(in_transit_shipment, ticket):
    stuck = replace(in_transit_shipment, scans=(ScanEvent(datetime(2026, 10, 24, 9, 0), "Delhi", "PICKED_UP"),))
    ledger = ActionLedger()
    box = Toolbox({"SHP-00001": stuck}, ticket, ledger)
    assert box.execute(call("create_reship", shipment_id="SHP-00001"))["executed"]
    second = Toolbox({"SHP-00001": stuck}, replace(ticket, ticket_id="TKT-00002"), ledger)
    assert second.execute(call("create_reship", shipment_id="SHP-00001"))["violations"] == ["already_compensated"]


def _delivered(shipment, value):
    return replace(shipment, status="delivered", declared_value_inr=value)


def test_refund_rules(in_transit_shipment, ticket):
    box = Toolbox({"SHP-00001": in_transit_shipment}, ticket)
    assert "refund_requires_delivered_shipment" in box.execute(
        call("issue_refund", shipment_id="SHP-00001", amount_inr=100, reason="damaged")
    )["violations"]

    box = Toolbox({"SHP-00001": _delivered(in_transit_shipment, 2400.0)}, ticket)
    assert box.execute(call("issue_refund", shipment_id="SHP-00001", amount_inr=3000, reason="damaged"))[
        "violations"
    ] == ["refund_above_declared_value"]
    assert box.execute(call("issue_refund", shipment_id="SHP-00001", amount_inr=0, reason="damaged"))[
        "violations"
    ] == ["refund_amount_not_positive"]
    assert box.execute(call("issue_refund", shipment_id="SHP-00001", amount_inr=2400, reason="damaged"))["executed"]

    box = Toolbox({"SHP-00001": _delivered(in_transit_shipment, 12000.0)}, ticket)
    assert box.execute(call("issue_refund", shipment_id="SHP-00001", amount_inr=12000, reason="damaged"))[
        "violations"
    ] == ["refund_above_limit"]


@pytest.mark.parametrize(
    "when, ok",
    [("2026-10-28", False), ("2026-10-29", True), ("2026-11-04", True), ("2026-11-05", False)],
)
def test_redelivery_window(in_transit_shipment, ticket, when, ok):
    failed = replace(in_transit_shipment, status="delivery_failed")
    box = Toolbox({"SHP-00001": failed}, ticket)
    obs = box.execute(call("schedule_redelivery", shipment_id="SHP-00001", date=when))
    assert obs.get("executed", False) is ok


def test_redelivery_to_an_invalid_address_is_blocked(in_transit_shipment, ticket):
    failed = replace(in_transit_shipment, status="delivery_failed", address_valid=False)
    box = Toolbox({"SHP-00001": failed}, ticket)
    obs = box.execute(call("schedule_redelivery", shipment_id="SHP-00001", date="2026-10-29"))
    assert obs["violations"] == ["redelivery_to_invalid_address"]


def test_escalation_is_never_blocked(box):
    assert box.execute(call("escalate", ticket_id="TKT-00001", reason="unclear_request"))["executed"]
