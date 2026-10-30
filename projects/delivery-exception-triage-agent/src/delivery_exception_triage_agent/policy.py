"""Guardrails checked before any action tool runs.

``check_action`` returns a list of violation codes; an empty list means the action
may run. Codes that match an escalation reason (``refund_above_limit``) are passed
on as that reason when the agent escalates a blocked action.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

from .models import Shipment
from .policy_rules import REDELIVERY_MAX_DAYS_AHEAD, REFUND_AUTO_LIMIT_INR, STUCK_AFTER_HOURS

if TYPE_CHECKING:  # pragma: no cover
    from .tools import Toolbox

COMPENSATION = ("issue_refund", "create_reship")


def _hours_since_last_scan(shipment: Shipment, toolbox: "Toolbox") -> float | None:
    if not shipment.scans:
        return None
    return (toolbox.now - shipment.scans[-1].at).total_seconds() / 3600


def check_action(name: str, args: dict[str, Any], shipment: Shipment | None, toolbox: "Toolbox") -> list[str]:
    if shipment is None:
        return ["shipment_not_found"]
    today = toolbox.now.date()
    v: list[str] = []

    if name in COMPENSATION and any(a.name in COMPENSATION for a in toolbox.ledger.for_shipment(shipment.shipment_id)):
        v.append("already_compensated")

    if name == "issue_refund":
        amount = args["amount_inr"]
        if shipment.status != "delivered":
            v.append("refund_requires_delivered_shipment")
        if amount <= 0:
            v.append("refund_amount_not_positive")
        if amount > shipment.declared_value_inr:
            v.append("refund_above_declared_value")
        if amount > REFUND_AUTO_LIMIT_INR:
            v.append("refund_above_limit")
    elif name == "create_reship":
        hours = _hours_since_last_scan(shipment, toolbox)
        if shipment.status != "in_transit":
            v.append("reship_requires_in_transit_shipment")
        elif hours is None:
            # Never scanned means never picked up: not lost in transit, so a person decides.
            v.append("reship_without_scan_history")
        elif hours <= STUCK_AFTER_HOURS:
            v.append("reship_before_stuck_threshold")
    elif name == "schedule_redelivery":
        if shipment.status != "delivery_failed":
            v.append("redelivery_requires_failed_delivery")
        if not shipment.address_valid:
            v.append("redelivery_to_invalid_address")
        d = args["date"]
        if not today + timedelta(days=1) <= d <= today + timedelta(days=REDELIVERY_MAX_DAYS_AHEAD):
            v.append("redelivery_date_out_of_window")
    elif name == "request_address":
        if shipment.status != "delivery_failed":
            v.append("address_request_requires_failed_delivery")
    elif name == "send_eta_update":
        if shipment.status == "delivered":
            v.append("eta_for_delivered_shipment")
        if args["eta_date"] < today:
            v.append("eta_in_the_past")
    return v
