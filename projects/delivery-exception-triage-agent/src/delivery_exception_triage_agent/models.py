"""Plain data types shared by the generator, tools, policy and agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

# Final actions a ticket can be resolved with. Each maps to one action tool.
ACTIONS = (
    "send_eta_update",
    "create_reship",
    "schedule_redelivery",
    "request_address",
    "issue_refund",
    "escalate",
    "close_ticket",
)

SHIPMENT_STATUSES = ("in_transit", "out_for_delivery", "delivery_failed", "delivered")


@dataclass(frozen=True)
class ScanEvent:
    """One carrier scan: when, where and what happened."""

    at: datetime
    location: str
    code: str  # e.g. PICKED_UP, HUB_IN, HUB_OUT, OUT_FOR_DELIVERY, DELIVERED, ADDRESS_ISSUE


@dataclass(frozen=True)
class Shipment:
    shipment_id: str
    carrier: str
    origin_hub: str
    destination_city: str
    promised_date: date
    status: str
    declared_value_inr: float
    address_valid: bool
    scans: tuple[ScanEvent, ...] = ()


@dataclass(frozen=True)
class Ticket:
    ticket_id: str
    shipment_id: str
    created_at: datetime
    message: str


@dataclass(frozen=True)
class Resolution:
    """Ground truth (from the generator) or the agent's final decision."""

    action: str
    amount_inr: float | None = None
    redelivery_date: date | None = None
    reason: str = ""


@dataclass(frozen=True)
class LabelledTicket:
    """A ticket plus the hidden scenario and expected resolution used for evaluation."""

    ticket: Ticket
    scenario: str
    expected: Resolution
    expected_tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
