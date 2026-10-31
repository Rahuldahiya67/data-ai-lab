"""Planners decide the next tool call from the ticket and the steps so far.

``Planner`` is the interface the agent loop drives. ``RuleBasedPlanner`` is the
offline baseline: keyword intent detection on the customer message plus the
shipment state from the read tools. An LLM-backed planner can implement the same
interface; nothing in the agent depends on how the decision is made.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Protocol

from .models import Ticket, ToolCall
from .policy_rules import REFUND_AUTO_LIMIT_INR, STUCK_AFTER_HOURS


@dataclass(frozen=True)
class Step:
    call: ToolCall
    observation: dict[str, Any] | None = None
    error: str | None = None


class Planner(Protocol):
    def next_call(self, ticket: Ticket, history: list[Step]) -> ToolCall: ...


DAMAGE_WORDS = (
    "damaged", "damage", "broken", "crushed", "cracked",
    "smashed", "shattered", "dented", "torn", "leaking",
)
NEGATIONS = ("no", "not", "without", "never", "isn't", "wasn't", "nothing")
NOT_RECEIVED_PATTERNS = (
    r"never (received|got)",
    r"did ?n[o']?t (get|receive)",
    r"nothing at my door",
    r"nobody here received",
)
THANKS_WORDS = ("thanks", "thank you", "all good")

# Words worth correcting when a customer mistypes them. Only tokens of five or
# more letters are corrected, and only for a swapped, missing or doubled letter.
# Substitutions are not corrected: they turn real words into key words
# ("broker" -> "broken", "rented" -> "dented").
_KEY_WORDS = tuple(sorted({*DAMAGE_WORDS, "received", "receive", "thanks"}))


def _within_one_edit(a: str, b: str) -> bool:
    """True if ``a`` becomes ``b`` with one inserted letter, one deleted letter or one adjacent swap."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = [i for i in range(la) if a[i] != b[i]]
        return len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]
    short, long_ = (a, b) if la < lb else (b, a)
    i = 0
    while i < len(short) and short[i] == long_[i]:
        i += 1
    return short[i:] == long_[i + 1 :]


def _correct(token: str) -> str:
    if len(token) < 5 or token in _KEY_WORDS:
        return token
    for word in _KEY_WORDS:
        if len(word) >= 5 and _within_one_edit(token, word):
            return word
    return token


def _tokens(message: str) -> list[str]:
    return [_correct(t) for t in re.findall(r"[a-z']+", message.lower())]


def _mentions_damage(tokens: list[str]) -> bool:
    """A damage word counts unless one of the two words before it negates it ("no damage")."""
    for i, tok in enumerate(tokens):
        if tok in DAMAGE_WORDS and not any(t in NEGATIONS for t in tokens[max(0, i - 2) : i]):
            return True
    return False


def detect_intent(message: str) -> str:
    """Return one of: damaged, not_received, thanks, status.

    Matching is on whole words after correcting one-letter typos in the key words,
    so "recieved" and "htanks" are understood and "no damage, just late" is not
    read as a damage claim.
    """
    tokens = _tokens(message)
    text = " ".join(tokens)
    if _mentions_damage(tokens):
        return "damaged"
    if any(re.search(p, text) for p in NOT_RECEIVED_PATTERNS):
        return "not_received"
    if any(re.search(rf"\b{w}\b", text) for w in THANKS_WORDS):
        return "thanks"
    return "status"


def _last_observation(history: list[Step], tool: str) -> dict[str, Any] | None:
    for step in reversed(history):
        if step.call.name == tool and step.observation is not None:
            return step.observation
    return None


class RuleBasedPlanner:
    """Deterministic baseline planner; no model, no network."""

    def next_call(self, ticket: Ticket, history: list[Step]) -> ToolCall:
        sid = ticket.shipment_id
        tid = ticket.ticket_id
        today: date = ticket.created_at.date()

        shipment = _last_observation(history, "get_shipment")
        if shipment is None:
            return ToolCall("get_shipment", {"shipment_id": sid})
        if not shipment["found"]:
            return ToolCall("escalate", {"ticket_id": tid, "reason": "shipment_not_found"})

        status = shipment["status"]
        intent = detect_intent(ticket.message)

        if status == "delivered":
            if intent == "damaged":
                value = shipment["declared_value_inr"]
                if value > REFUND_AUTO_LIMIT_INR:
                    return ToolCall("escalate", {"ticket_id": tid, "reason": "refund_above_limit"})
                return ToolCall("issue_refund", {"shipment_id": sid, "amount_inr": value, "reason": "damaged"})
            if intent == "not_received":
                if _last_observation(history, "get_tracking_events") is None:
                    return ToolCall("get_tracking_events", {"shipment_id": sid})
                return ToolCall("escalate", {"ticket_id": tid, "reason": "delivery_dispute"})
            if intent == "thanks":
                return ToolCall("close_ticket", {"ticket_id": tid})
            return ToolCall("escalate", {"ticket_id": tid, "reason": "unclear_request"})

        tracking = _last_observation(history, "get_tracking_events")
        if tracking is None:
            return ToolCall("get_tracking_events", {"shipment_id": sid})

        if status == "delivery_failed":
            address = _last_observation(history, "check_address")
            if address is None:
                return ToolCall("check_address", {"shipment_id": sid})
            if not address["address_valid"]:
                return ToolCall("request_address", {"ticket_id": tid})
            when = today + timedelta(days=1)
            return ToolCall("schedule_redelivery", {"shipment_id": sid, "date": when.isoformat()})

        # in_transit or out_for_delivery
        hours = tracking["hours_since_last_scan"]
        if status == "in_transit" and hours is not None and hours > STUCK_AFTER_HOURS:
            return ToolCall("create_reship", {"shipment_id": sid})
        promised = date.fromisoformat(shipment["promised_date"])
        eta = promised if promised >= today else today + timedelta(days=1)
        return ToolCall("send_eta_update", {"ticket_id": tid, "eta_date": eta.isoformat()})
