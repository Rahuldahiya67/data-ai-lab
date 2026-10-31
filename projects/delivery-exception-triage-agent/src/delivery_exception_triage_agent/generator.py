"""Synthetic shipments and labelled support tickets.

Each ticket is built from a hidden scenario. The scenario decides the shipment's
state (status, scans, address flag, declared value) and the expected resolution.
Customer messages are drawn from several phrasings, with occasional typos and
vague wording, so a planner cannot rely on one keyword per scenario.
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta

from .models import LabelledTicket, Resolution, ScanEvent, Shipment, Ticket
from .policy_rules import REFUND_AUTO_LIMIT_INR, STUCK_AFTER_HOURS

CARRIERS = ("BlueLine", "Kestrel Express", "TransIndo", "QuickHaul")
HUBS = ("Delhi", "Gurugram", "Noida", "Jaipur", "Lucknow", "Ahmedabad", "Mumbai", "Pune")
CITIES = ("Jaipur", "Chandigarh", "Dehradun", "Agra", "Indore", "Bhopal", "Surat", "Nagpur")

# scenario -> (weight, expected tools in order)
SCENARIOS: dict[str, tuple[float, tuple[str, ...]]] = {
    "on_time_inquiry": (0.12, ("get_shipment", "get_tracking_events", "send_eta_update")),
    "late_moving": (0.16, ("get_shipment", "get_tracking_events", "send_eta_update")),
    "late_stuck": (0.12, ("get_shipment", "get_tracking_events", "create_reship")),
    "failed_bad_address": (0.10, ("get_shipment", "get_tracking_events", "check_address", "request_address")),
    "failed_customer_absent": (0.10, ("get_shipment", "get_tracking_events", "check_address", "schedule_redelivery")),
    "damaged_low_value": (0.12, ("get_shipment", "issue_refund")),
    "damaged_high_value": (0.08, ("get_shipment", "escalate")),
    "delivered_not_received": (0.10, ("get_shipment", "get_tracking_events", "escalate")),
    "unknown_shipment": (0.05, ("get_shipment", "escalate")),
    "delivered_thanks": (0.05, ("get_shipment", "close_ticket")),
}

MESSAGES: dict[str, tuple[str, ...]] = {
    "where": (
        "Where is my parcel {sid}? It was supposed to arrive by now.",
        "Order {sid} still hasn't arrived, what is going on?",
        "Hi, any update on shipment {sid}?",
        "My delivery {sid} is late. When will it come?",
        "Tracking for {sid} has not moved, please check.",
        "Can you tell me when {sid} will be delivered?",
    ),
    "failed": (
        "The driver said he could not find my address for {sid}.",
        "Delivery attempt failed for {sid}, nobody called me.",
        "Got a message that {sid} could not be delivered. Please help.",
        "Hi, any update on shipment {sid}?",
    ),
    "damaged": (
        "Parcel {sid} arrived damaged, the box was crushed.",
        "The item in {sid} is broken. I want my money back.",
        "{sid} came today but the product inside is cracked.",
        "Received {sid} smashed, totally unusable.",
        "Package {sid} was delivered with water damage.",
    ),
    "not_received": (
        "Tracking says {sid} was delivered but I never received it.",
        "{sid} shows delivered, there is nothing at my door.",
        "I did not get my parcel {sid} even though the app says delivered.",
        "Nobody here received {sid}. Who signed for it?",
    ),
    "thanks": (
        "Got {sid}, thanks for the quick delivery.",
        "Received {sid} today, all good. You can close this.",
    ),
}

SCENARIO_MESSAGES = {
    "on_time_inquiry": "where",
    "late_moving": "where",
    "late_stuck": "where",
    "failed_bad_address": "failed",
    "failed_customer_absent": "failed",
    "damaged_low_value": "damaged",
    "damaged_high_value": "damaged",
    "delivered_not_received": "not_received",
    "unknown_shipment": "where",
    "delivered_thanks": "thanks",
}


def _typo(text: str, rng: random.Random) -> str:
    """Swap two adjacent letters in one word longer than four characters."""
    words = text.split(" ")
    candidates = [i for i, w in enumerate(words) if len(w) > 4 and w.isalpha()]
    if not candidates:
        return text
    i = rng.choice(candidates)
    w = words[i]
    j = rng.randrange(len(w) - 1)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2 :]
    return " ".join(words)


def _scans_in_transit(rng, origin, picked_up, last_scan_at) -> tuple[ScanEvent, ...]:
    hub = rng.choice([h for h in HUBS if h != origin])
    return (
        ScanEvent(picked_up, origin, "PICKED_UP"),
        ScanEvent(picked_up + timedelta(hours=6), origin, "HUB_OUT"),
        ScanEvent(last_scan_at, hub, "HUB_IN"),
    )


def _build_case(scenario: str, idx: int, as_of: datetime, rng: random.Random):
    sid = f"SHP-{idx:05d}"
    carrier = rng.choice(CARRIERS)
    origin = rng.choice(HUBS)
    city = rng.choice(CITIES)
    created_at = as_of - timedelta(minutes=rng.randint(0, 600))
    picked_up = created_at - timedelta(days=rng.randint(3, 6), hours=rng.randint(0, 12))
    value = round(rng.uniform(300, REFUND_AUTO_LIMIT_INR), 2)
    address_valid = True
    expected_kwargs: dict = {}

    if scenario == "on_time_inquiry":
        status = "in_transit"
        promised = created_at.date() + timedelta(days=rng.randint(1, 3))
        last = created_at - timedelta(hours=rng.randint(2, 30))
        scans = _scans_in_transit(rng, origin, picked_up, last)
        action = "send_eta_update"
    elif scenario == "late_moving":
        status = "in_transit"
        promised = created_at.date() - timedelta(days=rng.randint(1, 3))
        last = created_at - timedelta(hours=rng.randint(2, STUCK_AFTER_HOURS - 12))
        scans = _scans_in_transit(rng, origin, picked_up, last)
        action = "send_eta_update"
    elif scenario == "late_stuck":
        status = "in_transit"
        promised = created_at.date() - timedelta(days=rng.randint(2, 5))
        last = created_at - timedelta(hours=rng.randint(STUCK_AFTER_HOURS + 6, STUCK_AFTER_HOURS + 96))
        picked_up = min(picked_up, last - timedelta(hours=12))
        scans = _scans_in_transit(rng, origin, picked_up, last)
        action = "create_reship"
    elif scenario in ("failed_bad_address", "failed_customer_absent"):
        status = "delivery_failed"
        promised = created_at.date() - timedelta(days=rng.randint(0, 1))
        attempt = created_at - timedelta(hours=rng.randint(2, 20))
        address_valid = scenario == "failed_customer_absent"
        code = "CUSTOMER_ABSENT" if address_valid else "ADDRESS_ISSUE"
        scans = _scans_in_transit(rng, origin, picked_up, attempt - timedelta(hours=10)) + (
            ScanEvent(attempt - timedelta(hours=3), city, "OUT_FOR_DELIVERY"),
            ScanEvent(attempt, city, code),
        )
        if address_valid:
            action = "schedule_redelivery"
            expected_kwargs["redelivery_date"] = created_at.date() + timedelta(days=1)
        else:
            action = "request_address"
    elif scenario in ("damaged_low_value", "damaged_high_value", "delivered_not_received", "delivered_thanks"):
        status = "delivered"
        promised = created_at.date() - timedelta(days=rng.randint(0, 2))
        delivered_at = created_at - timedelta(hours=rng.randint(3, 40))
        scans = _scans_in_transit(rng, origin, picked_up, delivered_at - timedelta(hours=12)) + (
            ScanEvent(delivered_at - timedelta(hours=4), city, "OUT_FOR_DELIVERY"),
            ScanEvent(delivered_at, city, "DELIVERED"),
        )
        if scenario == "damaged_low_value":
            action = "issue_refund"
            expected_kwargs["amount_inr"] = value
        elif scenario == "damaged_high_value":
            value = round(rng.uniform(REFUND_AUTO_LIMIT_INR + 1, 40000), 2)
            action = "escalate"
            expected_kwargs["reason"] = "refund_above_limit"
        elif scenario == "delivered_not_received":
            action = "escalate"
            expected_kwargs["reason"] = "delivery_dispute"
        else:
            action = "close_ticket"
    elif scenario == "unknown_shipment":
        status = None
        promised = None
        scans = ()
        action = "escalate"
        expected_kwargs["reason"] = "shipment_not_found"
    else:  # pragma: no cover - guarded by SCENARIOS
        raise ValueError(f"unknown scenario {scenario!r}")

    shipment = None
    ticket_sid = sid
    if scenario == "unknown_shipment":
        ticket_sid = f"SHP-9{idx:04d}"  # never issued by the generator's numbering
    else:
        shipment = Shipment(
            shipment_id=sid,
            carrier=carrier,
            origin_hub=origin,
            destination_city=city,
            promised_date=promised,
            status=status,
            declared_value_inr=value,
            address_valid=address_valid,
            scans=scans,
        )

    template = rng.choice(MESSAGES[SCENARIO_MESSAGES[scenario]])
    message = template.format(sid=ticket_sid)
    if rng.random() < 0.15:
        message = _typo(message, rng)
    ticket = Ticket(f"TKT-{idx:05d}", ticket_sid, created_at, message)
    labelled = LabelledTicket(
        ticket=ticket,
        scenario=scenario,
        expected=Resolution(action=action, **expected_kwargs),
        expected_tools=SCENARIOS[scenario][1],
    )
    return shipment, labelled


def generate(
    n_tickets: int = 300,
    seed: int = 7,
    as_of: datetime = datetime(2026, 10, 28, 18, 0),
) -> tuple[dict[str, Shipment], list[LabelledTicket]]:
    """Return (shipments by id, labelled tickets). Deterministic for a given seed."""
    if isinstance(n_tickets, bool) or not isinstance(n_tickets, int) or n_tickets < 1:
        raise ValueError(f"n_tickets must be a positive integer, got {n_tickets!r}")
    rng = random.Random(seed)
    names = list(SCENARIOS)
    weights = [SCENARIOS[s][0] for s in names]
    shipments: dict[str, Shipment] = {}
    tickets: list[LabelledTicket] = []
    for idx in range(1, n_tickets + 1):
        scenario = rng.choices(names, weights=weights)[0]
        shipment, labelled = _build_case(scenario, idx, as_of, rng)
        if shipment is not None:
            shipments[shipment.shipment_id] = shipment
        tickets.append(labelled)
    return shipments, tickets


def main(argv: list[str] | None = None) -> int:
    """Write a sample dataset: ``triage-generate --out data --tickets 300 --seed 7``."""
    import argparse
    from pathlib import Path

    from .dataset import write_dataset

    parser = argparse.ArgumentParser(description="Generate synthetic shipments and labelled tickets.")
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--tickets", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    try:
        shipments, tickets = generate(n_tickets=args.tickets, seed=args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    write_dataset(shipments, tickets, args.out)
    print(f"wrote {len(shipments)} shipments and {len(tickets)} tickets to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
