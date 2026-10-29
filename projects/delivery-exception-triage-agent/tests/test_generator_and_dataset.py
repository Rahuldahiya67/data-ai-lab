import json
from collections import Counter

import pytest

from delivery_exception_triage_agent.dataset import (
    DatasetError,
    load_shipments,
    load_tickets,
    parse_shipment,
    write_dataset,
)
from delivery_exception_triage_agent.generator import SCENARIOS, generate
from delivery_exception_triage_agent.policy_rules import REFUND_AUTO_LIMIT_INR, STUCK_AFTER_HOURS


def test_generate_is_deterministic():
    a = generate(n_tickets=50, seed=3)
    b = generate(n_tickets=50, seed=3)
    assert a == b
    assert generate(n_tickets=50, seed=4) != a


def test_every_scenario_appears_in_a_large_sample():
    _, tickets = generate(n_tickets=400, seed=1)
    assert set(Counter(t.scenario for t in tickets)) == set(SCENARIOS)


def test_labels_follow_the_policy_thresholds():
    shipments, tickets = generate(n_tickets=400, seed=2)
    for t in tickets:
        s = shipments.get(t.ticket.shipment_id)
        if t.scenario == "unknown_shipment":
            assert s is None
            continue
        if t.scenario == "damaged_low_value":
            assert s.declared_value_inr <= REFUND_AUTO_LIMIT_INR
            assert t.expected.amount_inr == s.declared_value_inr
        if t.scenario == "damaged_high_value":
            assert s.declared_value_inr > REFUND_AUTO_LIMIT_INR
        if t.scenario == "late_stuck":
            gap = t.ticket.created_at - s.scans[-1].at
            assert gap.total_seconds() / 3600 > STUCK_AFTER_HOURS
        if t.scenario == "late_moving":
            gap = t.ticket.created_at - s.scans[-1].at
            assert gap.total_seconds() / 3600 < STUCK_AFTER_HOURS
            assert s.promised_date < t.ticket.created_at.date()
        if t.scenario == "failed_customer_absent":
            assert t.expected.redelivery_date > t.ticket.created_at.date()


def test_scans_are_in_time_order_and_before_the_ticket():
    shipments, tickets = generate(n_tickets=200, seed=5)
    by_id = {t.ticket.shipment_id: t for t in tickets}
    for s in shipments.values():
        times = [e.at for e in s.scans]
        assert times == sorted(times)
        assert times[-1] <= by_id[s.shipment_id].ticket.created_at


@pytest.mark.parametrize("bad", [0, -5, True, 2.5])
def test_generate_rejects_bad_ticket_counts(bad):
    with pytest.raises(ValueError):
        generate(n_tickets=bad)


def test_round_trip_through_files(tmp_path):
    shipments, tickets = generate(n_tickets=60, seed=9)
    write_dataset(shipments, tickets, tmp_path)
    assert load_shipments(tmp_path / "shipments.json") == shipments
    assert load_tickets(tmp_path / "tickets.jsonl") == tickets


def _good_shipment():
    shipments, _ = generate(n_tickets=5, seed=1)
    from delivery_exception_triage_agent.dataset import _shipment_to_dict

    return _shipment_to_dict(next(iter(shipments.values())))


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r: r.pop("status"), "missing field 'status'"),
        (lambda r: r.update(status="lost"), "unknown status"),
        (lambda r: r.update(declared_value_inr=-1), "positive finite"),
        (lambda r: r.update(declared_value_inr="500"), "must be a number"),
        (lambda r: r.update(promised_date="28/10/2026"), "not an ISO date"),
        (lambda r: r.update(address_valid="yes"), "must be bool"),
        (lambda r: r["scans"].reverse(), "not in time order"),
    ],
)
def test_parse_shipment_rejects_bad_records(mutate, message):
    rec = _good_shipment()
    mutate(rec)
    with pytest.raises(DatasetError, match=message):
        parse_shipment(rec)


def test_load_tickets_names_the_bad_line(tmp_path):
    _, tickets = generate(n_tickets=3, seed=1)
    path = tmp_path / "tickets.jsonl"
    write_dataset({}, tickets, tmp_path)
    lines = path.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["expected"]["action"] = "refund_everything"
    lines[1] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(DatasetError, match="unknown expected action"):
        load_tickets(path)


def test_load_tickets_rejects_duplicate_ids(tmp_path):
    _, tickets = generate(n_tickets=2, seed=1)
    write_dataset({}, tickets + tickets[:1], tmp_path)
    with pytest.raises(DatasetError, match="duplicate ticket_id"):
        load_tickets(tmp_path / "tickets.jsonl")
