from datetime import datetime

from rate_card_history_loader.generator import generate_history
from rate_card_history_loader.invoices import generate_invoices


def test_invoices_are_deterministic_and_cover_each_issue_type():
    history = generate_history(seed=7)
    a, b = generate_invoices(history, seed=5), generate_invoices(history, seed=5)
    assert a == b and len(a) == 400
    issues = {inv["expected_issue"] for inv in a}
    assert issues == {"", "stale_rate", "no_contract"}
    assert len({inv["invoice_id"] for inv in a}) == len(a)


def test_invoices_ship_inside_the_history_and_bill_positive_amounts():
    history = generate_history(seed=7)
    start, end = history[0].commit_ts, history[-1].commit_ts
    for inv in generate_invoices(history, n_invoices=100, seed=2):
        assert start <= datetime.fromisoformat(inv["ship_ts"]) <= end
        assert inv["weight_kg"] > 0 and inv["billed_amount"] > 0


def test_empty_history_gives_no_invoices():
    assert generate_invoices([]) == []
