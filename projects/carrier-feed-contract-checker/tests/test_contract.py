import pytest

from carrier_feed_contract_checker.contract import ContractError, parse_contract


def base():
    return {
        "name": "t",
        "columns": {
            "a": {"type": "string", "required": True},
            "b": {"type": "float", "min": 0},
            "c": {"type": "datetime", "format": "%Y-%m-%d"},
        },
    }


def test_shipped_contract_loads(contract):
    assert contract.name == "carrier_shipment_status"
    assert contract.columns["tracking_id"].unique
    assert {r.name for r in contract.rules} == {"delivered_after_pickup", "delivered_needs_timestamp"}
    assert contract.feed.max_reject_rate == 0.10


def test_minimal_contract_parses():
    c = parse_contract(base())
    assert c.columns["b"].min == 0
    assert c.feed.min_rows == 0


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r: r["columns"]["a"].update(type="decimal"), "unsupported type"),
        (lambda r: r["columns"]["a"].update(min=1), "min/max only apply"),
        (lambda r: r["columns"]["a"].update(nullable=True), "unknown keys"),
        (lambda r: r.update(columns={}), "at least one column"),
        (lambda r: r.update(rules=[{"name": "x", "type": "compare", "left": "b", "op": "~", "right": "b"}]), "unsupported op"),
        (lambda r: r.update(rules=[{"name": "x", "type": "compare", "left": "b", "op": "<", "right": "zz"}]), "unknown column"),
        (lambda r: r.update(rules=[{"name": "x", "type": "required_if", "column": "a"}]), "missing"),
        (lambda r: r.update(rules=[{"name": "x", "type": "sql"}]), "unsupported type"),
        (lambda r: r.update(feed={"max_reject_rate": 2}), "between 0 and 1"),
    ],
)
def test_malformed_contracts_rejected(mutate, message):
    raw = base()
    mutate(raw)
    with pytest.raises(ContractError, match=message):
        parse_contract(raw)
