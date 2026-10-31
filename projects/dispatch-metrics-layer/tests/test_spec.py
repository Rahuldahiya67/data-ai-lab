import copy

import pytest
import yaml

from dispatch_metrics_layer.expressions import check_fragment, qualify
from dispatch_metrics_layer.spec import SpecError, check_columns, parse_spec

from conftest import SPEC_PATH


@pytest.fixture
def raw():
    return yaml.safe_load(SPEC_PATH.read_text())


def problems_for(raw):
    with pytest.raises(SpecError) as info:
        parse_spec(raw)
    return "\n".join(info.value.problems)


def test_bundled_spec_is_valid(spec):
    assert spec.fact_table == "shipments"
    assert {"on_time_rate", "otif_rate", "cost_per_km"} <= set(spec.metrics)
    assert spec.metrics["on_time_rate"].measures == ["on_time_shipments", "delivered_shipments"]


def test_unknown_measure_reference(raw):
    raw["metrics"]["on_time_rate"]["denominator"] = "delivered"
    assert "metrics.on_time_rate.denominator: unknown measure 'delivered'" in problems_for(raw)


def test_all_problems_are_reported_at_once(raw):
    raw["metrics"]["on_time_rate"]["type"] = "average"
    raw["measures"]["freight_cost"]["agg"] = "median"
    raw["dimensions"]["carrier"]["table"] = "drivers"
    text = problems_for(raw)
    assert "metrics.on_time_rate.type" in text
    assert "measures.freight_cost.agg" in text
    assert "dimensions.carrier: unknown table 'drivers'" in text


def test_one_to_many_join_is_rejected(raw):
    raw["tables"]["shipments"]["joins"]["lanes"]["relationship"] = "one_to_many"
    assert "relationship must be 'many_to_one'" in problems_for(raw)


def test_bare_on_key_gets_a_clear_message(raw):
    raw["tables"]["shipments"]["joins"]["lanes"] = yaml.safe_load("{on: lane_id, relationship: many_to_one}")
    assert "YAML reads a bare 'on' key as the boolean true" in problems_for(raw)


def test_dimension_on_unjoined_table(raw):
    raw["tables"]["depots"] = {"primary_key": "depot_id"}
    raw["dimensions"]["depot"] = {"table": "depots", "column": "depot_name"}
    assert "table 'depots' is not joined to the fact table" in problems_for(raw)


def test_only_fact_table_may_join(raw):
    raw["tables"]["lanes"]["joins"] = {"carriers": {"key": "carrier_id", "relationship": "many_to_one"}}
    assert "only the fact table may declare joins" in problems_for(raw)


def test_sum_without_expr(raw):
    del raw["measures"]["freight_cost"]["expr"]
    assert "agg 'sum' needs an expr" in problems_for(raw)


def test_name_clash_and_bad_names(raw):
    raw["dimensions"]["shipment_count"] = {"table": "shipments", "column": "status"}
    raw["metrics"]["On Time"] = {"type": "simple", "measure": "shipments"}
    text = problems_for(raw)
    assert "both a dimension and a metric: ['shipment_count']" in text
    assert "must be lower_snake_case" in text


def test_unknown_keys_are_reported(raw):
    raw["metrics"]["on_time_rate"]["numerater"] = "x"
    assert "unknown keys ['numerater']" in problems_for(raw)


def test_check_columns_finds_missing_columns(spec, engine):
    schema = copy.deepcopy(engine.schema)
    schema["lanes"].discard("region")
    schema["shipments"].discard("promised_date")
    with pytest.raises(SpecError) as info:
        check_columns(spec, schema)
    text = "\n".join(info.value.problems)
    assert "dimensions.region: column lanes.region does not exist" in text
    assert "measures.on_time_shipments.filter: unknown column 'promised_date'" in text


@pytest.mark.parametrize(
    "fragment, expected",
    [
        ("status = 'delivered'; DROP TABLE shipments", "unexpected character ';'"),
        ("status = 'x' -- hidden", "comments are not allowed"),
        ("read_csv('secrets.csv') > 0", "function 'read_csv' is not allowed"),
        ("region = 'North'", "unknown column 'region'"),
        ("(status = 'delivered'", "unbalanced parentheses"),
    ],
)
def test_unsafe_fragments_are_rejected(fragment, expected):
    problems = check_fragment(fragment, {"status", "freight_cost"})
    assert any(expected in p for p in problems), problems


def test_safe_fragment_and_qualify():
    cols = {"status", "delivered_date", "promised_date"}
    fragment = "status = 'it''s' AND COALESCE(delivered_date, promised_date) <= promised_date"
    assert check_fragment(fragment, cols) == []
    assert qualify(fragment, cols, "f") == (
        "f.status = 'it''s' AND COALESCE ( f.delivered_date , f.promised_date ) <= f.promised_date"
    )
