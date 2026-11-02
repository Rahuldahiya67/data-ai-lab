"""Derived metrics: expressions over other metrics, checked against pandas."""

import pytest

from dispatch_metrics_layer.compiler import Filter, MetricRequest
from dispatch_metrics_layer.engine import MetricsEngine
from dispatch_metrics_layer.spec import SpecError, parse_spec


def reference(df):
    delivered = df[df.status == "delivered"]
    on_time = delivered.delivered_date <= delivered.promised_date
    in_full = delivered.units_delivered >= delivered.units_ordered
    n = len(delivered)
    return {
        "late_rate": (~on_time).sum() / n,
        "on_time_not_in_full_rate": (on_time & ~in_full).sum() / n,
        "otif_gap_to_target": (on_time & in_full).sum() / n - 0.95,
    }


DERIVED = ["late_rate", "on_time_not_in_full_rate", "otif_gap_to_target"]


def test_bundled_derived_metrics_record_their_dependencies(spec):
    assert spec.metrics["on_time_not_in_full_rate"].depends_on == ("on_time_rate", "otif_rate")
    assert spec.measures_for("late_rate") == ["on_time_shipments", "delivered_shipments"]


def test_derived_metrics_match_pandas_by_carrier(engine, frame):
    rows = engine.query(MetricRequest(DERIVED, ["carrier"])).records()
    assert len(rows) == frame.carrier_name.nunique()
    for row in rows:
        expected = reference(frame[frame.carrier_name == row["carrier"]])
        for name in DERIVED:
            assert row[name] == pytest.approx(expected[name]), (row["carrier"], name)


def test_derived_metrics_match_pandas_by_week(engine, frame):
    rows = engine.query(MetricRequest(["late_rate"], grain="week")).records()
    weeks = frame.dispatch_date.dt.to_period("W-SUN").dt.start_time.dt.date
    assert [r["period"] for r in rows] == sorted(weeks.unique())
    for row in rows:
        expected = reference(frame[weeks == row["period"]])["late_rate"]
        assert row["late_rate"] == pytest.approx(expected)


def test_derived_metric_compiles_only_the_measures_it_needs(engine):
    sql = engine.compile(MetricRequest(["late_rate"])).sql
    assert '"ms__on_time_shipments"' in sql and '"ms__delivered_shipments"' in sql
    assert "ms__in_full_shipments" not in sql and "ms__freight_cost" not in sql


def test_derived_of_derived_and_division_by_zero(raw_spec, data_dir):
    raw_spec["metrics"]["late_to_cancelled"] = {
        "type": "derived",
        "expr": "late_rate / cancellation_rate",
    }
    raw_spec["metrics"]["late_points"] = {"type": "derived", "expr": "ROUND(late_rate * 100, 1)"}
    engine = MetricsEngine.from_csv_dir(parse_spec(raw_spec), data_dir)
    (total,) = engine.query(
        MetricRequest(["late_rate", "late_points", "cancellation_rate", "late_to_cancelled"])
    ).records()
    assert total["late_points"] == round(total["late_rate"] * 100, 1)
    assert total["late_to_cancelled"] == pytest.approx(total["late_rate"] / total["cancellation_rate"])
    # No cancelled shipments among delivered ones: the denominator is 0, the result is NULL.
    (delivered_only,) = engine.query(
        MetricRequest(["late_to_cancelled"], filters=[Filter("status", "=", "delivered")])
    ).records()
    assert delivered_only["late_to_cancelled"] is None


def problems_for(raw):
    with pytest.raises(SpecError) as info:
        parse_spec(raw)
    return "\n".join(info.value.problems)


def test_derived_metric_validation(raw_spec):
    m = raw_spec["metrics"]
    m["bad_ref"] = {"type": "derived", "expr": "on_time_rate - on_time_shipments"}
    m["no_ref"] = {"type": "derived", "expr": "0.5"}
    m["no_expr"] = {"type": "derived"}
    m["bad_func"] = {"type": "derived", "expr": "sqrt(on_time_rate)"}
    m["on_time_rate"]["expr"] = "1"
    text = problems_for(raw_spec)
    assert "metrics.bad_ref.expr: unknown metric 'on_time_shipments'" in text
    assert "metrics.no_ref.expr: a derived metric must reference at least one metric" in text
    assert "metrics.no_expr.expr is required for a derived metric" in text
    assert "metrics.bad_func.expr: function 'sqrt' is not allowed" in text
    assert "metrics.on_time_rate: unknown keys ['expr']" in text


def test_derived_metric_cycles_are_rejected(raw_spec):
    m = raw_spec["metrics"]
    m["a_rate"] = {"type": "derived", "expr": "b_rate + 1"}
    m["b_rate"] = {"type": "derived", "expr": "a_rate - 1"}
    m["self_rate"] = {"type": "derived", "expr": "self_rate * 2"}
    text = problems_for(raw_spec)
    assert "derived metrics form a cycle: a_rate -> b_rate -> a_rate" in text
    assert "derived metrics form a cycle: self_rate -> self_rate" in text
