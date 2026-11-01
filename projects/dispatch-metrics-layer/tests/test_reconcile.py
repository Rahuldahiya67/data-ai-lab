import pytest

from dispatch_metrics_layer.compiler import Filter, MetricRequest, RequestError
from dispatch_metrics_layer.reconcile import AggregationCheck, reconcile

DEFS = ["on_time_rate", "on_time_rate_dispatched", "on_time_rate_units"]


def on_time_defs(df):
    delivered = df[df.status == "delivered"]
    on_time = delivered[delivered.delivered_date <= delivered.promised_date]
    live = df[df.status != "cancelled"]
    return {
        "on_time_rate": len(on_time) / len(delivered),
        "on_time_rate_dispatched": len(on_time) / len(live),
        "on_time_rate_units": on_time.units_delivered.sum() / delivered.units_delivered.sum(),
    }


def test_overall_matches_pandas(engine, frame):
    report = reconcile(engine, DEFS)
    expected = on_time_defs(frame)
    for name in DEFS:
        assert report.overall[name] == pytest.approx(expected[name]), name
    assert report.baseline == "on_time_rate"
    assert report.groups == [] and report.aggregation == []


def test_dispatched_denominator_is_never_above_delivered_one(engine):
    # Same numerator, a denominator that also counts in-transit shipments.
    report = reconcile(engine, DEFS, by="carrier")
    assert report.gap_to_baseline("on_time_rate_dispatched") < 0
    for g in report.groups:
        assert g.values["on_time_rate_dispatched"] <= g.values["on_time_rate"]


def test_groups_match_pandas(engine, frame):
    report = reconcile(engine, DEFS, by="region")
    assert [g.group for g in report.groups] == sorted(frame.region.unique())
    for g in report.groups:
        expected = on_time_defs(frame[frame.region == g.group])
        for name in DEFS:
            assert g.values[name] == pytest.approx(expected[name]), (g.group, name)


def test_aggregation_check_is_unweighted_mean_of_group_rates(engine, frame):
    report = reconcile(engine, ["on_time_rate", "shipment_count"], by="carrier")
    # shipment_count is a simple metric: no rate to average, so no check.
    (check,) = report.aggregation
    per_carrier = [on_time_defs(frame[frame.carrier_name == c])["on_time_rate"] for c in frame.carrier_name.unique()]
    assert check.metric == "on_time_rate"
    assert check.groups_used == len(per_carrier)
    assert check.average_of_groups == pytest.approx(sum(per_carrier) / len(per_carrier))
    assert check.pooled == pytest.approx(on_time_defs(frame)["on_time_rate"])
    assert check.gap == pytest.approx(check.average_of_groups - check.pooled)


def test_spread_and_gap(engine):
    report = reconcile(engine, DEFS)
    values = list(report.overall.values())
    assert report.spread == pytest.approx(max(values) - min(values))
    assert report.gap_to_baseline("on_time_rate") == 0


def test_filters_and_dates_carry_through(engine, frame):
    base = MetricRequest(["shipment_count"], filters=[Filter("mode", "=", "FTL")])
    report = reconcile(engine, DEFS, base=base)
    expected = on_time_defs(frame[frame["mode"] == "FTL"])
    assert report.overall["on_time_rate_units"] == pytest.approx(expected["on_time_rate_units"])


def test_no_rows_gives_empty_values(engine):
    base = MetricRequest(["shipment_count"], filters=[Filter("carrier", "=", "No Such Carrier")])
    report = reconcile(engine, DEFS, by="carrier", base=base)
    assert report.overall == {name: None for name in DEFS}
    assert report.groups == []
    assert report.spread is None
    assert all(c.average_of_groups is None and c.gap is None for c in report.aggregation)


def test_needs_two_metrics(engine):
    with pytest.raises(RequestError, match="at least two metrics"):
        reconcile(engine, ["on_time_rate"])


def test_unknown_metric_or_dimension_is_a_request_error(engine):
    with pytest.raises(RequestError, match="unknown metric"):
        reconcile(engine, ["on_time_rate", "nope"])
    with pytest.raises(RequestError, match="unknown dimension"):
        reconcile(engine, DEFS, by="planet")


def test_gap_is_none_when_a_side_is_missing():
    check = AggregationCheck("on_time_rate", pooled=None, average_of_groups=0.9, groups_used=1)
    assert check.gap is None
