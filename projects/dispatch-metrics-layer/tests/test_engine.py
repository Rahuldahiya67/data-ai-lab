from datetime import date

import duckdb
import pytest

from dispatch_metrics_layer.compiler import Filter, MetricRequest, RequestError
from dispatch_metrics_layer.engine import DuplicateKeyError, FanOutError, MetricsEngine
from dispatch_metrics_layer.generator import GeneratorConfig, generate, write_csvs
from dispatch_metrics_layer.spec import parse_spec


def ref_rates(df):
    delivered = df[df.status == "delivered"]
    on_time = delivered[delivered.delivered_date <= delivered.promised_date]
    in_full = delivered[delivered.units_delivered >= delivered.units_ordered]
    otif = on_time[on_time.units_delivered >= on_time.units_ordered]
    live = df[df.status != "cancelled"]
    return {
        "shipment_count": len(df),
        "on_time_rate": len(on_time) / len(delivered),
        "in_full_rate": len(in_full) / len(delivered),
        "otif_rate": len(otif) / len(delivered),
        "fill_rate": delivered.units_delivered.sum() / delivered.units_ordered.sum(),
        "cancellation_rate": (df.status == "cancelled").mean(),
        "freight_cost": live.freight_cost.sum(),
        "cost_per_km": live.freight_cost.sum() / live.distance_km.sum(),
        "active_lanes": df.lane_id.nunique(),
        "on_time_rate_dispatched": len(on_time) / len(live),
        "on_time_rate_units": on_time.units_delivered.sum() / delivered.units_delivered.sum(),
    }


ALL_METRICS = [
    "shipment_count", "on_time_rate", "in_full_rate", "otif_rate", "fill_rate",
    "cancellation_rate", "freight_cost", "cost_per_km", "active_lanes",
    "on_time_rate_dispatched", "on_time_rate_units",
]


def test_totals_match_pandas(engine, frame):
    (row,) = engine.query(MetricRequest(ALL_METRICS)).records()
    expected = ref_rates(frame)
    for name in ALL_METRICS:
        assert row[name] == pytest.approx(expected[name]), name


def test_grouped_by_carrier_matches_pandas(engine, frame):
    rows = engine.query(MetricRequest(ALL_METRICS, ["carrier"])).records()
    assert [r["carrier"] for r in rows] == sorted(frame.carrier_name.unique())
    for row in rows:
        expected = ref_rates(frame[frame.carrier_name == row["carrier"]])
        for name in ALL_METRICS:
            assert row[name] == pytest.approx(expected[name]), (row["carrier"], name)


def test_ratio_is_ratio_of_sums_not_average_of_ratios(engine, frame):
    by_region = engine.query(MetricRequest(["on_time_rate", "shipment_count"], ["region"])).records()
    (total,) = engine.query(MetricRequest(["on_time_rate"])).records()
    average_of_ratios = sum(r["on_time_rate"] for r in by_region) / len(by_region)
    assert total["on_time_rate"] == pytest.approx(ref_rates(frame)["on_time_rate"])
    # Regions have different volumes, so the naive average is a different number.
    assert total["on_time_rate"] != pytest.approx(average_of_ratios, abs=1e-6)


def test_weekly_grain_and_date_range(engine, frame):
    start, end = date(2026, 7, 13), date(2026, 8, 10)
    rows = engine.query(
        MetricRequest(["shipment_count", "on_time_rate"], grain="week", start=start, end=end)
    ).records()
    assert [r["period"] for r in rows] == [date(2026, 7, 13), date(2026, 7, 20), date(2026, 7, 27), date(2026, 8, 3)]
    window = frame[(frame.dispatch_date >= str(start)) & (frame.dispatch_date < str(end))]
    assert sum(r["shipment_count"] for r in rows) == len(window)
    week = window[window.dispatch_date.dt.to_period("W-SUN").dt.start_time == "2026-07-20"]
    assert rows[1]["on_time_rate"] == pytest.approx(ref_rates(week)["on_time_rate"])


def test_filters_are_bound_as_parameters(engine, frame):
    request = MetricRequest(
        ["shipment_count", "cost_per_km"],
        ["mode"],
        filters=[Filter("region", "in", ["North", "West"]), Filter("mode", "!=", "PARCEL")],
    )
    result = engine.query(request)
    assert "'North'" not in result.query.sql and "?" in result.query.sql
    sub = frame[frame.region.isin(["North", "West"]) & (frame["mode"] != "PARCEL")]
    rows = result.records()
    assert [r["mode"] for r in rows] == ["FTL", "LTL"]
    for row in rows:
        assert row["cost_per_km"] == pytest.approx(ref_rates(sub[sub["mode"] == row["mode"]])["cost_per_km"])


def test_only_needed_joins_and_measures_are_compiled(engine):
    sql = engine.compile(MetricRequest(["on_time_rate"], ["status"])).sql
    assert "JOIN" not in sql
    assert "ms__freight_cost" not in sql
    sql = engine.compile(MetricRequest(["on_time_rate"], filters=[Filter("carrier", "=", "Metro Parcel")])).sql
    assert 'JOIN "carriers"' in sql and 'JOIN "lanes"' not in sql


def test_sql_keywords_work_as_spec_names(raw_spec, data_dir, engine):
    # 'order' and 'group' are valid lower_snake_case names and SQL keywords.
    raw_spec["dimensions"]["order"] = raw_spec["dimensions"].pop("status")
    raw_spec["metrics"]["group"] = raw_spec["metrics"].pop("cancellation_rate")
    keyword_engine = MetricsEngine.from_csv_dir(parse_spec(raw_spec), data_dir)
    got = keyword_engine.query(
        MetricRequest(["group", "shipment_count"], ["order"], filters=[Filter("order", "!=", "x")])
    ).records()
    want = engine.query(
        MetricRequest(["cancellation_rate", "shipment_count"], ["status"], filters=[Filter("status", "!=", "x")])
    ).records()
    assert [(r["order"], r["group"], r["shipment_count"]) for r in got] == [
        (r["status"], r["cancellation_rate"], r["shipment_count"]) for r in want
    ]


def test_shared_column_names_do_not_clash(engine, frame):
    # lanes.distance_km and shipments.distance_km both exist; planned_km must use the fact column.
    (row,) = engine.query(MetricRequest(["cost_per_km"], filters=[Filter("region", "=", "South")])).records()
    assert row["cost_per_km"] == pytest.approx(ref_rates(frame[frame.region == "South"])["cost_per_km"])


def test_empty_selection_returns_no_groups(engine):
    rows = engine.query(MetricRequest(["shipment_count"], ["carrier"], filters=[Filter("region", "=", "Mars")])).rows
    assert rows == []


@pytest.mark.parametrize(
    "request_kwargs, message",
    [
        ({"metrics": []}, "at least one metric"),
        ({"metrics": ["on_time"]}, "unknown metric 'on_time'"),
        ({"metrics": ["on_time_rate"], "dimensions": ["driver"]}, "unknown dimension 'driver'"),
        ({"metrics": ["on_time_rate"], "grain": "quarter"}, "grain must be one of"),
        ({"metrics": ["on_time_rate"], "start": date(2026, 8, 1), "end": date(2026, 7, 1)}, "must be before end"),
        ({"metrics": ["on_time_rate"], "filters": [Filter("region", "in", [])]}, "non-empty list"),
        ({"metrics": ["on_time_rate"], "filters": [Filter("region", "like", "N%")]}, "filter operator"),
        ({"metrics": ["on_time_rate"], "filters": [Filter("region", "=", ["North"])]}, "single value"),
        ({"metrics": ["on_time_rate"], "filters": [Filter("region", ">", None)]}, "cannot compare with None"),
    ],
)
def test_bad_requests(engine, request_kwargs, message):
    with pytest.raises(RequestError, match=message):
        engine.query(MetricRequest(**request_kwargs))


def _write(tmp_path, mutate):
    tables = generate(GeneratorConfig(n_shipments=200, seed=2))
    mutate(tables)
    write_csvs(tables, tmp_path)
    return tmp_path


def test_duplicate_dimension_key_is_a_fan_out_error(spec, tmp_path):
    def dup_lane(tables):
        tables["lanes"].append(dict(tables["lanes"][0], region="Duplicate"))

    with pytest.raises(FanOutError, match=r"lanes\.lane_id is not unique \('L01' x2\)"):
        MetricsEngine.from_csv_dir(spec, _write(tmp_path, dup_lane))


def test_duplicate_fact_rows_are_rejected(spec, tmp_path):
    # A re-delivered extract repeats shipment rows; summing them would inflate every metric.
    def redeliver(tables):
        tables["shipments"].extend(dict(row) for row in tables["shipments"][:4])

    with pytest.raises(
        DuplicateKeyError,
        match=r"shipments\.shipment_id is not unique: 4 key\(s\) repeated, 4 extra row\(s\) \('S00001' x2",
    ):
        MetricsEngine.from_csv_dir(spec, _write(tmp_path, redeliver))


def test_null_fact_key_is_rejected(spec, tmp_path):
    def blank_key(tables):
        tables["shipments"][7]["shipment_id"] = None

    with pytest.raises(DuplicateKeyError, match=r"shipments\.shipment_id is NULL in 1 row"):
        MetricsEngine.from_csv_dir(spec, _write(tmp_path, blank_key))


def test_orphan_keys_are_counted_and_kept_in_totals(spec, tmp_path):
    def orphan(tables):
        for row in tables["shipments"][:3]:
            row["carrier_id"] = "C99"

    engine = MetricsEngine.from_csv_dir(spec, _write(tmp_path, orphan))
    assert engine.orphans == {"carriers": 3, "lanes": 0}
    rows = engine.query(MetricRequest(["shipment_count"], ["carrier"])).records()
    assert rows[-1] == {"carrier": None, "shipment_count": 3}
    assert sum(r["shipment_count"] for r in rows) == 200


def test_negative_filters_keep_rows_with_a_null_dimension(spec, tmp_path):
    # Orphan carrier keys give a NULL carrier name. "= X" and "!= X" must still
    # partition the data, and so must "in" and "not_in".
    def orphan(tables):
        for row in tables["shipments"][:5]:
            row["carrier_id"] = "C99"

    engine = MetricsEngine.from_csv_dir(spec, _write(tmp_path, orphan))

    def count(*filters):
        (row,) = engine.query(MetricRequest(["shipment_count"], filters=list(filters))).records()
        return row["shipment_count"]

    total = count()
    names = [r["carrier"] for r in engine.query(MetricRequest(["shipment_count"], ["carrier"])).records()]
    first, second = names[0], names[1]
    assert count(Filter("carrier", "=", first)) + count(Filter("carrier", "!=", first)) == total
    pair = [first, second]
    assert count(Filter("carrier", "in", pair)) + count(Filter("carrier", "not_in", pair)) == total
    # The orphan rows are the ones a plain SQL "!=" would have lost.
    assert count(Filter("carrier", "not_in", names[:-1])) == 5


def test_none_filters_select_the_rows_grouped_under_none(spec, tmp_path):
    # Grouped output shows orphan carrier keys as carrier=None. Filtering on None
    # must return exactly those rows; plain SQL "= NULL" / "IN (NULL)" would match
    # nothing, and "NOT IN (..., NULL)" would drop every row with a carrier.
    def orphan(tables):
        for row in tables["shipments"][:5]:
            row["carrier_id"] = "C99"

    engine = MetricsEngine.from_csv_dir(spec, _write(tmp_path, orphan))

    def count(*filters):
        (row,) = engine.query(MetricRequest(["shipment_count"], filters=list(filters))).records()
        return row["shipment_count"]

    total = count()
    first = engine.query(MetricRequest(["shipment_count"], ["carrier"])).records()[0]["carrier"]
    assert count(Filter("carrier", "=", None)) == 5
    assert count(Filter("carrier", "!=", None)) == total - 5
    assert count(Filter("carrier", "in", [None])) == 5
    assert count(Filter("carrier", "not_in", [None])) == total - 5
    with_first = count(Filter("carrier", "=", first))
    assert count(Filter("carrier", "in", [first, None])) == with_first + 5
    assert count(Filter("carrier", "not_in", [first, None])) == total - with_first - 5


def test_missing_data_file(spec, tmp_path):
    write_csvs({"carriers": generate(GeneratorConfig(n_shipments=5))["carriers"]}, tmp_path)
    with pytest.raises(FileNotFoundError, match="missing data file for table 'shipments'"):
        MetricsEngine.from_csv_dir(spec, tmp_path)


def test_engine_accepts_existing_connection(spec, data_dir):
    conn = duckdb.connect()
    for name in ("shipments", "carriers", "lanes"):
        conn.execute(f"CREATE TABLE {name} AS SELECT * FROM read_csv(?)", [str(data_dir / f"{name}.csv")])
    engine = MetricsEngine(spec, conn)
    assert engine.query(MetricRequest(["shipment_count"])).rows == [(1500,)]
