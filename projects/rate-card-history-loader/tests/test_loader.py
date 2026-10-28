import random
from datetime import timedelta

from rate_card_history_loader.loader import OPEN_END
from rate_card_history_loader.lookup import current_rates, lane_history

from conftest import T0, ev


def versions(loader, lane="L001-DEL-BOM"):
    return [(v.rate_per_kg, v.valid_from, v.valid_to) for v in lane_history(loader.con, lane)]


def ts(minutes):
    return T0 + timedelta(minutes=minutes)


def test_create_opens_one_current_version(loader):
    report = loader.load_batch([ev(1, op="c")], "b1")
    assert versions(loader) == [(10.0, T0, OPEN_END)]
    assert report.versions_opened == 1 and report.versions_closed == 0
    assert loader.watermark == 1


def test_update_closes_previous_version_at_change_time(loader):
    loader.load_batch([ev(1, op="c")], "b1")
    report = loader.load_batch([ev(2, minutes=30, rate=12.0)], "b2")
    assert versions(loader) == [(10.0, T0, ts(30)), (12.0, ts(30), OPEN_END)]
    assert report.versions_opened == 1 and report.versions_closed == 1


def test_update_without_value_change_does_not_create_a_version(loader):
    loader.load_batch([ev(1, op="c")], "b1")
    report = loader.load_batch([ev(2, minutes=30)], "b2")
    assert versions(loader) == [(10.0, T0, OPEN_END)]
    assert report.no_op == 1 and report.versions_opened == 0
    assert loader.watermark == 2


def test_delete_closes_lane_and_recreate_reopens_it(loader):
    loader.load_batch([ev(1, op="c"), ev(2, op="d", minutes=60)], "b1")
    assert versions(loader) == [(10.0, T0, ts(60))]
    assert current_rates(loader.con) == []
    loader.load_batch([ev(3, op="c", minutes=120, rate=11.0)], "b2")
    assert versions(loader) == [(10.0, T0, ts(60)), (11.0, ts(120), OPEN_END)]


def test_delete_of_unknown_lane_is_a_no_op(loader):
    report = loader.load_batch([ev(1, op="d")], "b1")
    assert versions(loader) == []
    assert report.no_op == 1


def test_several_changes_to_one_lane_in_a_shuffled_batch_form_a_chain(loader):
    batch = [ev(1, op="c"), ev(2, minutes=10, rate=11.0), ev(3, minutes=20, rate=11.0), ev(4, minutes=30, rate=9.5)]
    random.Random(3).shuffle(batch)
    report = loader.load_batch(batch, "b1")
    assert versions(loader) == [
        (10.0, T0, ts(10)),
        (11.0, ts(10), ts(30)),
        (9.5, ts(30), OPEN_END),
    ]
    assert report.no_op == 1  # lsn 3 repeats lsn 2's values
    assert report.versions_opened == 3 and report.versions_closed == 2


def test_replayed_batch_changes_nothing(loader):
    batch = [ev(1, op="c"), ev(2, minutes=10, rate=11.0)]
    loader.load_batch(batch, "b1")
    before = loader.con.execute("SELECT * FROM dim_rate_card ORDER BY source_lsn").fetchall()
    report = loader.load_batch(batch, "b1-replay")
    after = loader.con.execute("SELECT * FROM dim_rate_card ORDER BY source_lsn").fetchall()
    assert before == after
    assert report.below_watermark == 2 and report.versions_opened == 0


def test_rejected_events_are_quarantined_and_rest_of_batch_loads(loader):
    report = loader.load_batch([ev(1, op="c"), ev(2, minutes=5, rate=-1.0)], "b1")
    assert report.rejected == 1
    assert versions(loader) == [(10.0, T0, OPEN_END)]
    rows = loader.con.execute("SELECT batch_id, lsn, reason FROM quarantine").fetchall()
    assert rows == [("b1", 2, "rate_per_kg must be > 0, got -1")]


def test_duplicates_are_counted_in_report_and_log(loader):
    report = loader.load_batch([ev(1, op="c"), ev(1, op="c"), ev(2, minutes=5, rate=12.0)], "b1")
    assert report.duplicates == 1 and report.received == 3
    logged = loader.con.execute("SELECT batch_id, duplicates, versions_opened FROM load_log").fetchall()
    assert logged == [("b1", 1, 2)]


def test_lanes_are_independent(loader):
    loader.load_batch(
        [ev(1, op="c"), ev(2, op="c", lane="L002-BLR-MAA", rate=7.0), ev(3, minutes=5, rate=12.0)], "b1"
    )
    assert versions(loader, "L002-BLR-MAA") == [(7.0, T0, OPEN_END)]
    assert len(versions(loader)) == 2


def test_change_committed_before_the_open_version_is_quarantined(loader):
    loader.load_batch([ev(1, op="c", minutes=60)], "b1")
    report = loader.load_batch([ev(2, minutes=30, rate=12.0), ev(3, op="c", lane="L002-BLR-MAA")], "b2")
    assert versions(loader) == [(10.0, ts(60), OPEN_END)]
    assert versions(loader, "L002-BLR-MAA") == [(10.0, T0, OPEN_END)]
    assert report.rejected == 1
    (reason,) = loader.con.execute("SELECT reason FROM quarantine WHERE lsn = 2").fetchone()
    assert "earlier than the lane's previous change" in reason


def test_change_committed_before_a_withdrawal_is_quarantined(loader):
    loader.load_batch([ev(1, op="c"), ev(2, op="d", minutes=60)], "b1")
    report = loader.load_batch([ev(3, op="c", minutes=30, rate=12.0)], "b2")
    assert report.rejected == 1
    assert versions(loader) == [(10.0, T0, ts(60))]


def test_out_of_order_commit_inside_one_batch_is_quarantined(loader):
    batch = [ev(1, op="c"), ev(2, minutes=30, rate=11.0), ev(3, minutes=20, rate=12.0), ev(4, minutes=40, rate=13.0)]
    report = loader.load_batch(batch, "b1")
    assert versions(loader) == [(10.0, T0, ts(30)), (11.0, ts(30), ts(40)), (13.0, ts(40), OPEN_END)]
    assert report.rejected == 1


def test_changes_at_the_same_instant_keep_only_the_last(loader):
    report = loader.load_batch([ev(1, op="c"), ev(2, rate=11.0), ev(3, minutes=10, rate=12.0)], "b1")
    assert versions(loader) == [(11.0, T0, ts(10)), (12.0, ts(10), OPEN_END)]
    assert report.no_op == 1 and report.rejected == 0


def test_change_at_the_same_instant_as_open_version_replaces_it(loader):
    loader.load_batch([ev(1, op="c"), ev(2, minutes=10, rate=11.0)], "b1")
    report = loader.load_batch([ev(3, minutes=10, rate=12.0)], "b2")
    assert versions(loader) == [(10.0, T0, ts(10)), (12.0, ts(10), OPEN_END)]
    assert report.versions_replaced == 1 and report.versions_closed == 0
    zero_length = loader.con.execute("SELECT count(*) FROM dim_rate_card WHERE valid_to <= valid_from")
    assert zero_length.fetchone()[0] == 0


def test_late_event_below_the_watermark_is_quarantined_not_dropped(loader):
    loader.load_batch([ev(1, op="c"), ev(3, minutes=20, rate=11.0)], "b1")
    report = loader.load_batch([ev(2, minutes=10, rate=15.0)], "b2")
    assert report.below_watermark == 0 and report.rejected == 1
    (reason,) = loader.con.execute("SELECT reason FROM quarantine WHERE lsn = 2").fetchone()
    assert reason == "late event: lsn 2 is below the watermark 3 and was never applied"
    assert versions(loader) == [(10.0, T0, ts(20)), (11.0, ts(20), OPEN_END)]


def test_redelivery_with_a_different_payload_in_a_later_batch_is_quarantined(loader):
    loader.load_batch([ev(1, op="c"), ev(2, minutes=10, rate=11.0)], "b1")
    report = loader.load_batch([ev(2, minutes=10, rate=11.5)], "b2")
    assert report.rejected == 1 and report.below_watermark == 0
    (reason,) = loader.con.execute("SELECT reason FROM quarantine WHERE batch_id = 'b2'").fetchone()
    assert reason == "lsn 2 was already applied with a different payload"


def test_no_op_and_superseded_events_count_as_applied_on_redelivery(loader):
    batch = [ev(1, op="c"), ev(2, rate=11.0), ev(3, minutes=10)]
    loader.load_batch(batch, "b1")
    report = loader.load_batch(batch, "b1-replay")
    assert report.below_watermark == 3 and report.rejected == 0


def test_redelivery_of_a_quarantined_event_is_quarantined_again(loader):
    loader.load_batch([ev(1, op="c", minutes=60), ev(2, minutes=30, rate=12.0), ev(3, minutes=90)], "b1")
    report = loader.load_batch([ev(2, minutes=30, rate=12.0)], "b2")
    assert report.rejected == 1
    reasons = [r for (r,) in loader.con.execute("SELECT reason FROM quarantine ORDER BY batch_id").fetchall()]
    assert reasons[1] == "lsn 2 was quarantined in an earlier batch"
