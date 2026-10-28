"""Load the full synthetic stream and compare the dimension with the true history."""

from datetime import timedelta

import duckdb
import pytest

from rate_card_history_loader.generator import generate_stream
from rate_card_history_loader.loader import RateCardLoader
from rate_card_history_loader.lookup import current_rates, rate_at


def true_state_at(history, at):
    """Replay the true history up to ``at`` (inclusive) and return lane -> rate."""
    state = {}
    for event in history:
        if event.commit_ts > at:
            break
        if event.op == "d":
            state.pop(event.lane_id, None)
        else:
            state[event.lane_id] = (event.rate_per_kg, event.min_charge)
    return state


@pytest.fixture(scope="module")
def loaded():
    stream = generate_stream(seed=7)
    con = duckdb.connect()
    loader = RateCardLoader(con)
    reports = [loader.load_batch(batch, f"batch_{i}") for i, batch in enumerate(stream.batches, 1)]
    yield stream, con, reports
    con.close()


def test_current_rates_match_true_final_state(loaded):
    stream, con, _ = loaded
    truth = true_state_at(stream.history, stream.history[-1].commit_ts)
    loaded_state = {v.lane_id: (v.rate_per_kg, v.min_charge) for v in current_rates(con)}
    assert loaded_state == truth


def test_point_in_time_lookups_match_truth(loaded):
    stream, con, _ = loaded
    start, end = stream.history[0].commit_ts, stream.history[-1].commit_ts
    lanes = sorted({e.lane_id for e in stream.history})
    checked = 0
    for step in range(0, int((end - start).total_seconds() // 3600), 37):
        at = start + timedelta(hours=step)
        truth = true_state_at(stream.history, at)
        for lane in lanes:
            version = rate_at(con, lane, at)
            got = (version.rate_per_kg, version.min_charge) if version else None
            assert got == truth.get(lane), (lane, at)
            checked += 1
    assert checked > 1000


def test_versions_of_a_lane_never_overlap_or_leave_negative_spans(loaded):
    _, con, _ = loaded
    bad = con.execute(
        """
        SELECT count(*) FROM (
            SELECT valid_from, valid_to,
                   lead(valid_from) OVER (PARTITION BY lane_id ORDER BY valid_from) AS next_from
            FROM dim_rate_card
        ) WHERE valid_to <= valid_from OR valid_to > next_from
        """
    ).fetchone()[0]
    assert bad == 0
    assert con.execute(
        "SELECT count(*) FROM (SELECT lane_id FROM dim_rate_card WHERE is_current GROUP BY 1 HAVING count(*) > 1)"
    ).fetchone()[0] == 0


def test_final_replay_batch_is_a_no_op(loaded):
    _, _, reports = loaded
    replay = reports[-1]
    assert replay.versions_opened == 0 and replay.versions_closed == 0


def test_injected_bad_events_end_up_in_quarantine(loaded):
    _, con, reports = loaded
    reasons = [r for (r,) in con.execute("SELECT reason FROM quarantine").fetchall()]
    assert sum(r.rejected for r in reports) == len(reasons) > 0
    assert any("op must be one of" in r for r in reasons)
    assert any("rate_per_kg must be > 0" in r for r in reasons)
