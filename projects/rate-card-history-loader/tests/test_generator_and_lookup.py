from datetime import timedelta

import pytest

from rate_card_history_loader.generator import generate_history, generate_stream
from rate_card_history_loader.lookup import RateVersion, rate_at

from conftest import T0, ev


def test_generator_is_deterministic_and_ordered():
    a, b = generate_history(seed=3), generate_history(seed=3)
    assert a == b
    assert all(x.lsn < y.lsn for x, y in zip(a, a[1:]))
    assert all(x.commit_ts <= y.commit_ts for x, y in zip(a, a[1:]))


def test_delivery_contains_duplicates_and_a_replay():
    stream = generate_stream(seed=7)
    delivered = [r["lsn"] for batch in stream.batches for r in batch]
    assert len(delivered) > len(set(delivered))
    assert stream.batches[-1] == stream.batches[1]


def test_rate_at_uses_inclusive_start_and_exclusive_end(loader):
    loader.load_batch([ev(1, op="c"), ev(2, minutes=30, rate=12.0)], "b1")
    lane = "L001-DEL-BOM"
    assert rate_at(loader.con, lane, T0 - timedelta(seconds=1)) is None
    assert rate_at(loader.con, lane, T0).rate_per_kg == 10.0
    assert rate_at(loader.con, lane, T0 + timedelta(minutes=29, seconds=59)).rate_per_kg == 10.0
    assert rate_at(loader.con, lane, T0 + timedelta(minutes=30)).rate_per_kg == 12.0


def test_rate_at_returns_none_after_withdrawal(loader):
    loader.load_batch([ev(1, op="c"), ev(2, op="d", minutes=10)], "b1")
    assert rate_at(loader.con, "L001-DEL-BOM", T0 + timedelta(minutes=10)) is None


def test_charge_applies_minimum_and_rejects_bad_weight():
    version = RateVersion("L1", "Gati Express", 10.0, 300.0, "INR", T0, T0, 1)
    assert version.charge(12) == 300.0
    assert version.charge(45.5) == 455.0
    with pytest.raises(ValueError):
        version.charge(0)
