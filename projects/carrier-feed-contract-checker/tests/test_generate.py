import pytest

from carrier_feed_contract_checker.generate import DEFECT_TYPES, GeneratorSpec, generate_feed, write_feed


def test_same_seed_gives_same_feed():
    a = generate_feed(GeneratorSpec(rows=200, seed=7))
    b = generate_feed(GeneratorSpec(rows=200, seed=7))
    assert a.equals(b)


def test_defect_rate_is_roughly_respected():
    df = generate_feed(GeneratorSpec(rows=5000, defect_rate=0.1, seed=1))
    share = (df["_defect"] != "").mean()
    assert 0.08 < share < 0.12
    assert set(df.loc[df["_defect"] != "", "_defect"]) <= set(DEFECT_TYPES)


def test_zero_defect_rate_gives_clean_feed():
    df = generate_feed(GeneratorSpec(rows=300, defect_rate=0.0))
    assert (df["_defect"] == "").all()
    assert df["tracking_id"].is_unique


@pytest.mark.parametrize("kwargs", [{"rows": 0}, {"defect_rate": 1.5}])
def test_invalid_spec_rejected(kwargs):
    with pytest.raises(ValueError):
        generate_feed(GeneratorSpec(**kwargs))


def test_write_feed_drops_labels_by_default(tmp_path):
    path = tmp_path / "feed.csv"
    write_feed(str(path), GeneratorSpec(rows=50))
    header = path.read_text().splitlines()[0]
    assert "_defect" not in header
