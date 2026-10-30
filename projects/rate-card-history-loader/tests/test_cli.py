import pytest

from rate_card_history_loader.cli import main


@pytest.fixture
def workspace(tmp_path):
    data = tmp_path / "data"
    assert main(["generate", "--out", str(data)]) == 0
    db = str(tmp_path / "rates.duckdb")
    return data, db


def test_generate_load_and_query(workspace, capsys):
    data, db = workspace
    assert sorted(p.name for p in (data / "cdc_batches").iterdir())[0] == "batch_0001.jsonl"
    assert (data / "invoices.csv").exists()
    capsys.readouterr()

    assert main(["load", str(data / "cdc_batches"), "--db", db]) == 0
    out = capsys.readouterr().out
    assert "batch_0006" in out and "current lanes" in out

    assert main(["history", "L001-CCU-BLR", "--db", db]) == 0
    assert "open" in capsys.readouterr().out
    assert main(["rate-at", "L001-CCU-BLR", "2026-07-15T10:00:00", "--db", db]) == 0
    assert "/kg" in capsys.readouterr().out
    assert main(["rate-at", "L001-CCU-BLR", "2020-01-01T00:00:00", "--db", db]) == 1


def test_loading_the_same_directory_twice_changes_nothing(workspace, capsys):
    data, db = workspace
    main(["load", str(data / "cdc_batches"), "--db", db])
    first = capsys.readouterr().out.splitlines()[-1]
    main(["load", str(data / "cdc_batches"), "--db", db])
    second = capsys.readouterr().out.splitlines()
    assert second[-1].split(";")[0] == first.split(";")[0]  # same versions and current lanes
    assert all(line.split()[6] == "0" for line in second[1:-1])  # nothing opened


def test_rerate_prints_summary_and_writes_report(workspace, tmp_path, capsys):
    data, db = workspace
    main(["load", str(data / "cdc_batches"), "--db", db])
    capsys.readouterr()
    out_csv = tmp_path / "report.csv"
    assert main(["rerate", str(data / "invoices.csv"), "--db", db, "--out", str(out_csv)]) == 0
    out = capsys.readouterr().out
    assert "overbilled" in out and "no_rate" in out
    assert out_csv.read_text().startswith("invoice_id,")


def test_load_of_an_empty_directory_fails_cleanly(tmp_path, capsys):
    assert main(["load", str(tmp_path), "--db", str(tmp_path / "x.duckdb")]) == 2
    assert "no .jsonl batches" in capsys.readouterr().err


def test_rate_at_rejects_bad_timestamps(tmp_path):
    with pytest.raises(SystemExit):
        main(["rate-at", "L1", "yesterday", "--db", str(tmp_path / "x.duckdb")])


@pytest.mark.parametrize("command", [["history", "L1"], ["rate-at", "L1", "2026-07-01T00:00:00"]])
def test_query_commands_report_a_missing_database_without_creating_it(tmp_path, capsys, command):
    db = tmp_path / "missing.duckdb"
    assert main([*command, "--db", str(db)]) == 2
    assert "not found" in capsys.readouterr().err
    assert not db.exists()


def test_query_commands_report_a_database_that_was_never_loaded(tmp_path, capsys):
    import duckdb

    db = tmp_path / "empty.duckdb"
    duckdb.connect(str(db)).close()
    assert main(["history", "L1", "--db", str(db)]) == 2
    assert "no dim_rate_card table" in capsys.readouterr().err


def test_rerate_reports_a_missing_invoice_file(workspace, tmp_path, capsys):
    data, db = workspace
    main(["load", str(data / "cdc_batches"), "--db", db])
    capsys.readouterr()
    assert main(["rerate", str(tmp_path / "nope.csv"), "--db", db]) == 2
    assert "invoice file" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "abc"])
def test_rerate_rejects_bad_tolerance(tmp_path, value):
    with pytest.raises(SystemExit):
        main(["rerate", "x.csv", "--db", str(tmp_path / "x.duckdb"), "--tolerance", value])


def test_rerate_reports_an_invoice_file_with_missing_columns(workspace, tmp_path, capsys):
    data, db = workspace
    main(["load", str(data / "cdc_batches"), "--db", db])
    bad = tmp_path / "bad.csv"
    bad.write_text("invoice_id,lane_id,ship_ts,weight,billed_amount,currency\n")
    capsys.readouterr()
    assert main(["rerate", str(bad), "--db", db]) == 2
    assert "missing column(s) weight_kg" in capsys.readouterr().err
