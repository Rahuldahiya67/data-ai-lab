import json
from pathlib import Path

import pandas as pd
import pytest

from carrier_feed_contract_checker.cli import EXIT_ERROR, EXIT_FAIL, EXIT_PASS, main

CONTRACT = str(Path(__file__).resolve().parents[1] / "contracts" / "carrier_shipment_status.yaml")


def test_generate_then_check_passes(tmp_path, capsys):
    feed = tmp_path / "feed.csv"
    assert main(["generate", "--out", str(feed), "--rows", "800", "--seed", "4"]) == EXIT_PASS
    out_dir = tmp_path / "out"
    code = main(["check", str(feed), "--contract", CONTRACT, "--out-dir", str(out_dir)])
    assert code == EXIT_PASS
    assert capsys.readouterr().out.splitlines()[-2].startswith("PASS:")

    accepted = pd.read_csv(out_dir / "accepted.csv", dtype=str, keep_default_na=False)
    quarantined = pd.read_csv(out_dir / "quarantine.csv", dtype=str, keep_default_na=False)
    assert len(accepted) + len(quarantined) == 800
    assert "_violations" in quarantined.columns
    report = json.loads((out_dir / "report.json").read_text())
    assert report["rows"]["quarantined"] == len(quarantined)
    assert (out_dir / "report.md").read_text().startswith("# Feed quality report: feed.csv")


def test_check_returns_fail_code_when_reject_rate_too_high(tmp_path, capsys):
    feed = tmp_path / "bad.csv"
    main(["generate", "--out", str(feed), "--rows", "500", "--defect-rate", "0.4"])
    code = main(["check", str(feed), "--contract", CONTRACT, "--out-dir", str(tmp_path / "o")])
    assert code == EXIT_FAIL
    assert "failed feed.max_reject_rate" in capsys.readouterr().out


def test_check_reports_bad_contract(tmp_path, capsys):
    contract = tmp_path / "c.yaml"
    contract.write_text("columns:\n  a:\n    type: string\n    regex: '[A-'\n")
    feed = tmp_path / "f.csv"
    feed.write_text("a\nx\n")
    code = main(["check", str(feed), "--contract", str(contract), "--out-dir", str(tmp_path / "o")])
    assert code == EXIT_ERROR
    assert "invalid regex" in capsys.readouterr().err


def test_check_reports_missing_feed(tmp_path, capsys):
    code = main(["check", str(tmp_path / "nope.csv"), "--contract", CONTRACT])
    assert code == EXIT_ERROR
    assert "error:" in capsys.readouterr().err


def test_generate_rejects_bad_rate(tmp_path, capsys):
    code = main(["generate", "--out", str(tmp_path / "f.csv"), "--defect-rate", "2"])
    assert code == EXIT_ERROR


def test_check_reports_unparseable_yaml(tmp_path, capsys):
    contract = tmp_path / "c.yaml"
    contract.write_text("columns: [unclosed\n")
    feed = tmp_path / "f.csv"
    feed.write_text("a\nx\n")
    code = main(["check", str(feed), "--contract", str(contract), "--out-dir", str(tmp_path / "o")])
    assert code == EXIT_ERROR


def test_misspelt_feed_limit_is_an_error_not_a_silent_pass(tmp_path, capsys):
    feed = tmp_path / "feed.csv"
    main(["generate", "--out", str(feed), "--rows", "200", "--defect-rate", "0.5"])
    contract = tmp_path / "typo.yaml"
    text = Path(CONTRACT).read_text().replace("max_reject_rate:", "max_reject_rates:")
    contract.write_text(text)
    code = main(["check", str(feed), "--contract", str(contract), "--out-dir", str(tmp_path / "out")])
    assert code == EXIT_ERROR
    assert "max_reject_rates" in capsys.readouterr().err


def test_check_rejects_a_negative_sample_size(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["check", str(tmp_path / "f.csv"), "--contract", CONTRACT, "--sample", "-1"])
    assert exc.value.code == 2
