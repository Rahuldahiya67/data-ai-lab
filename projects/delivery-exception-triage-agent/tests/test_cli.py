import json

import pytest

from delivery_exception_triage_agent.cli import eval_main, run_main
from delivery_exception_triage_agent.dataset import write_dataset
from delivery_exception_triage_agent.generator import generate


@pytest.fixture
def data_dir(tmp_path):
    shipments, tickets = generate(n_tickets=40, seed=11)
    write_dataset(shipments, tickets, tmp_path / "data")
    return tmp_path / "data"


def test_eval_cli_writes_both_reports(data_dir, tmp_path, capsys):
    assert eval_main(["--data", str(data_dir), "--out", str(tmp_path / "reports")]) == 0
    out = capsys.readouterr().out
    assert out.startswith("40 tickets | action accuracy")
    data = json.loads((tmp_path / "reports" / "evaluation.json").read_text())
    assert data["planner"] == "rule-based" and data["summary"]["tickets"] == 40
    assert (tmp_path / "reports" / "evaluation.md").read_text().startswith("# Triage evaluation: rule-based")


def test_eval_cli_reports_a_missing_dataset(tmp_path, capsys):
    assert eval_main(["--data", str(tmp_path / "nope")]) == 2
    assert "run triage-generate" in capsys.readouterr().err


def test_eval_cli_rejects_a_zero_step_budget(data_dir):
    with pytest.raises(SystemExit):
        eval_main(["--data", str(data_dir), "--max-steps", "0"])


def test_run_cli_prints_the_trace(data_dir, capsys):
    assert run_main(["TKT-00001", "--data", str(data_dir)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("TKT-00001 (")
    assert out[1].startswith("  1. get_shipment(")
    assert out[-2].startswith("resolution: ")
    assert out[-1].startswith("expected:   ")


def test_run_cli_unknown_ticket(data_dir, capsys):
    assert run_main(["TKT-99999", "--data", str(data_dir)]) == 2
    assert "no ticket 'TKT-99999'" in capsys.readouterr().err
