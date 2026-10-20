import pandas as pd

from intermittent_demand_forecaster.cli import main


def test_generate_backtest_reorder_end_to_end(tmp_path, capsys):
    data = tmp_path / "demand.csv"
    assert main(["generate", "--skus", "8", "--weeks", "60", "--seed", "5", "--out", str(data)]) == 0
    assert len(pd.read_csv(data)) == 8 * 60

    results = tmp_path / "results.csv"
    assert main(["backtest", "--input", str(data), "--holdout", "12", "--out", str(results)]) == 0
    assert pd.read_csv(results)["sku"].nunique() == 8
    out = capsys.readouterr().out
    assert "Mean RMSSE by demand class" in out and "Selected method per SKU" in out

    report = tmp_path / "out" / "reorder.csv"
    code = main(
        ["reorder", "--input", str(data), "--holdout", "12", "--lead-time", "3",
         "--service-level", "0.9", "--out", str(report)]
    )
    assert code == 0
    table = pd.read_csv(report)
    assert len(table) == 8
    assert (table["lead_time_weeks"] == 3).all()
    out = capsys.readouterr().out
    assert "service level 90%" in out
    assert "achieved_service" in out


def test_bad_input_returns_error_code(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"sku": ["A"], "week": ["2024-01-01"], "demand": [-1]}).to_csv(bad, index=False)
    assert main(["backtest", "--input", str(bad)]) == 2
    assert "non-negative" in capsys.readouterr().err


def test_missing_file_returns_error_code(tmp_path, capsys):
    assert main(["reorder", "--input", str(tmp_path / "nope.csv")]) == 2
    assert "error:" in capsys.readouterr().err
