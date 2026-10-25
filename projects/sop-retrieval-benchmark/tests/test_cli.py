import json
from pathlib import Path

import pytest

from sop_retrieval_benchmark.cli import _sites_from_documents, build_retriever, main
from sop_retrieval_benchmark.generate import SITES, generate

SAMPLE = Path(__file__).resolve().parents[1] / "data"


def test_generate_writes_both_files(tmp_path, capsys):
    assert main(["generate", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "sop_documents.jsonl").exists()
    assert (tmp_path / "questions.jsonl").exists()
    assert "wrote" in capsys.readouterr().out


def test_run_prints_table_with_groups(capsys):
    assert main(["run", "--data", str(SAMPLE)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].split()[:3] == ["retriever", "group", "n"]
    groups = [line.split()[1] for line in out[2:]]
    assert groups == ["all", "lexical", "paraphrase"]


def test_run_all_writes_json(tmp_path, capsys):
    out = tmp_path / "res" / "results.json"
    assert main(["run", "--data", str(SAMPLE), "--all", "--by", "none", "--json", str(out)]) == 0
    rows = json.loads(out.read_text())
    assert [r["retriever"] for r in rows] == ["bm25", "bm25+site", "expanded", "expanded+site"]
    assert all(r["n"] == 210 and 0 <= r["recall@5"] <= 1 for r in rows)
    # Expansion should help overall recall on this question set.
    assert rows[2]["recall@5"] > rows[0]["recall@5"]


def test_missing_data_is_a_clean_error(tmp_path, capsys):
    assert main(["run", "--data", str(tmp_path)]) == 2
    assert "does not exist" in capsys.readouterr().err


def test_sites_are_read_from_titles():
    docs, _ = generate()
    assert sorted(_sites_from_documents(docs), key=lambda s: s.code) == sorted(SITES, key=lambda s: s.code)


def test_build_retriever_rejects_unknown_name():
    docs, _ = generate()
    from sop_retrieval_benchmark.chunk import chunk_documents

    with pytest.raises(ValueError, match="unknown retriever"):
        build_retriever(chunk_documents(docs), "dense", False)
