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
    assert groups == ["all", "heldout", "lexical", "paraphrase"]


def test_run_all_writes_json(tmp_path, capsys):
    out = tmp_path / "res" / "results.json"
    assert main(["run", "--data", str(SAMPLE), "--all", "--by", "none", "--json", str(out)]) == 0
    rows = json.loads(out.read_text())
    assert [r["retriever"] for r in rows] == ["bm25", "bm25+site", "expanded", "expanded+site"]
    assert all(r["n"] == 315 and 0 <= r["recall@5"] <= 1 for r in rows)
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


def test_no_title_removes_the_site_name_from_chunks_and_causes_site_confusion(tmp_path, capsys):
    with_title, without_title = tmp_path / "a.json", tmp_path / "b.json"
    assert main(["run", "--data", str(SAMPLE), "--by", "none", "--json", str(with_title)]) == 0
    assert main(["run", "--data", str(SAMPLE), "--by", "none", "--no-title", "--json", str(without_title)]) == 0
    a = json.loads(with_title.read_text())[0]
    b = json.loads(without_title.read_text())[0]
    # Without the title, sections from different sites read almost the same,
    # so the top hit is often the right rule from another site.
    assert a["site_confusion"] == 0
    assert b["site_confusion"] > 0.1
    assert b["recall@1"] < a["recall@1"]


def test_small_max_words_works_without_an_explicit_overlap(capsys):
    # Before the overlap was capped, any --max-words of 20 or less failed with
    # an error about an overlap option the CLI did not expose.
    assert main(["run", "--data", str(SAMPLE), "--by", "none", "--max-words", "15"]) == 0
    assert capsys.readouterr().out.splitlines()[2].split()[0] == "bm25"


def test_default_overlap_keeps_twenty_words_for_normal_chunks():
    from sop_retrieval_benchmark.cli import default_overlap

    assert default_overlap(120) == 20
    assert default_overlap(30) == 15
    assert default_overlap(1) == 0


def test_explicit_overlap_is_passed_through_and_validated(tmp_path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    assert main(["run", "--data", str(SAMPLE), "--by", "none", "--json", str(a)]) == 0
    assert main(["run", "--data", str(SAMPLE), "--by", "none", "--overlap", "20", "--json", str(b)]) == 0
    assert json.loads(a.read_text()) == json.loads(b.read_text())
    assert main(["run", "--data", str(SAMPLE), "--max-words", "10", "--overlap", "10"]) == 2
    assert "overlap must be in [0, max_words)" in capsys.readouterr().err
