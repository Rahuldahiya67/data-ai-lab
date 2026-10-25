import json

import pytest

from sop_retrieval_benchmark.generate import generate
from sop_retrieval_benchmark.io import DataValidationError, load_documents, load_questions, write_jsonl


@pytest.fixture
def files(tmp_path):
    docs, questions = generate()
    dp, qp = tmp_path / "docs.jsonl", tmp_path / "questions.jsonl"
    write_jsonl(dp, (d.to_dict() for d in docs))
    write_jsonl(qp, (q.to_dict() for q in questions))
    return dp, qp, docs, questions


def test_round_trip(files):
    dp, qp, docs, questions = files
    loaded_docs = load_documents(dp)
    loaded_q = load_questions(qp, loaded_docs)
    assert [d.to_dict() for d in loaded_docs] == [d.to_dict() for d in docs]
    assert [q.to_dict() for q in loaded_q] == [q.to_dict() for q in questions]


def _write(path, rows):
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n")


def test_invalid_json_reports_line(tmp_path):
    p = tmp_path / "d.jsonl"
    _write(p, ['{"doc_id": "a"', ])
    with pytest.raises(DataValidationError, match=r"d.jsonl:1: invalid JSON"):
        load_documents(p)


def test_missing_field_and_empty_section(tmp_path):
    p = tmp_path / "d.jsonl"
    _write(p, [{"doc_id": "a", "title": "t", "topic": "x", "sections": []}])
    with pytest.raises(DataValidationError, match="missing field"):
        load_documents(p)
    _write(p, [{"doc_id": "a", "title": "t", "topic": "x", "site_code": "S", "sections": [{"heading": "h", "text": " "}]}])
    with pytest.raises(DataValidationError, match="empty text"):
        load_documents(p)


def test_duplicate_doc_id(tmp_path):
    row = {"doc_id": "a", "title": "t", "topic": "x", "site_code": "S", "sections": [{"heading": "h", "text": "x"}]}
    p = tmp_path / "d.jsonl"
    _write(p, [row, row])
    with pytest.raises(DataValidationError, match="duplicate doc_id"):
        load_documents(p)


def test_question_gold_reference_checked(files, tmp_path):
    dp, qp, docs, questions = files
    bad = questions[0].to_dict() | {"gold_section": "Nope"}
    p = tmp_path / "q.jsonl"
    _write(p, [bad])
    with pytest.raises(DataValidationError, match="gold_section 'Nope'"):
        load_questions(p, docs)
    # Without a corpus the reference cannot be checked, so it loads.
    assert load_questions(p)[0].gold_section == "Nope"


def test_bad_style_and_empty_file(files, tmp_path):
    _, _, _, questions = files
    p = tmp_path / "q.jsonl"
    _write(p, [questions[0].to_dict() | {"style": "other"}])
    with pytest.raises(DataValidationError, match="style must be"):
        load_questions(p)
    p.write_text("\n")
    with pytest.raises(DataValidationError, match="no records"):
        load_questions(p)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "nope.jsonl")


def test_committed_sample_data_matches_generator(tmp_path):
    from pathlib import Path

    from sop_retrieval_benchmark.io import write_benchmark

    data = Path(__file__).resolve().parents[1] / "data"
    fresh_docs, fresh_q = write_benchmark(tmp_path, seed=7)
    assert (data / "sop_documents.jsonl").read_text() == fresh_docs.read_text()
    assert (data / "questions.jsonl").read_text() == fresh_q.read_text()
    # and the committed files pass validation together
    load_questions(data / "questions.jsonl", load_documents(data / "sop_documents.jsonl"))


def _doc(doc_id, headings):
    return {
        "doc_id": doc_id,
        "title": f"{doc_id} SOP",
        "topic": "returns",
        "site_code": "DEL-01",
        "sections": [{"heading": h, "text": f"{h} text."} for h in headings],
    }


def test_repeated_heading_in_one_document_is_rejected(tmp_path):
    # Gold matching is by (doc_id, heading), so duplicates make labels ambiguous (#15).
    p = tmp_path / "d.jsonl"
    _write(p, [_doc("a", ["Scope", "Exceptions", "Exceptions"])])
    with pytest.raises(DataValidationError, match=r"d.jsonl:1: section 2 repeats heading 'Exceptions'"):
        load_documents(p)


def test_same_heading_in_different_documents_is_fine(tmp_path):
    p = tmp_path / "d.jsonl"
    _write(p, [_doc("a", ["Scope", "Records"]), _doc("b", ["Scope", "Records"])])
    assert len(load_documents(p)) == 2


def test_empty_corpus_still_checks_gold_references(files, tmp_path):
    _, qp, _, _ = files
    with pytest.raises(DataValidationError, match="is not in the corpus"):
        load_questions(qp, documents=[])
