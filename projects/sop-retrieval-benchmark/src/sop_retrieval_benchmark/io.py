"""Read and write the corpus and question set as JSON Lines, with validation.

Validation errors name the file and line so a broken hand-edited file is easy
to fix; nothing is silently skipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .generate import Document, Question, Section

DOC_FIELDS = ("doc_id", "title", "topic", "site_code", "sections")
QUESTION_FIELDS = ("qid", "question", "gold_doc_id", "gold_section", "answer", "topic", "site_code", "style")
QUESTION_STYLES = ("lexical", "paraphrase")


class DataValidationError(ValueError):
    """Raised when a corpus or question file does not match the expected schema."""


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _read_jsonl(path: Path) -> list[tuple[int, dict]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    rows = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataValidationError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
            if not isinstance(row, dict):
                raise DataValidationError(f"{path}:{lineno}: expected a JSON object")
            rows.append((lineno, row))
    if not rows:
        raise DataValidationError(f"{path}: file has no records")
    return rows


def _require(row: dict, fields: tuple[str, ...], where: str) -> None:
    missing = [f for f in fields if f not in row]
    if missing:
        raise DataValidationError(f"{where}: missing field(s) {', '.join(missing)}")
    for f in fields:
        if f != "sections" and (not isinstance(row[f], str) or not row[f].strip()):
            raise DataValidationError(f"{where}: field '{f}' must be a non-empty string")


def load_documents(path: str | Path) -> list[Document]:
    path = Path(path)
    docs: list[Document] = []
    seen: set[str] = set()
    for lineno, row in _read_jsonl(path):
        where = f"{path}:{lineno}"
        _require(row, DOC_FIELDS, where)
        if row["doc_id"] in seen:
            raise DataValidationError(f"{where}: duplicate doc_id '{row['doc_id']}'")
        seen.add(row["doc_id"])
        sections = row["sections"]
        if not isinstance(sections, list) or not sections:
            raise DataValidationError(f"{where}: 'sections' must be a non-empty list")
        parsed = []
        headings: set[str] = set()
        for i, sec in enumerate(sections):
            if not isinstance(sec, dict) or not isinstance(sec.get("heading"), str) or not isinstance(
                sec.get("text"), str
            ):
                raise DataValidationError(f"{where}: section {i} needs string 'heading' and 'text'")
            if not sec["text"].strip():
                raise DataValidationError(f"{where}: section {i} ('{sec['heading']}') has empty text")
            if sec["heading"] in headings:
                # Gold labels are (doc_id, heading); a repeated heading makes them ambiguous.
                raise DataValidationError(f"{where}: section {i} repeats heading '{sec['heading']}'")
            headings.add(sec["heading"])
            parsed.append(Section(sec["heading"], sec["text"]))
        docs.append(Document(row["doc_id"], row["title"], row["topic"], row["site_code"], parsed))
    return docs


def load_questions(path: str | Path, documents: list[Document] | None = None) -> list[Question]:
    """Load questions; if documents are given, check every gold reference exists."""
    path = Path(path)
    sections_by_doc = {d.doc_id: {s.heading for s in d.sections} for d in documents} if documents is not None else None
    out: list[Question] = []
    seen: set[str] = set()
    for lineno, row in _read_jsonl(path):
        where = f"{path}:{lineno}"
        _require(row, QUESTION_FIELDS, where)
        if row["qid"] in seen:
            raise DataValidationError(f"{where}: duplicate qid '{row['qid']}'")
        seen.add(row["qid"])
        if row["style"] not in QUESTION_STYLES:
            raise DataValidationError(f"{where}: style must be one of {QUESTION_STYLES}, got '{row['style']}'")
        if sections_by_doc is not None:
            if row["gold_doc_id"] not in sections_by_doc:
                raise DataValidationError(f"{where}: gold_doc_id '{row['gold_doc_id']}' is not in the corpus")
            if row["gold_section"] not in sections_by_doc[row["gold_doc_id"]]:
                raise DataValidationError(
                    f"{where}: gold_section '{row['gold_section']}' is not a section of '{row['gold_doc_id']}'"
                )
        out.append(Question(**{f: row[f] for f in QUESTION_FIELDS}))
    return out


def write_benchmark(out_dir: str | Path, seed: int = 7) -> tuple[Path, Path]:
    """Generate the corpus and questions and write them to ``out_dir``."""
    from .generate import generate

    docs, questions = generate(seed=seed)
    out = Path(out_dir)
    docs_path, q_path = out / "sop_documents.jsonl", out / "questions.jsonl"
    write_jsonl(docs_path, (d.to_dict() for d in docs))
    write_jsonl(q_path, (q.to_dict() for q in questions))
    return docs_path, q_path
