"""Score a retriever against the labelled question set.

A retrieved chunk is correct when it comes from the gold document and the gold
section. Metrics:

- recall@k: share of questions with a correct chunk in the top k.
- MRR: mean of 1 / rank of the first correct chunk (0 when none is in the top k).
- site confusion: share of questions whose top hit is the right topic and
  section but another site's document. This is the failure that produces a
  confident answer with the wrong site's rule.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol, Sequence

from .bm25 import Hit
from .generate import Question


class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[Hit]: ...


@dataclass(frozen=True)
class QuestionResult:
    qid: str
    style: str
    topic: str
    rank: int | None  # 1-based rank of the first correct chunk, None if not retrieved
    top_doc_id: str | None
    top_section: str | None
    site_confused: bool
    depth: int  # how many hits were retrieved; recall is only defined up to here


@dataclass(frozen=True)
class Summary:
    n: int
    recall: dict[int, float]
    mrr: float
    site_confusion: float


def _is_correct(hit: Hit, q: Question) -> bool:
    return hit.chunk.doc_id == q.gold_doc_id and hit.chunk.section == q.gold_section


def evaluate(retriever: Retriever, questions: Sequence[Question], k: int = 5) -> list[QuestionResult]:
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if not questions:
        raise ValueError("no questions to evaluate")
    results = []
    for q in questions:
        hits = retriever.search(q.question, k=k)
        rank = next((i for i, h in enumerate(hits, start=1) if _is_correct(h, q)), None)
        top = hits[0].chunk if hits else None
        confused = (
            top is not None
            and top.doc_id != q.gold_doc_id
            and top.topic == q.topic
            and top.section == q.gold_section
        )
        results.append(
            QuestionResult(
                q.qid,
                q.style,
                q.topic,
                rank,
                top.doc_id if top else None,
                top.section if top else None,
                confused,
                k,
            )
        )
    return results


def summarise(results: Sequence[QuestionResult], ks: Sequence[int] = (1, 3, 5)) -> Summary:
    if not results:
        raise ValueError("no results to summarise")
    depth = min(r.depth for r in results)
    too_deep = [k for k in ks if k > depth]
    if too_deep:
        raise ValueError(
            f"cannot report recall@{too_deep[0]}: results were retrieved with k={depth}; "
            f"evaluate with a larger k or pass ks up to {depth}"
        )
    n = len(results)
    recall = {k: sum(r.rank is not None and r.rank <= k for r in results) / n for k in ks}
    mrr = sum(1 / r.rank for r in results if r.rank) / n
    confusion = sum(r.site_confused for r in results) / n
    return Summary(n, recall, mrr, confusion)


def summarise_by(results: Sequence[QuestionResult], field: str, ks: Sequence[int] = (1, 3, 5)) -> dict[str, Summary]:
    """Summaries per value of ``field`` (e.g. 'style' or 'topic')."""
    if field not in QuestionResult.__dataclass_fields__:
        raise ValueError(f"unknown field '{field}'")
    groups: dict[str, list[QuestionResult]] = defaultdict(list)
    for r in results:
        groups[getattr(r, field)].append(r)
    return {key: summarise(group, ks) for key, group in sorted(groups.items())}
