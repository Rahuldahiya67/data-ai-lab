import pytest

from sop_retrieval_benchmark.bm25 import BM25Retriever, Hit
from sop_retrieval_benchmark.chunk import Chunk, chunk_documents
from sop_retrieval_benchmark.evaluate import evaluate, summarise, summarise_by
from sop_retrieval_benchmark.generate import Question, generate


def _q(qid="q1", doc="returns-del-01", section="Return window", style="lexical"):
    return Question(qid, "anything", doc, section, "7 days", "returns", "DEL-01", style)


def _c(doc, section, topic="returns"):
    return Chunk(f"{doc}#{section}", doc, section, doc[-6:].upper(), topic, "", "")


class FixedRetriever:
    """Returns the same ranked list for every query."""

    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, k=5):
        return [Hit(c, 1.0) for c in self.chunks[:k]]


def test_rank_and_metrics():
    r = FixedRetriever([_c("returns-bom-02", "Grading"), _c("returns-del-01", "Return window")])
    res = evaluate(r, [_q()], k=5)
    assert res[0].rank == 2
    s = summarise(res)
    assert s.recall == {1: 0.0, 3: 1.0, 5: 1.0}
    assert s.mrr == 0.5
    assert s.site_confusion == 0.0


def test_site_confusion_flagged_when_top_hit_is_other_sites_same_section():
    r = FixedRetriever([_c("returns-bom-02", "Return window")])
    res = evaluate(r, [_q()])
    assert res[0].rank is None and res[0].site_confused
    assert summarise(res).site_confusion == 1.0


def test_wrong_section_same_doc_is_not_site_confusion():
    r = FixedRetriever([_c("returns-del-01", "Grading")])
    res = evaluate(r, [_q()])
    assert not res[0].site_confused


def test_empty_hits_count_as_miss():
    res = evaluate(FixedRetriever([]), [_q()])
    assert res[0].rank is None and res[0].top_doc_id is None
    assert summarise(res).mrr == 0.0


def test_k_limits_what_counts():
    r = FixedRetriever([_c("x-1", "a"), _c("x-2", "a"), _c("returns-del-01", "Return window")])
    assert evaluate(r, [_q()], k=2)[0].rank is None


def test_group_by_style():
    r = FixedRetriever([_c("returns-del-01", "Return window")])
    res = evaluate(r, [_q("q1"), _q("q2", style="paraphrase", section="Grading")])
    by = summarise_by(res, "style")
    assert by["lexical"].recall[1] == 1.0
    assert by["paraphrase"].recall[1] == 0.0
    with pytest.raises(ValueError):
        summarise_by(res, "nope")


def test_validation():
    with pytest.raises(ValueError):
        evaluate(FixedRetriever([]), [])
    with pytest.raises(ValueError):
        evaluate(FixedRetriever([]), [_q()], k=0)
    with pytest.raises(ValueError):
        summarise([])


def test_bm25_on_generated_benchmark_is_reasonable():
    docs, questions = generate()
    res = evaluate(BM25Retriever(chunk_documents(docs)), questions)
    by = summarise_by(res, "style")
    # Keyword questions should mostly be found; paraphrases are harder.
    assert by["lexical"].recall[5] >= 0.8
    assert by["paraphrase"].recall[5] < by["lexical"].recall[5]
