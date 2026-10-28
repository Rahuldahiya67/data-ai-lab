import pytest

from sop_retrieval_benchmark.bm25 import BM25Retriever
from sop_retrieval_benchmark.chunk import Chunk, chunk_documents
from sop_retrieval_benchmark.generate import generate


def _chunk(cid, text):
    return Chunk(cid, cid, "s", "S", "t", text, text)


def test_matching_chunk_ranks_first():
    r = BM25Retriever([_chunk("a", "reefer trailer pre-cooling"), _chunk("b", "returns bay refund"), _chunk("c", "spill kit")])
    hits = r.search("when is the refund released", k=2)
    assert hits[0].chunk.chunk_id == "b"
    assert len(hits) == 1  # only chunks that share a term are returned


def test_rarer_terms_weigh_more():
    r = BM25Retriever([_chunk("a", "dock slot"), _chunk("b", "dock detention"), _chunk("c", "dock lanes")])
    assert r.search("dock detention")[0].chunk.chunk_id == "b"


def test_no_overlap_returns_empty():
    r = BM25Retriever([_chunk("a", "dock slot")])
    assert r.search("photos of damage") == []


def test_ties_are_deterministic():
    r = BM25Retriever([_chunk("a", "same words"), _chunk("b", "same words")])
    assert [h.chunk.chunk_id for h in r.search("same")] == ["a", "b"]


def test_site_in_question_picks_that_sites_version():
    docs, _ = generate()
    r = BM25Retriever(chunk_documents(docs))
    hit = r.search("How many days after delivery are customer returns accepted at BOM-02?", k=1)[0]
    assert hit.chunk.doc_id == "returns-bom-02"
    assert hit.chunk.section == "Return window"


def test_validation():
    with pytest.raises(ValueError):
        BM25Retriever([])
    with pytest.raises(ValueError):
        BM25Retriever([_chunk("a", "x"), _chunk("a", "y")])
    with pytest.raises(ValueError):
        BM25Retriever([_chunk("a", "x")], b=1.5)
    with pytest.raises(ValueError):
        BM25Retriever([_chunk("a", "x")]).search("x", k=0)
