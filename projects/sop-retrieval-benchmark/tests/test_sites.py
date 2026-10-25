import pytest

from sop_retrieval_benchmark.bm25 import BM25Retriever
from sop_retrieval_benchmark.chunk import chunk_documents
from sop_retrieval_benchmark.evaluate import evaluate, summarise
from sop_retrieval_benchmark.generate import SITES, Site, generate
from sop_retrieval_benchmark.sites import SiteDetector, SiteFilteredRetriever


@pytest.fixture(scope="module")
def bench():
    docs, questions = generate()
    return chunk_documents(docs), questions


def test_detects_code_and_city():
    d = SiteDetector(SITES)
    assert d.detect("Return window at site BOM-02?") == {"BOM-02"}
    assert d.detect("What is the return window at the Mumbai warehouse?") == {"BOM-02"}
    assert d.detect("Return window in delhi or chennai") == {"DEL-01", "MAA-04"}
    assert d.detect("What is the return window?") == set()


def test_longest_alias_wins():
    d = SiteDetector([Site("NYC-01", "New York"), Site("YRK-02", "York")])
    assert d.detect("dock rules in New York") == {"NYC-01"}
    assert d.detect("dock rules in York") == {"YRK-02"}
    assert d.detect("new rules for york") == {"YRK-02"}


def test_rejects_ambiguous_aliases():
    with pytest.raises(ValueError, match="both"):
        SiteDetector([Site("A-1", "Pune"), Site("B-2", "Pune")])
    with pytest.raises(ValueError):
        SiteDetector([])


def test_filter_keeps_only_named_site(bench):
    chunks, _ = bench
    r = SiteFilteredRetriever(BM25Retriever(chunks), SiteDetector(SITES))
    hits = r.search("How many days are customer returns accepted at Kolkata?", k=5)
    assert len(hits) == 5
    assert {h.chunk.site_code for h in hits} == {"CCU-05"}
    assert hits[0].chunk.section == "Return window"


def test_no_site_or_several_sites_means_no_filter(bench):
    chunks, _ = bench
    base = BM25Retriever(chunks)
    r = SiteFilteredRetriever(base, SiteDetector(SITES))
    for q in ("How many days are customer returns accepted?", "Return window at Delhi and Mumbai"):
        assert [h.chunk.chunk_id for h in r.search(q)] == [h.chunk.chunk_id for h in base.search(q)]


def test_filter_removes_site_confusion_on_benchmark(bench):
    chunks, questions = bench
    base = BM25Retriever(chunks)
    filtered = SiteFilteredRetriever(base, SiteDetector(SITES))
    before = summarise(evaluate(base, questions))
    after = summarise(evaluate(filtered, questions))
    assert after.site_confusion == 0.0
    assert after.recall[5] >= before.recall[5]


def test_rejects_bad_k(bench):
    chunks, _ = bench
    with pytest.raises(ValueError):
        SiteFilteredRetriever(BM25Retriever(chunks), SiteDetector(SITES)).search("x", k=0)
