import pytest

from sop_retrieval_benchmark.bm25 import BM25Retriever
from sop_retrieval_benchmark.chunk import chunk_documents
from sop_retrieval_benchmark.evaluate import evaluate, summarise_by
from sop_retrieval_benchmark.expand import ExpandedRetriever, QueryExpander
from sop_retrieval_benchmark.generate import generate


def test_expansion_appends_document_terms_and_keeps_original():
    e = QueryExpander({"refrigerated": "reefer", "trucking company": "carrier"})
    q = "When does a refrigerated trucking company arrive?"
    assert e.expand(q) == f"{q} reefer carrier"


def test_phrases_match_whole_tokens_in_order():
    e = QueryExpander({"money back": "refund"})
    assert e.matches("When do I get my money back?") == ["refund"]
    assert e.matches("back the money") == []
    assert e.matches("moneyback") == []


def test_matching_uses_the_same_stemming_as_documents():
    e = QueryExpander({"pictures": "photo"})
    assert e.matches("How many picture files?") == ["photo"]


def test_no_match_returns_query_unchanged():
    e = QueryExpander({"reefer": "chilled"})
    assert e.expand("Return window at Delhi") == "Return window at Delhi"


def test_rejects_empty_glossary_and_stopword_only_phrases():
    with pytest.raises(ValueError):
        QueryExpander({})
    with pytest.raises(ValueError, match="no searchable words"):
        QueryExpander({"the": "x"})
    with pytest.raises(ValueError, match="expansion"):
        QueryExpander({"truck": "of the"})


def test_default_glossary_narrows_paraphrase_gap_without_hurting_keywords():
    docs, questions = generate()
    base = BM25Retriever(chunk_documents(docs))
    before = summarise_by(evaluate(base, questions), "style")
    after = summarise_by(evaluate(ExpandedRetriever(base), questions), "style")
    assert after["lexical"].recall[5] >= before["lexical"].recall[5]
    assert after["paraphrase"].recall[5] > before["paraphrase"].recall[5]


def test_single_word_entries_fire_out_of_context():
    # Known limitation, kept visible on purpose: a one-word entry matches the
    # word wherever it appears. "lift truck" is a forklift, not a trailer, and
    # a "cold lorry" is about pre-cooling, not the storage temperature.
    e = QueryExpander()
    assert "trailer" in e.matches("Above what load do pallets require a sit-down lift truck?")
    assert "temperature chilled" in e.matches("Before stock goes onto a cold lorry, how long must its unit run?")


def test_glossary_written_for_paraphrases_does_not_carry_over_to_heldout_wording():
    # The glossary lifts the paraphrase set it was written against but not
    # wording it has not seen; on the held-out set it costs recall. If this
    # starts failing because held-out recall went up, the glossary was most
    # likely tuned on the held-out questions, which defeats their purpose.
    docs, questions = generate()
    base = BM25Retriever(chunk_documents(docs))
    before = summarise_by(evaluate(base, questions), "style")
    after = summarise_by(evaluate(ExpandedRetriever(base), questions), "style")
    assert after["paraphrase"].recall[5] - before["paraphrase"].recall[5] > 0.3
    assert after["heldout"].recall[5] <= before["heldout"].recall[5]
