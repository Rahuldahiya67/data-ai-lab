import hashlib
import json
from collections import Counter

import pytest

from sop_retrieval_benchmark.expand import LOGISTICS_GLOSSARY
from sop_retrieval_benchmark.generate import HELDOUT_QUESTIONS, SITES, TOPICS, Site, generate
from sop_retrieval_benchmark.text import tokenize


def test_one_document_per_topic_and_site():
    docs, _ = generate()
    assert len(docs) == len(TOPICS) * len(SITES)
    assert len({d.doc_id for d in docs}) == len(docs)


def test_three_questions_per_fact_one_per_style():
    docs, questions = generate()
    n_facts = sum(len(t["facts"]) for t in TOPICS.values()) * len(SITES)
    assert len(questions) == 3 * n_facts
    assert Counter(q.style for q in questions) == {"lexical": n_facts, "paraphrase": n_facts, "heldout": n_facts}
    assert len({q.qid for q in questions}) == len(questions)


def test_every_fact_has_exactly_one_heldout_template():
    headings = {(key, f.heading) for key, t in TOPICS.items() for f in t["facts"]}
    assert set(HELDOUT_QUESTIONS) == headings


# sha256 of the 210 lexical and paraphrase questions (seed 7) that the glossary
# was written against, as committed on day 2.
ORIGINAL_QUESTIONS_SHA256 = "0aa81555bb7f94c1658ec89137096aab41ff877d0ae30528c5c614792c1d0b93"


def test_heldout_questions_did_not_change_the_original_ones():
    # The first two styles must stay identical to the set the glossary was
    # written against, or the held-out comparison is not like for like.
    _, questions = generate()
    originals = [q.to_dict() for q in questions if q.style != "heldout"]
    digest = hashlib.sha256(json.dumps(originals, sort_keys=True).encode()).hexdigest()
    assert digest == ORIGINAL_QUESTIONS_SHA256
    assert all(q.style == "heldout" for q in questions[len(originals) :])


def test_heldout_wording_differs_from_the_paraphrase_for_the_same_fact():
    for key, topic in TOPICS.items():
        for fact in topic["facts"]:
            heldout = HELDOUT_QUESTIONS[(key, fact.heading)]
            assert heldout != fact.paraphrase_q and heldout != fact.lexical_q


def test_heldout_questions_avoid_most_glossary_phrases():
    # A held-out question that reuses the paraphrase's everyday wording would
    # measure the glossary on text it was written for. Allow incidental overlap
    # (common words such as "delay"), but not the bulk of the matches.
    phrases = [tuple(tokenize(p)) for p in LOGISTICS_GLOSSARY]

    def n_matches(text):
        toks = tokenize(text)
        return sum(any(tuple(toks[i : i + len(p)]) == p for i in range(len(toks))) for p in phrases)

    paraphrase = sum(n_matches(f.paraphrase_q) for t in TOPICS.values() for f in t["facts"])
    heldout = sum(n_matches(q) for q in HELDOUT_QUESTIONS.values())
    assert heldout <= paraphrase // 3, (heldout, paraphrase)


def test_every_question_points_at_a_real_section_holding_the_answer():
    docs, questions = generate()
    by_id = {d.doc_id: d for d in docs}
    for q in questions:
        doc = by_id[q.gold_doc_id]
        assert doc.site_code == q.site_code and doc.topic == q.topic
        section = next(s for s in doc.sections if s.heading == q.gold_section)
        # The number or code in the answer appears in the gold text.
        key = q.answer.split()[1] if q.answer.split()[0] in {"every", "after", "bay", "lanes", "the", "INR"} else q.answer.split()[0]
        assert key in section.text, (q.qid, q.answer, section.text)


def test_same_seed_is_reproducible_and_different_seed_differs():
    a = generate(seed=1)
    b = generate(seed=1)
    c = generate(seed=2)
    assert [d.to_dict() for d in a[0]] == [d.to_dict() for d in b[0]]
    assert [q.to_dict() for q in a[1]] == [q.to_dict() for q in b[1]]
    assert [d.to_dict() for d in a[0]] != [d.to_dict() for d in c[0]]


def test_site_versions_share_headings_and_title_names_the_site():
    docs, _ = generate()
    returns = [d for d in docs if d.topic == "returns"]
    headings = {tuple(s.heading for s in d.sections) for d in returns}
    assert len(headings) == 1
    for d in returns:
        assert d.site_code in d.title


def test_rejects_empty_and_duplicate_sites():
    with pytest.raises(ValueError):
        generate(sites=())
    with pytest.raises(ValueError):
        generate(sites=(Site("X-1", "A"), Site("X-1", "B")))
