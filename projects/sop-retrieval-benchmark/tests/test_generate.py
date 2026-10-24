from collections import Counter

import pytest

from sop_retrieval_benchmark.generate import SITES, TOPICS, Site, generate


def test_one_document_per_topic_and_site():
    docs, _ = generate()
    assert len(docs) == len(TOPICS) * len(SITES)
    assert len({d.doc_id for d in docs}) == len(docs)


def test_two_questions_per_fact_with_both_styles():
    docs, questions = generate()
    n_facts = sum(len(t["facts"]) for t in TOPICS.values()) * len(SITES)
    assert len(questions) == 2 * n_facts
    assert Counter(q.style for q in questions) == {"lexical": n_facts, "paraphrase": n_facts}
    assert len({q.qid for q in questions}) == len(questions)


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
