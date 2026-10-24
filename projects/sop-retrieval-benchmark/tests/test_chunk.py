import pytest

from sop_retrieval_benchmark.chunk import chunk_documents
from sop_retrieval_benchmark.generate import Document, Section, generate


def _doc(text, heading="Rule"):
    return Document("d1", "Returns SOP - Delhi (DEL-01)", "returns", "DEL-01", [Section(heading, text)])


def test_one_chunk_per_short_section_with_title_header():
    docs, _ = generate()
    chunks = chunk_documents(docs)
    assert len(chunks) == sum(len(d.sections) for d in docs)
    c = chunks[0]
    assert c.indexed_text.startswith(docs[0].title)
    assert c.section in c.indexed_text


def test_title_can_be_left_out():
    c = chunk_documents([_doc("keep it cold")], include_title=False)[0]
    assert c.indexed_text == "Rule. keep it cold"


def test_long_section_is_windowed_with_overlap():
    words = [f"w{i}" for i in range(25)]
    chunks = chunk_documents([_doc(" ".join(words))], max_words=10, overlap=2)
    texts = [c.text.split() for c in chunks]
    assert texts[0] == words[0:10]
    assert texts[1][:2] == words[8:10]
    # every word is covered and the last window ends at the last word
    assert texts[-1][-1] == "w24"
    assert {w for t in texts for w in t} == set(words)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_exact_fit_does_not_create_a_tail_chunk():
    words = " ".join(f"w{i}" for i in range(10))
    assert len(chunk_documents([_doc(words)], max_words=10, overlap=2)) == 1


@pytest.mark.parametrize("max_words,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_window_settings(max_words, overlap):
    with pytest.raises(ValueError):
        chunk_documents([_doc("x")], max_words=max_words, overlap=overlap)
