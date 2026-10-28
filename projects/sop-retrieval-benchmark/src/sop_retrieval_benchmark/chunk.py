"""Split SOP documents into retrievable chunks.

Chunks follow section boundaries, so a chunk never mixes two rules. Long
sections are split into windows of ``max_words`` words with ``overlap`` words
repeated between neighbours. Each chunk's indexed text starts with the
document title and section heading: without it, a section such as "Return
window" carries no site information and every site's version looks the same
to the retriever.
"""

from __future__ import annotations

from dataclasses import dataclass

from .generate import Document


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    section: str
    site_code: str
    topic: str
    text: str  # the section text (or window of it) as written
    indexed_text: str  # what the retriever sees: title + heading + text


def chunk_documents(
    docs: list[Document], max_words: int = 120, overlap: int = 20, include_title: bool = True
) -> list[Chunk]:
    if max_words < 1:
        raise ValueError(f"max_words must be at least 1, got {max_words}")
    if not 0 <= overlap < max_words:
        raise ValueError(f"overlap must be in [0, max_words), got {overlap} with max_words={max_words}")

    chunks: list[Chunk] = []
    step = max_words - overlap
    for doc in docs:
        for s_idx, section in enumerate(doc.sections):
            words = section.text.split()
            if not words:
                continue
            starts = range(0, max(len(words) - overlap, 1), step)
            for part, start in enumerate(starts):
                window = " ".join(words[start : start + max_words])
                header = f"{doc.title}. {section.heading}." if include_title else f"{section.heading}."
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}#s{s_idx}p{part}",
                        doc_id=doc.doc_id,
                        section=section.heading,
                        site_code=doc.site_code,
                        topic=doc.topic,
                        text=window,
                        indexed_text=f"{header} {window}",
                    )
                )
    return chunks
