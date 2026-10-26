"""Glossary-based query expansion for the vocabulary gap.

People ask about a "refrigerated truck"; the SOP says "reefer trailer". Keyword
search cannot connect the two, which is most of the gap between keyword-style
and paraphrased questions in this benchmark. The common fix in production
search (Elasticsearch/OpenSearch synonym files, Solr synonym filters) is a
maintained glossary that maps everyday wording to the document vocabulary.

``QueryExpander`` appends the document terms for every glossary phrase found in
the question; the original words are kept, so expansion can add matches but
never removes one.

Caveat for reading results: ``LOGISTICS_GLOSSARY`` was written for this corpus,
by someone who had seen the question set. Its gain on the paraphrased questions
is therefore an upper bound on what a glossary buys, not a generalisation
result. The held-out questions (style ``heldout``, written after this glossary
was frozen) are the fair check: on them expansion gains nothing and costs
recall, because one-word entries such as "truck" and "cold" also fire in
"lift truck" and "cold lorry" and pull the query towards the wrong section.
"""

from __future__ import annotations

from typing import Mapping

from .bm25 import Hit
from .sites import _SearchableIndex
from .text import tokenize

# everyday phrase -> SOP vocabulary. Both sides are tokenised (and stemmed) the
# same way as the documents, so hyphenated words such as "sent-back" are written
# with the hyphen: the tokenizer keeps them as one token.
LOGISTICS_GLOSSARY: dict[str, str] = {
    # cold chain
    "refrigerated": "reefer chilled",
    "truck": "trailer",
    "medicine": "pharmaceutical",
    "cold": "temperature chilled",
    "out of range": "temperature excursion",
    "on hold": "quarantine quality hold",
    # returns
    "goods back": "return",
    "sent-back": "return returned",
    "buyer": "customer",
    "money back": "refund",
    "classify": "grade",
    "inspect": "grade inspected",
    # hazmat
    "petrol-type": "flammable liquid class 3",
    "fire-risk": "flammable",
    "oxidizing": "oxidiser",
    "storage position": "rack bay",
    "leak": "spill",
    "clean-up equipment": "spill kit",
    "gap": "distance metre",
    # claims
    "transporter": "carrier",
    "haulier": "carrier",
    "trucking company": "carrier",
    "broken": "damage damaged",
    "complain": "claim filed",
    "pictures": "photo",
    "sign off": "approval",
    "money people": "finance controller",
    # docks
    "reserve": "book",
    "unloading time": "dock slot",
    "booked time": "dock slot",
    "waiting fee": "detention charge",
    "delay": "late",
    # inventory
    "stock check": "cycle count",
    "fast-moving": "a-class sku",
    "mismatch": "variance",
    "count again": "recount",
    # cross-dock
    "transfer stock": "cross-dock freight",
    "hand jack": "counterbalance forklift",
    "mass": "weight heavier kg",
    "line up": "staged lane",
}


class QueryExpander:
    def __init__(self, glossary: Mapping[str, str] = LOGISTICS_GLOSSARY):
        if not glossary:
            raise ValueError("glossary is empty")
        self._entries: list[tuple[tuple[str, ...], str]] = []
        for phrase, expansion in glossary.items():
            tokens = tuple(tokenize(phrase))
            if not tokens:
                raise ValueError(f"glossary phrase '{phrase}' has no searchable words")
            if not tokenize(expansion):
                raise ValueError(f"glossary expansion for '{phrase}' has no searchable words")
            self._entries.append((tokens, expansion))

    def matches(self, query: str) -> list[str]:
        """Expansions whose phrase occurs in ``query``, in glossary order."""
        tokens = tokenize(query)
        out = []
        for phrase, expansion in self._entries:
            n = len(phrase)
            if any(tuple(tokens[i : i + n]) == phrase for i in range(len(tokens) - n + 1)):
                out.append(expansion)
        return out

    def expand(self, query: str) -> str:
        extra = self.matches(query)
        return f"{query} {' '.join(extra)}" if extra else query


class ExpandedRetriever:
    """Wrap a retriever and search with the expanded query."""

    def __init__(self, base: _SearchableIndex, expander: QueryExpander | None = None):
        self.base = base
        self.expander = expander or QueryExpander()
        self.chunks = base.chunks

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return self.base.search(self.expander.expand(query), k=k)
