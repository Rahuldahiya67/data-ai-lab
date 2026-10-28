"""Site-aware retrieval: detect which site a question is about and keep only
that site's chunks.

Every site has its own copy of each SOP, so the most damaging retrieval error is
the right rule from the wrong site. When the question names a site (by code or
by city), there is no reason to rank other sites' chunks at all. This is the
metadata filter that most vector stores offer, written out in plain Python.

If a question names no site, or names more than one, nothing is filtered and the
wrapped retriever's ranking is returned unchanged.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from .bm25 import Hit
from .chunk import Chunk
from .generate import Site
from .text import tokenize


class _SearchableIndex(Protocol):
    chunks: list[Chunk]

    def search(self, query: str, k: int = 5) -> list[Hit]: ...


class SiteDetector:
    """Find site codes mentioned in a question by code or city name."""

    def __init__(self, sites: Sequence[Site]):
        if not sites:
            raise ValueError("at least one site is required")
        self._aliases: list[tuple[tuple[str, ...], str]] = []
        for site in sites:
            for alias in (site.code, site.city):
                tokens = tuple(tokenize(alias))
                if not tokens:
                    raise ValueError(f"site alias '{alias}' has no searchable words")
                for other, code in self._aliases:
                    if other == tokens and code != site.code:
                        raise ValueError(f"alias '{alias}' is used by both {code} and {site.code}")
                self._aliases.append((tokens, site.code))

    def detect(self, query: str) -> set[str]:
        """Site codes named in ``query``.

        Matching is greedy longest-first over tokens, so "New York" is one
        mention of New York and not also a mention of York.
        """
        tokens = tokenize(query)
        by_length = sorted(self._aliases, key=lambda a: -len(a[0]))
        found: set[str] = set()
        i = 0
        while i < len(tokens):
            for alias, code in by_length:
                if tuple(tokens[i : i + len(alias)]) == alias:
                    found.add(code)
                    i += len(alias)
                    break
            else:
                i += 1
        return found


class SiteFilteredRetriever:
    """Wrap a retriever and drop hits from sites the question does not name."""

    def __init__(self, base: _SearchableIndex, detector: SiteDetector):
        self.base = base
        self.detector = detector
        self.chunks = base.chunks

    def search(self, query: str, k: int = 5) -> list[Hit]:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        sites = self.detector.detect(query)
        if len(sites) != 1:
            return self.base.search(query, k=k)
        # Rank the whole index, then filter, so scores (and IDF) are the same
        # as without the filter; only other sites' chunks are removed.
        hits = self.base.search(query, k=len(self.base.chunks))
        return [h for h in hits if h.chunk.site_code in sites][:k]
