"""BM25 (Okapi) retriever in plain Python.

No external search engine is needed, so the benchmark runs anywhere. The
corpus here is a few hundred chunks; a dict-based inverted index is fast
enough and easy to read.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .chunk import Chunk
from .text import tokenize


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        if not chunks:
            raise ValueError("cannot build an index over zero chunks")
        if k1 < 0:
            raise ValueError(f"k1 must be >= 0, got {k1}")
        if not 0 <= b <= 1:
            raise ValueError(f"b must be in [0, 1], got {b}")
        ids = [c.chunk_id for c in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("chunk IDs must be unique")

        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._tf: list[Counter[str]] = [Counter(tokenize(c.indexed_text)) for c in chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg_len = sum(self._len) / len(self._len)
        # Length normalisation depends only on the chunk, so compute it once.
        self._norm = [k1 * (1 - b + b * n_i / self._avg_len) for n_i in self._len]
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(chunks)
        # Lucene-style IDF: always positive, even for terms in most chunks.
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self._postings: dict[str, list[int]] = {}
        for i, tf in enumerate(self._tf):
            for term in tf:
                self._postings.setdefault(term, []).append(i)

    def score(self, query: str) -> dict[int, float]:
        scores: dict[int, float] = {}
        for term in set(tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i in self._postings[term]:
                f = self._tf[i][term]
                scores[i] = scores.get(i, 0.0) + idf * f * (self.k1 + 1) / (f + self._norm[i])
        return scores

    def search(self, query: str, k: int = 5) -> list[Hit]:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        scores = self.score(query)
        # Ties are broken by chunk order so results are deterministic.
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [Hit(self.chunks[i], s) for i, s in ranked]
