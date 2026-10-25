"""Text normalisation shared by the chunker and the retrievers."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")

# Small English stopword list. Kept short on purpose: words such as "before",
# "after", "within" and "not" carry meaning in SOP rules and are kept.
STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the
    their this to was were which will with what how many much does do""".split()
)


def _stem(token: str) -> str:
    """Very light suffix stripping so 'pallets'/'pallet' and 'loading'/'load' match.

    Numbers and codes (anything containing a digit) are returned unchanged.
    """
    if any(ch.isdigit() for ch in token) or len(token) <= 3:
        return token
    for suffix in ("ing", "ed", "es", "s"):
        if suffix == "s" and token.endswith("ss"):
            # "process", "glass", "access": the final s is part of the word.
            return token
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            stem = token[: -len(suffix)]
            if suffix == "es" and not stem.endswith(("s", "x", "z", "ch", "sh")):
                # "litres" -> "litre", not "litr"
                return token[:-1]
            return stem
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase, split into word/number tokens, drop stopwords, light-stem.

    Site codes such as ``BOM-02`` and decimals such as ``2.5`` stay one token.
    """
    if not isinstance(text, str):
        raise TypeError(f"tokenize expects str, got {type(text).__name__}")
    tokens = _TOKEN_RE.findall(text.lower())
    return [_stem(t) for t in tokens if t not in STOPWORDS]
