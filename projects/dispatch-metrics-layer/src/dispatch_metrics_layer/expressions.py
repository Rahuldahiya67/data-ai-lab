"""Safe SQL fragments for measure expressions and filters.

Measure ``expr`` and ``filter`` values are written by people into the spec and
pasted into generated SQL. To keep that safe and reviewable, a fragment may only
contain:

* column names of the fact table,
* numbers and single-quoted strings,
* comparison and arithmetic operators and parentheses,
* a short list of keywords (AND, OR, NOT, IS, NULL, IN, BETWEEN, CASE ...),
* a short list of functions (COALESCE, ABS, LEAST, GREATEST, ROUND, DATE_DIFF).

Anything else (semicolons, comments, unknown functions, other tables' columns)
is rejected with a message that says which token is the problem.
"""

from __future__ import annotations

import re

KEYWORDS = {
    "and", "or", "not", "is", "null", "in", "between", "case", "when", "then",
    "else", "end", "true", "false", "like",
}
FUNCTIONS = {"coalesce", "abs", "least", "greatest", "round", "date_diff"}

_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<string>'(?:[^']|'')*')
  | (?P<number>\d+(?:\.\d+)?)
  | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<op><=|>=|<>|!=|=|<|>|\+|-|\*|/|%)
  | (?P<paren>[(),])
    """,
    re.VERBOSE,
)


def tokenize(fragment: str) -> list[tuple[str, str]]:
    """Split a fragment into ``(kind, text)`` tokens. Raises ValueError on anything unknown."""
    if "--" in fragment or "/*" in fragment:
        raise ValueError("comments are not allowed")
    tokens = []
    pos = 0
    while pos < len(fragment):
        m = _TOKEN_RE.match(fragment, pos)
        if m is None:
            raise ValueError(f"unexpected character {fragment[pos]!r} at position {pos}")
        kind = m.lastgroup
        if kind != "ws":
            tokens.append((kind, m.group()))
        pos = m.end()
    return tokens


def check_fragment(fragment: str, columns: set[str]) -> list[str]:
    """Return problems with ``fragment`` (empty list if it is safe).

    ``columns`` are the column names the fragment may reference.
    """
    try:
        tokens = tokenize(fragment)
    except ValueError as exc:
        return [str(exc)]
    problems = []
    depth = 0
    for i, (kind, text) in enumerate(tokens):
        if kind == "paren":
            depth += {"(": 1, ")": -1}.get(text, 0)
            if depth < 0:
                problems.append("unbalanced parentheses")
                depth = 0
        if kind != "ident":
            continue
        low = text.lower()
        followed_by_paren = i + 1 < len(tokens) and tokens[i + 1] == ("paren", "(")
        if followed_by_paren:
            if low not in FUNCTIONS:
                problems.append(f"function {text!r} is not allowed (allowed: {sorted(FUNCTIONS)})")
        elif low in KEYWORDS:
            continue
        elif text not in columns:
            problems.append(f"unknown column {text!r}")
    if depth != 0:
        problems.append("unbalanced parentheses")
    return problems
