"""Command line entry point: ``sop-bench generate`` and ``sop-bench run``.

Examples::

    sop-bench generate --out data
    sop-bench run --data data
    sop-bench run --data data --retriever expanded --site-filter --by topic
    sop-bench run --data data --all --json results.json
    sop-bench run --data data --no-title   # chunks without the document title
    sop-bench run --data data --max-words 40 --overlap 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .bm25 import BM25Retriever
from .chunk import chunk_documents
from .evaluate import Summary, evaluate, summarise, summarise_by
from .expand import ExpandedRetriever
from .generate import SITES, Site
from .io import DataValidationError, load_documents, load_questions, write_benchmark
from .sites import SiteDetector, SiteFilteredRetriever

RETRIEVERS = ("bm25", "expanded")
KS = (1, 3, 5)
DEFAULT_OVERLAP = 20


def default_overlap(max_words: int) -> int:
    """Overlap used when --overlap is not given.

    20 words for the default 120-word chunks, but never more than half a chunk,
    so a smaller --max-words on its own still produces valid windows.
    """
    return max(0, min(DEFAULT_OVERLAP, max_words // 2))


def _sites_from_documents(docs) -> list[Site]:
    """Site codes and cities as they appear in the corpus titles: '... - Delhi (DEL-01)'."""
    sites: dict[str, Site] = {}
    for d in docs:
        city = d.title.rsplit(" - ", 1)[-1].rsplit(" (", 1)[0].strip() if " - " in d.title else d.site_code
        sites.setdefault(d.site_code, Site(d.site_code, city))
    return list(sites.values())


def build_retriever(chunks, name: str, site_filter: bool, sites: Sequence[Site] = SITES):
    if name not in RETRIEVERS:
        raise ValueError(f"unknown retriever '{name}', choose from {RETRIEVERS}")
    r = BM25Retriever(chunks)
    if name == "expanded":
        r = ExpandedRetriever(r)
    if site_filter:
        r = SiteFilteredRetriever(r, SiteDetector(sites))
    return r


def _label(name: str, site_filter: bool) -> str:
    return f"{name}+site" if site_filter else name


def _row(label: str, group: str, s: Summary) -> dict:
    return {
        "retriever": label,
        "group": group,
        "n": s.n,
        **{f"recall@{k}": round(v, 3) for k, v in s.recall.items()},
        "mrr": round(s.mrr, 3),
        "site_confusion": round(s.site_confusion, 3),
    }


def format_table(rows: list[dict]) -> str:
    cols = list(rows[0])
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    line = lambda vals: "  ".join(str(v).ljust(widths[c]) for c, v in zip(cols, vals)).rstrip()  # noqa: E731
    return "\n".join([line(cols), line("-" * widths[c] for c in cols), *(line(r.values()) for r in rows)])


def run_benchmark(
    data_dir: Path,
    configs: list[tuple[str, bool]],
    by: str,
    max_words: int,
    include_title: bool = True,
    overlap: int | None = None,
) -> list[dict]:
    if overlap is None:
        overlap = default_overlap(max_words)
    docs = load_documents(data_dir / "sop_documents.jsonl")
    questions = load_questions(data_dir / "questions.jsonl", docs)
    chunks = chunk_documents(docs, max_words=max_words, overlap=overlap, include_title=include_title)
    sites = _sites_from_documents(docs)
    rows = []
    for name, site_filter in configs:
        results = evaluate(build_retriever(chunks, name, site_filter, sites), questions, k=max(KS))
        label = _label(name, site_filter)
        rows.append(_row(label, "all", summarise(results, KS)))
        if by != "none":
            rows.extend(_row(label, g, s) for g, s in summarise_by(results, by, KS).items())
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sop-bench", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="write the synthetic corpus and question set")
    g.add_argument("--out", type=Path, default=Path("data"))
    g.add_argument("--seed", type=int, default=7)

    r = sub.add_parser("run", help="score retrievers on a corpus and question set")
    r.add_argument("--data", type=Path, default=Path("data"))
    r.add_argument("--retriever", choices=RETRIEVERS, default="bm25")
    r.add_argument("--site-filter", action="store_true", help="only rank chunks from the site the question names")
    r.add_argument("--all", action="store_true", help="compare every retriever with and without the site filter")
    r.add_argument("--by", choices=("style", "topic", "none"), default="style")
    r.add_argument("--max-words", type=int, default=120, help="chunk size in words")
    r.add_argument(
        "--overlap",
        type=int,
        help="words repeated between neighbouring chunks of a long section "
        f"(default: {DEFAULT_OVERLAP}, capped at half of --max-words)",
    )
    r.add_argument(
        "--no-title",
        action="store_true",
        help="index chunks without the document title (shows how much the site name in the title matters)",
    )
    r.add_argument("--json", type=Path, help="also write the rows to this JSON file")

    args = p.parse_args(argv)
    if args.command == "generate":
        docs_path, q_path = write_benchmark(args.out, seed=args.seed)
        print(f"wrote {docs_path} and {q_path}")
        return 0

    configs = [(n, f) for n in RETRIEVERS for f in (False, True)] if args.all else [(args.retriever, args.site_filter)]
    try:
        rows = run_benchmark(
            args.data, configs, args.by, args.max_words, include_title=not args.no_title, overlap=args.overlap
        )
    except (FileNotFoundError, DataValidationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(format_table(rows))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
