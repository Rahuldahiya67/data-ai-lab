# SOP Retrieval Benchmark

An offline benchmark that measures whether search over site-specific warehouse SOPs brings back the right clause for the right site, and compares a few common retrieval fixes on that test.

## The problem

Warehouse and transport teams keep their operating rules in standard operating procedures (SOPs): cold-chain limits, hazmat storage, return windows, claim deadlines, dock booking rules. The same SOP usually exists in a slightly different version for every site, so "what is the return window?" has one answer in Delhi and another in Mumbai.

Teams that put a search box or an LLM assistant over these documents rarely measure whether retrieval returns the right section. When it doesn't, the assistant answers confidently from the wrong section, or with another site's rule. This project builds a labelled test set for exactly that, and uses it to check three things people usually add: query expansion with a synonym glossary, a site filter on metadata, and the document title in every chunk.

## How it works

```mermaid
flowchart LR
    G[generate<br/>7 topics x 5 sites<br/>+ 3 questions per fact] --> D[(sop_documents.jsonl<br/>questions.jsonl)]
    D --> L[load + validate]
    L --> C[chunk<br/>by section, title header]
    C --> B[BM25]
    B --> X[query expansion<br/>optional]
    X --> F[site filter<br/>optional]
    F --> E[evaluate<br/>recall@k, MRR,<br/>site confusion]
```

1. **Generate** (`generate.py`). 7 SOP topics x 5 sites = 35 documents, 3 facts per document. Each site's version has its own numbers but almost the same wording, which is what makes the wrong-site error likely. Every fact gets three questions, each with a known gold document, section and answer:
   - `lexical`: uses the SOP's own terms ("How long must reefer trailers be pre-cooled before loading at Delhi?").
   - `paraphrase`: everyday wording ("How much time should a refrigerated truck run its cooling before we put stock on it at Delhi?").
   - `heldout`: a second paraphrase, written after the expansion glossary was frozen, so the glossary can be judged on wording it was not written for.

   Questions refer to sites by code or city in four different ways.
2. **Load** (`io.py`). JSON Lines with schema checks. Errors give the file and line; duplicate IDs, duplicate section headings and questions pointing at a missing section are rejected.
3. **Chunk** (`chunk.py`). One chunk per section (split above 120 words with a 20-word overlap; `--max-words` and `--overlap` change both, and without `--overlap` the overlap is capped at half a chunk). Each chunk starts with the document title and section heading, for example `Customer Returns Processing SOP - Delhi (DEL-01). Return window.`
4. **Retrieve**.
   - `bm25.py`: BM25 (k1 = 1.5, b = 0.75) in plain Python with a small suffix stemmer.
   - `expand.py`: appends SOP terms for everyday phrases from a logistics glossary ("transporter" -> "carrier"). The original words are kept.
   - `sites.py`: detects the site a question names (code or city, longest match first) and drops other sites' chunks after ranking. Questions that name no site, or several, are not filtered.
5. **Evaluate** (`evaluate.py`). A hit counts as correct only if it is the gold document and the gold section.
   - recall@k: share of questions with a correct chunk in the top k.
   - MRR: mean of 1 / rank of the first correct chunk (0 if not in the top 5).
   - site confusion: share of questions whose top hit is the right section of another site's document.

## How to run

```bash
cd projects/sop-retrieval-benchmark
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q

sop-bench generate --out data                 # rewrites the committed sample data (seed 7)
sop-bench run --data data --all --by style    # all four retrievers, per question style
sop-bench run --data data --all --by none --no-title
sop-bench run --data data --retriever expanded --site-filter --by topic --json results.json
sop-bench run --data data --by none --max-words 40 --overlap 5   # smaller chunks
```

## Results

All numbers below come from the committed sample data (seed 7, 35 documents, 315 questions, 105 per style).

`sop-bench run --data data --all --by style`, recall@5 and MRR:

| Retriever | lexical | paraphrase | heldout | all (MRR) |
|---|---|---|---|---|
| bm25 | 1.000 | 0.476 | 0.695 | 0.724 (0.583) |
| bm25 + site filter | 1.000 | 0.543 | 0.762 | 0.768 (0.597) |
| expanded | 1.000 | 0.905 | 0.600 | 0.835 (0.761) |
| expanded + site filter | 1.000 | 0.952 | 0.762 | 0.905 (0.792) |

Recall@1 on the held-out questions: bm25 0.419, expanded 0.324.

`sop-bench run --data data --all --by none --no-title` (chunks without the document title):

| Retriever | recall@1 | recall@5 | MRR | site confusion |
|---|---|---|---|---|
| bm25 | 0.225 | 0.498 | 0.315 | 0.203 |
| bm25 + site filter | 0.429 | 0.648 | 0.508 | 0.000 |
| expanded | 0.352 | 0.698 | 0.463 | 0.330 |
| expanded + site filter | 0.683 | 0.832 | 0.742 | 0.000 |

With titles kept in the chunks, site confusion is 0.000 in every configuration.

### What the numbers say

- **Keyword questions are easy; wording is the real gap.** BM25 finds every lexical question in the top 5, but fewer than half of the paraphrased ones.
- **The glossary result does not generalise.** Expansion lifts paraphrase recall@5 from 0.476 to 0.905, but those are the questions the glossary was written against. On the held-out wording it lowers recall@5 from 0.695 to 0.600. Expansion fires on only 15 of the 105 held-out questions and makes 10 of them worse: one-word entries match out of context ("sit-down lift **truck**" gets "trailer" and lands on reefer pre-cooling; "**cold** lorry" gets "temperature chilled" and lands on the temperature range section). A glossary only scores well on the questions it was written from, so it needs its own held-out check, and phrase entries are safer than single words.
- **The title header does most of the site work.** With the site name in every chunk, BM25 never ranks another site's version first. Without it, plain BM25 gives the right rule from the wrong site for 20% of questions, and expansion makes that worse (33%) because it adds topic words, not site words.
- **The site filter is cheap and always helped here.** It never lowered a metric, it brings site confusion to zero when titles are missing, and with titles it still adds 0.04 to 0.16 recall@5 by clearing other sites' chunks out of the top 5. It only helps when the question names the site, which every question here does.

## Design decisions

- **BM25 instead of embeddings.** The point is the benchmark, not the retriever. BM25 has no model download, no API key and gives the same result on every machine, which keeps the numbers in this README reproducible from a clean install. A dense retriever can be added behind the same `search(query, k)` interface and scored with the same code.
- **Gold labels at section level.** "Right document, wrong section" is still a wrong answer for an SOP assistant, so recall only counts the gold section.
- **Held-out questions use a separate random stream.** They were added after the glossary, and a checksum test pins the original 210 questions, so adding them changed nothing in the earlier comparison.
- **The glossary was not tuned after the held-out results.** Fixing "truck" or "cold" now would just move the overfit to the held-out set. A test fails if held-out recall goes up with expansion, as a reminder.
- **Filter after ranking.** The site filter ranks the whole index and then drops other sites, so scores and IDF are identical with and without the filter; only the candidate set changes.
- **Not built:** extractive answers with citations (in the original plan) were left out to keep the three days on measurement. The question set already stores the expected answer for each question, so an answer step can be scored later.

## Data

All documents, sites, numbers and questions are synthetic, produced by `generate.py`. The site codes and cities are used as labels only; the rules do not describe any real warehouse or company.

## Tests

91 tests (`pytest -q`): tokenizer and stemmer, generator (including the pinned original questions and held-out coverage), loader validation, chunking, BM25 scoring and parameter validation, metrics and their edge cases, site detection and filtering, query expansion, and the CLI end to end on the committed data (including chunk size and overlap options).
