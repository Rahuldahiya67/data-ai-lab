# Data & AI Lab

Small, complete projects on what's current in **data engineering, analytics and applied AI**, one at a time. Each project solves one concrete problem end to end, runs offline with sample data, and has its own tests and README.

Main portfolio: [rahuldahiya67.github.io](https://rahuldahiya67.github.io)

## Projects

<!-- PROJECT-INDEX:START -->
| Project | What it shows | Stack | Completed |
|---|---|---|---|
| [Carrier Feed Contract Checker](projects/carrier-feed-contract-checker) | Validates carrier shipment feeds against a YAML data contract, quarantines bad rows and gates the load on reject rate | Python, pandas, PyYAML | 17 Oct 2026 |
| [Intermittent Demand Forecaster](projects/intermittent-demand-forecaster) | Classifies spare-parts demand, backtests Croston, SBA and TSB against standard methods, picks one per SKU and sets reorder points with a service-level check | Python, pandas, NumPy | 20 Oct 2026 |
| [SOP Retrieval Benchmark](projects/sop-retrieval-benchmark) | Measures whether search over site-specific warehouse SOPs returns the right clause for the right site; tests query expansion on held-out wording, a site filter and title headers | Python (BM25 from scratch) | 26 Oct 2026 |
| [Rate Card History Loader](projects/rate-card-history-loader) | Loads an at-least-once CDC stream of carrier rate changes into an SCD Type 2 table with a watermark, ledger and quarantine, then re-rates invoices at the rate valid on the ship date | Python, DuckDB | 29 Oct 2026 |
| [Delivery Exception Triage Agent](projects/delivery-exception-triage-agent) | Tool-calling agent for delivery exception tickets with typed tools, policy guardrails checked before every action, and an evaluation of action, reason and tool-call accuracy on labelled tickets | Python | 30 Oct 2026 |
<!-- PROJECT-INDEX:END -->

## How every project is laid out

```
projects/<project-name>/          e.g. projects/freight-invoice-auditor/
  README.md        problem, how it works, how to run, results
  pyproject.toml   or requirements.txt
  src/             code
  tests/           pytest, must pass offline
  data/            small sample data (synthetic)
```

Each project installs and runs on its own:

```bash
cd projects/<project-name>
pip install -e ".[dev]"      # or: pip install -r requirements.txt
pytest -q
```

CI runs every project's tests on each push.
