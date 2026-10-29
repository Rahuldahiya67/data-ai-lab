"""Synthetic carrier invoices for testing the re-rating report.

Each invoice bills one shipment on one lane. Most are billed at the rate that
was valid on the ship date. A share are billed with known mistakes that a
freight audit team looks for:

- ``stale_rate``: billed at the rate valid on the invoice date (some days after
  shipping) instead of the ship date, so a rate change in between is applied
  too early
- ``no_contract``: the lane was withdrawn on the ship date, but the carrier
  billed it at the last rate it had

The true error type is kept in ``expected_issue`` so tests can check the report
against it. The re-rating report itself never reads that column.
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

from .events import ChangeEvent

FIELDS = ("invoice_id", "lane_id", "ship_ts", "weight_kg", "billed_amount", "currency", "expected_issue")


def _state_at(history: list[ChangeEvent], at: datetime) -> dict[str, ChangeEvent]:
    state: dict[str, ChangeEvent] = {}
    for event in history:
        if event.commit_ts > at:
            break
        if event.op == "d":
            state.pop(event.lane_id, None)
        else:
            state[event.lane_id] = event
    return state


def _last_version_before(history: list[ChangeEvent], lane_id: str, at: datetime) -> ChangeEvent | None:
    last = None
    for event in history:
        if event.commit_ts > at:
            break
        if event.lane_id == lane_id and event.op != "d":
            last = event
    return last


def _charge(event: ChangeEvent, weight_kg: float) -> float:
    return round(max(weight_kg * event.rate_per_kg, event.min_charge), 2)


def generate_invoices(
    history: list[ChangeEvent],
    n_invoices: int = 400,
    stale_rate_share: float = 0.08,
    seed: int = 21,
) -> list[dict]:
    """Generate invoices for shipments over the span of ``history``."""
    if not history:
        return []
    rng = random.Random(seed)
    start, end = history[0].commit_ts, history[-1].commit_ts
    span = (end - start).total_seconds()
    lanes = sorted({e.lane_id for e in history})
    invoices: list[dict] = []
    while len(invoices) < n_invoices:
        ship_ts = start + timedelta(seconds=rng.uniform(span * 0.05, span * 0.9))
        ship_ts = ship_ts.replace(microsecond=0)
        lane_id = rng.choice(lanes)
        weight = round(rng.choice((rng.uniform(5, 40), rng.uniform(40, 900))), 1)
        live = _state_at(history, ship_ts).get(lane_id)
        issue = ""
        if live is None:
            billed_from = _last_version_before(history, lane_id, ship_ts)
            if billed_from is None:
                continue  # lane not created yet: no carrier would bill it
            issue = "no_contract"
        elif rng.random() < stale_rate_share:
            invoice_ts = ship_ts + timedelta(days=rng.randint(5, 25))
            later = _state_at(history, invoice_ts).get(lane_id)
            if later is None or _charge(later, weight) == _charge(live, weight):
                billed_from = live  # no rate change in between: the mistake is invisible
            else:
                billed_from, issue = later, "stale_rate"
        else:
            billed_from = live
        invoices.append(
            {
                "invoice_id": f"INV-{len(invoices) + 1:05d}",
                "lane_id": lane_id,
                "ship_ts": ship_ts.isoformat(),
                "weight_kg": weight,
                "billed_amount": _charge(billed_from, weight),
                "currency": billed_from.currency,
                "expected_issue": issue,
            }
        )
    return invoices


def write_invoices(invoices: list[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(invoices)
    return len(invoices)
