"""Re-rate carrier invoices against the rate that was valid on the ship date.

This is the question the SCD Type 2 history exists to answer: for each invoice,
which contracted rate applied when the shipment moved, what should the carrier
have billed, and how far off is the billed amount?

All invoices are matched to their rate version in one range join, then priced
with ``RateVersion.charge`` so that the report and the lookup API agree on
rounding and minimum charges.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import duckdb

from .lookup import RateVersion

STATUSES = ("ok", "overbilled", "underbilled", "no_rate", "currency_mismatch")


@dataclass(frozen=True)
class Invoice:
    invoice_id: str
    lane_id: str
    ship_ts: datetime
    weight_kg: float
    billed_amount: float
    currency: str


@dataclass(frozen=True)
class RerateLine:
    invoice: Invoice
    status: str
    expected_amount: float | None
    difference: float | None  # billed minus expected
    rate_per_kg: float | None
    rate_valid_from: datetime | None


@dataclass
class RerateReport:
    lines: list[RerateLine]
    invalid: list[tuple[int, str]] = field(default_factory=list)  # (line number, reason)

    def count(self, status: str) -> int:
        return sum(1 for line in self.lines if line.status == status)

    def totals_by_currency(self) -> dict[str, dict[str, float]]:
        """Money totals per currency.

        Amounts in different currencies cannot be added, so every total is kept
        per currency: overbilled and underbilled amounts in the currency of the
        matched rate, amounts billed with no contracted rate in the invoice's
        own currency. Currency mismatches carry no total because the billed and
        expected amounts are in different currencies.
        """
        totals: dict[str, dict[str, float]] = {}

        def add(currency: str, key: str, amount: float) -> None:
            bucket = totals.setdefault(
                currency, {"overbilled": 0.0, "underbilled": 0.0, "no_rate_billed": 0.0}
            )
            bucket[key] += amount

        for line in self.lines:
            if line.status == "overbilled":
                add(line.invoice.currency, "overbilled", line.difference)
            elif line.status == "underbilled":
                add(line.invoice.currency, "underbilled", -line.difference)
            elif line.status == "no_rate":
                add(line.invoice.currency, "no_rate_billed", line.invoice.billed_amount)
        return {
            currency: {key: round(value, 2) for key, value in bucket.items()}
            for currency, bucket in sorted(totals.items())
        }

    def _single_currency_total(self, key: str) -> float:
        totals = self.totals_by_currency()
        if len(totals) > 1:
            raise ValueError(
                f"report mixes currencies ({', '.join(totals)}); use totals_by_currency()"
            )
        return next(iter(totals.values()))[key] if totals else 0.0

    @property
    def overbilled_total(self) -> float:
        return self._single_currency_total("overbilled")

    @property
    def underbilled_total(self) -> float:
        return self._single_currency_total("underbilled")

    @property
    def no_rate_billed_total(self) -> float:
        return self._single_currency_total("no_rate_billed")

    def summary(self) -> dict:
        out: dict = {status: self.count(status) for status in STATUSES}
        out["invalid"] = len(self.invalid)
        out["totals_by_currency"] = self.totals_by_currency()
        return out


class InvalidInvoice(ValueError):
    pass


def _number(row: dict, name: str) -> float:
    raw = (row.get(name) or "").strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise InvalidInvoice(f"{name} is not a number: {raw!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise InvalidInvoice(f"{name} must be a positive finite number, got {raw}")
    return value


def parse_invoice(row: dict) -> Invoice:
    invoice_id = (row.get("invoice_id") or "").strip()
    lane_id = (row.get("lane_id") or "").strip()
    currency = (row.get("currency") or "").strip()
    if not invoice_id or not lane_id or not currency:
        raise InvalidInvoice("invoice_id, lane_id and currency are required")
    raw_ts = (row.get("ship_ts") or "").strip()
    try:
        ship_ts = datetime.fromisoformat(raw_ts)
    except ValueError as exc:
        raise InvalidInvoice(f"ship_ts is not ISO-8601: {raw_ts!r}") from exc
    if ship_ts.tzinfo is not None:
        raise InvalidInvoice("ship_ts must be naive UTC (no offset)")
    return Invoice(
        invoice_id=invoice_id,
        lane_id=lane_id,
        ship_ts=ship_ts,
        weight_kg=_number(row, "weight_kg"),
        billed_amount=_number(row, "billed_amount"),
        currency=currency,
    )


def read_invoices(path: Path) -> tuple[list[Invoice], list[tuple[int, str]]]:
    """Read an invoice CSV. Bad rows are returned with their line number, not raised."""
    invoices, invalid = [], []
    seen: set[str] = set()
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for line_no, row in enumerate(reader, start=2):
            try:
                invoice = parse_invoice(row)
            except InvalidInvoice as exc:
                invalid.append((line_no, str(exc)))
                continue
            if invoice.invoice_id in seen:
                invalid.append((line_no, f"duplicate invoice_id {invoice.invoice_id}"))
                continue
            seen.add(invoice.invoice_id)
            invoices.append(invoice)
    return invoices, invalid


_MATCH = """
SELECT i.idx, d.lane_id, d.carrier, d.rate_per_kg, d.min_charge, d.currency,
       d.valid_from, d.valid_to, d.source_lsn
FROM stg_invoices i
JOIN dim_rate_card d
  ON d.lane_id = i.lane_id AND d.valid_from <= i.ship_ts AND i.ship_ts < d.valid_to
"""


def rerate(
    con: duckdb.DuckDBPyConnection,
    invoices: list[Invoice],
    tolerance: float = 0.01,
    invalid: list[tuple[int, str]] | None = None,
) -> RerateReport:
    """Price every invoice at the rate valid on its ship date and compare."""
    if tolerance < 0:
        raise ValueError(f"tolerance must be >= 0, got {tolerance}")
    con.execute("CREATE OR REPLACE TEMP TABLE stg_invoices (idx INTEGER, lane_id VARCHAR, ship_ts TIMESTAMP)")
    if invoices:
        con.executemany(
            "INSERT INTO stg_invoices VALUES (?, ?, ?)",
            [[i, inv.lane_id, inv.ship_ts] for i, inv in enumerate(invoices)],
        )
    matched = {row[0]: RateVersion(*row[1:]) for row in con.execute(_MATCH).fetchall()}
    lines = []
    for idx, invoice in enumerate(invoices):
        version = matched.get(idx)
        if version is None:
            lines.append(RerateLine(invoice, "no_rate", None, None, None, None))
            continue
        expected = version.charge(invoice.weight_kg)
        difference = round(invoice.billed_amount - expected, 2)
        if invoice.currency != version.currency:
            status = "currency_mismatch"
        elif abs(difference) <= tolerance:
            status = "ok"
        elif difference > 0:
            status = "overbilled"
        else:
            status = "underbilled"
        lines.append(RerateLine(invoice, status, expected, difference, version.rate_per_kg, version.valid_from))
    return RerateReport(lines=lines, invalid=list(invalid or []))


def write_report(report: RerateReport, path: Path) -> None:
    """Write one row per invoice that needs attention (everything except ``ok``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["invoice_id", "lane_id", "ship_ts", "weight_kg", "billed_amount", "expected_amount",
             "difference", "status", "rate_per_kg", "rate_valid_from"]
        )
        for line in report.lines:
            if line.status == "ok":
                continue
            inv = line.invoice
            writer.writerow(
                [inv.invoice_id, inv.lane_id, inv.ship_ts.isoformat(), inv.weight_kg, inv.billed_amount,
                 line.expected_amount, line.difference, line.status, line.rate_per_kg,
                 line.rate_valid_from.isoformat() if line.rate_valid_from else ""]
            )
