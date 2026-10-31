"""Read and write the shipment and ticket files, with input validation.

Files:
- shipments.json: a list of shipments, each with its scan history.
- tickets.jsonl: one labelled ticket per line (ticket fields, scenario, expected resolution).

Loading rejects malformed records with a DatasetError that names the record and
field, instead of letting a KeyError or a bad date surface deep inside the agent.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .models import ACTIONS, SHIPMENT_STATUSES, LabelledTicket, Resolution, ScanEvent, Shipment, Ticket


class DatasetError(ValueError):
    """A shipment or ticket record is missing a field or has an invalid value."""


def _shipment_to_dict(s: Shipment) -> dict[str, Any]:
    return {
        "shipment_id": s.shipment_id,
        "carrier": s.carrier,
        "origin_hub": s.origin_hub,
        "destination_city": s.destination_city,
        "promised_date": s.promised_date.isoformat(),
        "status": s.status,
        "declared_value_inr": s.declared_value_inr,
        "address_valid": s.address_valid,
        "scans": [{"at": e.at.isoformat(), "location": e.location, "code": e.code} for e in s.scans],
    }


def _ticket_to_dict(t: LabelledTicket) -> dict[str, Any]:
    exp = t.expected
    return {
        "ticket_id": t.ticket.ticket_id,
        "shipment_id": t.ticket.shipment_id,
        "created_at": t.ticket.created_at.isoformat(),
        "message": t.ticket.message,
        "scenario": t.scenario,
        "expected": {
            "action": exp.action,
            "amount_inr": exp.amount_inr,
            "redelivery_date": exp.redelivery_date.isoformat() if exp.redelivery_date else None,
            "reason": exp.reason,
        },
        "expected_tools": list(t.expected_tools),
    }


def write_dataset(shipments: dict[str, Shipment], tickets: list[LabelledTicket], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = [_shipment_to_dict(s) for s in shipments.values()]
    (out_dir / "shipments.json").write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    with open(out_dir / "tickets.jsonl", "w", encoding="utf-8") as fh:
        for t in tickets:
            fh.write(json.dumps(_ticket_to_dict(t)) + "\n")


def _require(rec: dict, key: str, kind, where: str):
    if key not in rec:
        raise DatasetError(f"{where}: missing field {key!r}")
    value = rec[key]
    if kind is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DatasetError(f"{where}: {key!r} must be a number, got {value!r}")
        return float(value)
    if not isinstance(value, kind):
        raise DatasetError(f"{where}: {key!r} must be {kind.__name__}, got {value!r}")
    return value


def _parse_date(value: str, key: str, where: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        raise DatasetError(f"{where}: {key!r} is not an ISO date: {value!r}") from None


def _parse_datetime(value: str, key: str, where: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        raise DatasetError(f"{where}: {key!r} is not an ISO timestamp: {value!r}") from None


def parse_shipment(rec: dict, where: str = "shipment") -> Shipment:
    if not isinstance(rec, dict):
        raise DatasetError(f"{where}: expected an object, got {type(rec).__name__}")
    sid = _require(rec, "shipment_id", str, where)
    where = f"shipment {sid}"
    status = _require(rec, "status", str, where)
    if status not in SHIPMENT_STATUSES:
        raise DatasetError(f"{where}: unknown status {status!r}")
    value = _require(rec, "declared_value_inr", float, where)
    if not value > 0 or value == float("inf"):
        raise DatasetError(f"{where}: declared_value_inr must be a positive finite number, got {value!r}")
    scans = []
    for i, e in enumerate(_require(rec, "scans", list, where)):
        w = f"{where} scan {i}"
        if not isinstance(e, dict):
            raise DatasetError(f"{w}: expected an object")
        scans.append(
            ScanEvent(
                _parse_datetime(_require(e, "at", str, w), "at", w),
                _require(e, "location", str, w),
                _require(e, "code", str, w),
            )
        )
    if any(b.at < a.at for a, b in zip(scans, scans[1:])):
        raise DatasetError(f"{where}: scans are not in time order")
    return Shipment(
        shipment_id=sid,
        carrier=_require(rec, "carrier", str, where),
        origin_hub=_require(rec, "origin_hub", str, where),
        destination_city=_require(rec, "destination_city", str, where),
        promised_date=_parse_date(_require(rec, "promised_date", str, where), "promised_date", where),
        status=status,
        declared_value_inr=value,
        address_valid=_require(rec, "address_valid", bool, where),
        scans=tuple(scans),
    )


def parse_ticket(rec: dict, where: str = "ticket") -> LabelledTicket:
    if not isinstance(rec, dict):
        raise DatasetError(f"{where}: expected an object, got {type(rec).__name__}")
    tid = _require(rec, "ticket_id", str, where)
    where = f"ticket {tid}"
    ticket = Ticket(
        ticket_id=tid,
        shipment_id=_require(rec, "shipment_id", str, where),
        created_at=_parse_datetime(_require(rec, "created_at", str, where), "created_at", where),
        message=_require(rec, "message", str, where),
    )
    exp = _require(rec, "expected", dict, where)
    action = _require(exp, "action", str, where + " expected")
    if action not in ACTIONS:
        raise DatasetError(f"{where}: unknown expected action {action!r}")
    amount = exp.get("amount_inr")
    if amount is not None:
        amount = _require(exp, "amount_inr", float, where + " expected")
    rd = exp.get("redelivery_date")
    expected = Resolution(
        action=action,
        amount_inr=amount,
        redelivery_date=_parse_date(rd, "redelivery_date", where) if rd is not None else None,
        reason=exp.get("reason") or "",
    )
    return LabelledTicket(
        ticket=ticket,
        scenario=_require(rec, "scenario", str, where),
        expected=expected,
        expected_tools=tuple(rec.get("expected_tools", ())),
    )


def load_shipments(path: Path) -> dict[str, Shipment]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{path}: not valid JSON ({exc})") from None
    if not isinstance(raw, list):
        raise DatasetError(f"{path}: expected a list of shipments")
    out: dict[str, Shipment] = {}
    for i, rec in enumerate(raw):
        s = parse_shipment(rec, where=f"shipment #{i}")
        if s.shipment_id in out:
            raise DatasetError(f"duplicate shipment_id {s.shipment_id}")
        out[s.shipment_id] = s
    return out


def load_tickets(path: Path) -> list[LabelledTicket]:
    out = []
    seen = set()
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path} line {lineno}: not valid JSON ({exc})") from None
            t = parse_ticket(rec, where=f"line {lineno}")
            if t.ticket.ticket_id in seen:
                raise DatasetError(f"duplicate ticket_id {t.ticket.ticket_id}")
            seen.add(t.ticket.ticket_id)
            out.append(t)
    return out
