"""Load and validate a YAML data contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

COLUMN_TYPES = {"string", "int", "float", "datetime"}
COMPARE_OPS = {"<", "<=", ">", ">=", "==", "!="}
RULE_TYPES = {"compare", "required_if"}


class ContractError(ValueError):
    """Raised when a contract file is malformed."""


@dataclass
class ColumnSpec:
    name: str
    type: str = "string"
    required: bool = False
    allowed: list[str] | None = None
    regex: str | None = None
    min: float | None = None
    max: float | None = None
    unique: bool = False
    format: str | None = None


@dataclass
class RuleSpec:
    name: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedSpec:
    min_rows: int = 0
    max_reject_rate: float = 1.0


@dataclass
class Contract:
    name: str
    version: int
    columns: dict[str, ColumnSpec]
    rules: list[RuleSpec]
    feed: FeedSpec


def _column(name: str, raw: dict[str, Any]) -> ColumnSpec:
    if not isinstance(raw, dict):
        raise ContractError(f"column '{name}' must be a mapping")
    unknown = set(raw) - (set(ColumnSpec.__dataclass_fields__) - {"name"})
    if unknown:
        raise ContractError(f"column '{name}' has unknown keys: {sorted(unknown)}")
    spec = ColumnSpec(name=name, **raw)
    if spec.type not in COLUMN_TYPES:
        raise ContractError(f"column '{name}' has unsupported type '{spec.type}'")
    if (spec.min is not None or spec.max is not None) and spec.type not in {"int", "float"}:
        raise ContractError(f"column '{name}': min/max only apply to numeric columns")
    if spec.allowed is not None:
        spec.allowed = [str(v) for v in spec.allowed]
    return spec


def _rule(raw: dict[str, Any], columns: dict[str, ColumnSpec]) -> RuleSpec:
    if not isinstance(raw, dict) or "name" not in raw or "type" not in raw:
        raise ContractError("each rule needs a 'name' and a 'type'")
    params = {k: v for k, v in raw.items() if k not in {"name", "type"}}
    rule = RuleSpec(name=raw["name"], type=raw["type"], params=params)
    if rule.type not in RULE_TYPES:
        raise ContractError(f"rule '{rule.name}' has unsupported type '{rule.type}'")
    if rule.type == "compare":
        for key in ("left", "op", "right"):
            if key not in params:
                raise ContractError(f"rule '{rule.name}' is missing '{key}'")
        if params["op"] not in COMPARE_OPS:
            raise ContractError(f"rule '{rule.name}' has unsupported op '{params['op']}'")
        referenced = [params["left"], params["right"]]
    else:
        for key in ("column", "when_column", "equals"):
            if key not in params:
                raise ContractError(f"rule '{rule.name}' is missing '{key}'")
        referenced = [params["column"], params["when_column"]]
    for col in referenced:
        if col not in columns:
            raise ContractError(f"rule '{rule.name}' refers to unknown column '{col}'")
    return rule


def parse_contract(raw: dict[str, Any]) -> Contract:
    """Build a :class:`Contract` from an already-loaded mapping."""
    if not isinstance(raw, dict):
        raise ContractError("contract must be a mapping")
    if not raw.get("columns"):
        raise ContractError("contract must define at least one column")
    columns = {name: _column(name, spec or {}) for name, spec in raw["columns"].items()}
    rules = [_rule(r, columns) for r in raw.get("rules") or []]
    feed_raw = raw.get("feed") or {}
    feed = FeedSpec(
        min_rows=int(feed_raw.get("min_rows", 0)),
        max_reject_rate=float(feed_raw.get("max_reject_rate", 1.0)),
    )
    if not 0.0 <= feed.max_reject_rate <= 1.0:
        raise ContractError("feed.max_reject_rate must be between 0 and 1")
    return Contract(
        name=str(raw.get("name", "unnamed")),
        version=int(raw.get("version", 1)),
        columns=columns,
        rules=rules,
        feed=feed,
    )


def load_contract(path: str | Path) -> Contract:
    """Read a YAML contract from disk."""
    with open(path, encoding="utf-8") as fh:
        return parse_contract(yaml.safe_load(fh))
