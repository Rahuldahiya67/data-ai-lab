"""Load and validate a YAML data contract."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

COLUMN_TYPES = {"string", "int", "float", "datetime"}
COMPARE_OPS = {"<", "<=", ">", ">=", "==", "!="}
RULE_TYPES = {"compare", "required_if"}
TOP_LEVEL_KEYS = {"name", "version", "owner", "description", "columns", "rules", "feed"}
FEED_KEYS = {"min_rows", "max_reject_rate"}


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
    for bound in ("min", "max"):
        value = getattr(spec, bound)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ContractError(f"column '{name}': {bound} must be a number, got {value!r}")
    if spec.min is not None and spec.max is not None and spec.min > spec.max:
        raise ContractError(f"column '{name}': min ({spec.min}) is greater than max ({spec.max})")
    if spec.allowed is not None:
        if not isinstance(spec.allowed, list) or not spec.allowed:
            raise ContractError(f"column '{name}': allowed must be a non-empty list")
        spec.allowed = [str(v) for v in spec.allowed]
    if spec.regex is not None:
        if not isinstance(spec.regex, str):
            raise ContractError(f"column '{name}': regex must be a string")
        try:
            re.compile(spec.regex)
        except re.error as exc:
            raise ContractError(f"column '{name}' has an invalid regex: {exc}") from None
    if spec.format is not None and spec.type != "datetime":
        raise ContractError(f"column '{name}': format only applies to datetime columns")
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


def _feed(raw: Any) -> FeedSpec:
    if raw is None:
        return FeedSpec()
    if not isinstance(raw, dict):
        raise ContractError("feed must be a mapping")
    unknown = set(raw) - FEED_KEYS
    if unknown:
        # A misspelt limit would otherwise fall back to its default and the
        # reject-rate gate would silently stop rejecting anything.
        raise ContractError(f"feed has unknown keys: {sorted(unknown)}; expected {sorted(FEED_KEYS)}")
    min_rows = raw.get("min_rows", 0)
    if isinstance(min_rows, bool) or not isinstance(min_rows, int) or min_rows < 0:
        raise ContractError(f"feed.min_rows must be a non-negative integer, got {min_rows!r}")
    rate = raw.get("max_reject_rate", 1.0)
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise ContractError(f"feed.max_reject_rate must be a number, got {rate!r}")
    if not 0.0 <= rate <= 1.0:
        raise ContractError("feed.max_reject_rate must be between 0 and 1")
    return FeedSpec(min_rows=min_rows, max_reject_rate=float(rate))


def column_check_names(spec: ColumnSpec) -> list[str]:
    """Names of the per-column checks the validator reports for ``spec``."""
    names = []
    if spec.required:
        names.append(f"{spec.name}.required")
    if spec.type != "string":
        names.append(f"{spec.name}.type_{spec.type}")
    for key in ("allowed", "regex", "min", "max"):
        if getattr(spec, key) is not None:
            names.append(f"{spec.name}.{key}")
    if spec.unique:
        names.append(f"{spec.name}.unique")
    return names


def _check_rule_names(rules: list[RuleSpec], columns: dict[str, ColumnSpec]) -> None:
    # The validator keys failing-row masks by check name, so a repeated name
    # would overwrite an earlier check and its failures would go unreported.
    taken = {name for spec in columns.values() for name in column_check_names(spec)}
    for rule in rules:
        if not isinstance(rule.name, str) or not rule.name.strip():
            raise ContractError(f"rule names must be non-empty strings, got {rule.name!r}")
        if rule.name in taken:
            raise ContractError(
                f"rule name '{rule.name}' is used more than once or clashes with a column check"
            )
        taken.add(rule.name)


def parse_contract(raw: dict[str, Any]) -> Contract:
    """Build a :class:`Contract` from an already-loaded mapping."""
    if not isinstance(raw, dict):
        raise ContractError("contract must be a mapping")
    unknown = set(raw) - TOP_LEVEL_KEYS
    if unknown:
        # e.g. "rule:" instead of "rules:" would otherwise drop every rule.
        raise ContractError(f"contract has unknown top-level keys: {sorted(unknown)}")
    if not raw.get("columns"):
        raise ContractError("contract must define at least one column")
    if not isinstance(raw["columns"], dict):
        raise ContractError("columns must be a mapping of column name to spec")
    columns = {name: _column(name, spec or {}) for name, spec in raw["columns"].items()}
    raw_rules = raw.get("rules") or []
    if not isinstance(raw_rules, list):
        raise ContractError("rules must be a list")
    rules = [_rule(r, columns) for r in raw_rules]
    _check_rule_names(rules, columns)
    return Contract(
        name=str(raw.get("name", "unnamed")),
        version=int(raw.get("version", 1)),
        columns=columns,
        rules=rules,
        feed=_feed(raw.get("feed")),
    )


def load_contract(path: str | Path) -> Contract:
    """Read a YAML contract from disk."""
    with open(path, encoding="utf-8") as fh:
        return parse_contract(yaml.safe_load(fh))
