"""Check a feed against a contract and split it into accepted and quarantined rows."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .contract import ColumnSpec, Contract, RuleSpec

OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


@dataclass
class Violation:
    """One rule broken by one row."""

    row: int
    column: str
    rule: str
    value: str


@dataclass
class FeedCheck:
    """A check on the feed as a whole (schema, row count, reject rate)."""

    name: str
    passed: bool
    detail: str


@dataclass
class ValidationResult:
    accepted: pd.DataFrame
    quarantined: pd.DataFrame
    violations: list[Violation]
    feed_checks: list[FeedCheck] = field(default_factory=list)

    @property
    def total_rows(self) -> int:
        return len(self.accepted) + len(self.quarantined)

    @property
    def reject_rate(self) -> float:
        return len(self.quarantined) / self.total_rows if self.total_rows else 0.0

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.feed_checks)

    def violation_counts(self) -> dict[str, int]:
        """Number of violations per rule name, largest first."""
        counts: dict[str, int] = {}
        for v in self.violations:
            counts[v.rule] = counts.get(v.rule, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _parse(series: pd.Series, spec: ColumnSpec) -> pd.Series:
    """Convert raw string values to the contract type; unparseable values become NA."""
    if spec.type in {"int", "float"}:
        return pd.to_numeric(series, errors="coerce")
    if spec.type == "datetime":
        return pd.to_datetime(series, format=spec.format, errors="coerce")
    return series


def _check_column(raw: pd.Series, spec: ColumnSpec) -> dict[str, pd.Series]:
    """Return {rule_name: boolean mask of failing rows} for one column."""
    failures: dict[str, pd.Series] = {}
    blank = raw.str.strip() == ""
    present = ~blank

    if spec.required:
        failures[f"{spec.name}.required"] = blank

    parsed = _parse(raw, spec)
    if spec.type != "string":
        failures[f"{spec.name}.type_{spec.type}"] = present & parsed.isna()

    if spec.allowed is not None:
        failures[f"{spec.name}.allowed"] = present & ~raw.isin(spec.allowed)
    if spec.regex is not None:
        failures[f"{spec.name}.regex"] = present & ~raw.str.fullmatch(spec.regex)
    if spec.min is not None:
        failures[f"{spec.name}.min"] = present & (parsed < spec.min)
    if spec.max is not None:
        failures[f"{spec.name}.max"] = present & (parsed > spec.max)
    if spec.unique:
        failures[f"{spec.name}.unique"] = present & raw.duplicated(keep="first")
    return failures


def _check_rule(df: pd.DataFrame, parsed: dict[str, pd.Series], rule: RuleSpec) -> pd.Series:
    """Return a boolean mask of rows failing a cross-column rule."""
    p = rule.params
    if rule.type == "compare":
        left, right = parsed[p["left"]], parsed[p["right"]]
        both = left.notna() & right.notna()
        result = pd.Series(False, index=df.index)
        result[both] = ~OPS[p["op"]](left[both], right[both])
        return result
    if rule.type == "required_if":
        condition = df[p["when_column"]] == str(p["equals"])
        return condition & (df[p["column"]].str.strip() == "")
    raise ValueError(f"unsupported rule type {rule.type}")


def validate_frame(df: pd.DataFrame, contract: Contract) -> ValidationResult:
    """Validate a DataFrame of raw values against ``contract``.

    Values are compared as strings. Missing values (NaN, None, pd.NA, NaT) are
    turned into empty strings first, so a frame from ``pd.read_csv`` with default
    settings, a database query or hand-built code is checked the same way as one
    from :func:`read_feed`.
    """
    df = df.reset_index(drop=True)
    df = df.astype(object).where(df.notna(), "").astype(str)
    checks: list[FeedCheck] = []

    missing = [c for c in contract.columns if c not in df.columns]
    extra = [c for c in df.columns if c not in contract.columns]
    checks.append(
        FeedCheck(
            "schema.columns_present",
            not missing,
            f"missing columns: {missing}" if missing else "all contract columns present",
        )
    )
    if extra:
        checks.append(FeedCheck("schema.no_extra_columns", True, f"ignored extra columns: {extra}"))

    if missing:
        # Without the agreed columns no row can be trusted; quarantine everything.
        quarantined = df.copy()
        quarantined["_violations"] = "schema.columns_present"
        checks.append(FeedCheck("feed.min_rows", len(df) >= contract.feed.min_rows, f"{len(df)} rows"))
        return ValidationResult(df.iloc[0:0].copy(), quarantined, [], checks)

    masks: dict[str, pd.Series] = {}
    rule_column: dict[str, str] = {}
    parsed: dict[str, pd.Series] = {}
    for name, spec in contract.columns.items():
        parsed[name] = _parse(df[name], spec)
        for rule_name, mask in _check_column(df[name], spec).items():
            masks[rule_name] = mask.fillna(False).astype(bool)
            rule_column[rule_name] = name
    for rule in contract.rules:
        masks[rule.name] = _check_rule(df, parsed, rule).fillna(False).astype(bool)
        rule_column[rule.name] = rule.params.get("left") or rule.params.get("column")

    violations: list[Violation] = []
    reasons: dict[int, list[str]] = {}
    for rule_name, mask in masks.items():
        column = rule_column[rule_name]
        for idx in mask[mask].index:
            violations.append(Violation(int(idx), column, rule_name, df.at[idx, column]))
            reasons.setdefault(int(idx), []).append(rule_name)

    bad = df.index.isin(list(reasons))
    accepted = df[~bad].copy()
    quarantined = df[bad].copy()
    quarantined["_violations"] = ["; ".join(reasons[int(i)]) for i in quarantined.index]

    result = ValidationResult(accepted, quarantined, violations, checks)
    checks.append(
        FeedCheck(
            "feed.min_rows",
            len(df) >= contract.feed.min_rows,
            f"{len(df)} rows (minimum {contract.feed.min_rows})",
        )
    )
    checks.append(
        FeedCheck(
            "feed.max_reject_rate",
            result.reject_rate <= contract.feed.max_reject_rate,
            f"{result.reject_rate:.1%} rejected (limit {contract.feed.max_reject_rate:.0%})",
        )
    )
    return result


def read_feed(path: str | Path) -> pd.DataFrame:
    """Read a CSV feed keeping every value as a string, so nothing is silently coerced."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def validate_file(path: str | Path, contract: Contract) -> ValidationResult:
    return validate_frame(read_feed(path), contract)
