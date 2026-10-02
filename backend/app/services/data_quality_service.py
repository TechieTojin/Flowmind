"""Deterministic data-quality checks and the V1 quality score.

QUALITY SCORE (V1)
------------------
The score starts at 100. Each category below subtracts `weight * ratio`, where the
ratio is between 0 and 1, so no single category can subtract more than its weight.
The weights add up to 100 and the result is rounded and clamped to 0-100.

    Category                  Weight  Ratio
    missing_values              40    missing cells / total cells (excluding fully empty columns)
    duplicate_rows              20    duplicate rows / total rows
    empty_columns               15    fully empty columns / total columns
    invalid_numeric_values      10    numeric columns containing +/-Infinity / total columns
    mixed_types                 10    columns flagged as mostly-numeric-with-text / total columns
    duplicate_column_names       5    columns sharing a duplicated name / total columns

Informational findings (constant columns, high cardinality, negative values,
outliers) never reduce the score: they may be completely legitimate.

Example: 1,000 rows x 10 columns, 3% of cells missing and 20 duplicate rows:
    100 - 40*0.03 - 20*0.02 = 98.4 -> 98
"""

import datetime as dt
import re
from collections import Counter

import numpy as np
import pandas as pd

from app.schemas.analysis import ColumnProfile, Finding, OutlierSummary, QualityReport, ScorePenalty

SCORE_WEIGHTS: dict[str, float] = {
    "missing_values": 40,
    "duplicate_rows": 20,
    "empty_columns": 15,
    "invalid_numeric_values": 10,
    "mixed_types": 10,
    "duplicate_column_names": 5,
}

MISSING_WARNING_PCT = 5.0
MISSING_CRITICAL_PCT = 50.0
DUPLICATE_ROWS_CRITICAL_PCT = 50.0
HIGH_CARDINALITY_MIN_VALUES = 20
HIGH_CARDINALITY_RATIO = 0.9
MIXED_NUMERIC_SHARE = 0.5

_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}
_NUMBER_PATTERN = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def evaluate_quality(
    dataframe: pd.DataFrame,
    columns: list[ColumnProfile],
    outliers: list[OutlierSummary],
    duplicate_column_names: list[str],
) -> QualityReport:
    duplicate_rows = int(dataframe.duplicated().sum()) if len(dataframe) else 0
    infinite_counts = {c.name: _infinite_count(dataframe[c.name]) for c in columns if c.kind == "numeric"}
    mixed = {
        c.name: breakdown
        for c in columns
        if c.kind == "text" and len(breakdown := mixed_type_breakdown(dataframe[c.name])) > 1
    }

    findings: list[Finding] = []
    findings += _missing_value_findings(columns)
    findings += _empty_column_findings(columns)
    findings += _duplicate_row_findings(duplicate_rows, len(dataframe))
    findings += _duplicate_name_findings(duplicate_column_names)
    findings += _constant_column_findings(columns, len(dataframe))
    findings += _high_cardinality_findings(columns)
    findings += _negative_value_findings(dataframe, columns)
    findings += _infinite_value_findings(infinite_counts)
    findings += _mixed_type_findings(mixed)
    findings += _outlier_findings(outliers)
    findings.sort(key=lambda f: _SEVERITY_ORDER[f.severity])

    mixed_warning_columns = sum(
        1 for f in findings if f.code == "MIXED_TYPES" and f.severity == "warning"
    )
    penalties = compute_penalties(
        columns=columns,
        rows=len(dataframe),
        duplicate_rows=duplicate_rows,
        infinite_columns=sum(1 for count in infinite_counts.values() if count > 0),
        mixed_type_columns=mixed_warning_columns,
        duplicate_name_columns=sum(1 for c in columns if _base_name(c.name) in duplicate_column_names),
    )
    return QualityReport(score=score_from_penalties(penalties), penalties=penalties, findings=findings)


# ---------------------------------------------------------------- score


def compute_penalties(
    *,
    columns: list[ColumnProfile],
    rows: int,
    duplicate_rows: int,
    infinite_columns: int,
    mixed_type_columns: int,
    duplicate_name_columns: int,
) -> list[ScorePenalty]:
    total_columns = len(columns)
    non_empty = [c for c in columns if c.kind != "empty"]
    missing_cells = sum(c.null_count for c in non_empty)
    total_cells = rows * len(non_empty)
    empty_columns = total_columns - len(non_empty)

    ratios = {
        "missing_values": (_ratio(missing_cells, total_cells), f"{missing_cells} of {total_cells} cells missing"),
        "duplicate_rows": (_ratio(duplicate_rows, rows), f"{duplicate_rows} of {rows} rows duplicated"),
        "empty_columns": (_ratio(empty_columns, total_columns), f"{empty_columns} of {total_columns} columns empty"),
        "invalid_numeric_values": (
            _ratio(infinite_columns, total_columns),
            f"{infinite_columns} of {total_columns} columns contain infinite values",
        ),
        "mixed_types": (
            _ratio(mixed_type_columns, total_columns),
            f"{mixed_type_columns} of {total_columns} columns have mixed types",
        ),
        "duplicate_column_names": (
            _ratio(duplicate_name_columns, total_columns),
            f"{duplicate_name_columns} of {total_columns} columns share a duplicated name",
        ),
    }
    return [
        ScorePenalty(
            category=category,
            max_penalty=SCORE_WEIGHTS[category],
            penalty=round(SCORE_WEIGHTS[category] * ratio, 2),
            detail=detail,
        )
        for category, (ratio, detail) in ratios.items()
    ]


def score_from_penalties(penalties: list[ScorePenalty]) -> int:
    raw = 100.0 - sum(p.penalty for p in penalties)
    return int(max(0, min(100, round(raw))))


# ---------------------------------------------------------------- mixed types


def _value_category(value: object) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "boolean"
    if isinstance(value, (int, float, np.number)):
        return "number"
    if isinstance(value, (dt.date, dt.time, pd.Timestamp)):
        return "datetime"
    if isinstance(value, str):
        return "number" if _NUMBER_PATTERN.match(value.strip().replace(",", "")) else "text"
    return "other"


def mixed_type_breakdown(series: pd.Series) -> dict[str, int]:
    """Count non-null values per category (number/text/datetime/boolean/other)."""
    return dict(Counter(_value_category(value) for value in series.dropna()))


# ---------------------------------------------------------------- findings


def _missing_value_findings(columns: list[ColumnProfile]) -> list[Finding]:
    findings = []
    for c in columns:
        if c.null_count == 0 or c.kind == "empty":
            continue
        if c.null_percentage >= MISSING_CRITICAL_PCT:
            severity = "critical"
        elif c.null_percentage >= MISSING_WARNING_PCT:
            severity = "warning"
        else:
            severity = "info"
        findings.append(
            Finding(
                code="MISSING_VALUES",
                severity=severity,
                title="Missing values detected",
                description=f"{c.null_count} values ({c.null_percentage}%) are missing in {c.name}.",
                column=c.name,
                count=c.null_count,
            )
        )
    return findings


def _empty_column_findings(columns: list[ColumnProfile]) -> list[Finding]:
    return [
        Finding(
            code="EMPTY_COLUMN",
            severity="warning",
            title="Completely empty column",
            description=f"Column {c.name} contains no values.",
            column=c.name,
            count=c.null_count,
        )
        for c in columns
        if c.kind == "empty"
    ]


def _duplicate_row_findings(duplicate_rows: int, rows: int) -> list[Finding]:
    if duplicate_rows == 0:
        return []
    pct = round(100.0 * duplicate_rows / rows, 2)
    return [
        Finding(
            code="DUPLICATE_ROWS",
            severity="critical" if pct >= DUPLICATE_ROWS_CRITICAL_PCT else "warning",
            title="Duplicate rows detected",
            description=f"{duplicate_rows} rows ({pct}%) are exact duplicates of an earlier row.",
            count=duplicate_rows,
        )
    ]


def _duplicate_name_findings(duplicate_column_names: list[str]) -> list[Finding]:
    return [
        Finding(
            code="DUPLICATE_COLUMN_NAMES",
            severity="warning",
            title="Duplicate column name",
            description=f"Column name {name} appears more than once; later copies were renamed {name}.1, {name}.2, ...",
            column=name,
        )
        for name in duplicate_column_names
    ]


def _constant_column_findings(columns: list[ColumnProfile], rows: int) -> list[Finding]:
    if rows < 2:
        return []
    return [
        Finding(
            code="CONSTANT_COLUMN",
            severity="info",
            title="Constant column",
            description=f"Column {c.name} has the same value in every non-empty row.",
            column=c.name,
        )
        for c in columns
        if c.unique_count == 1 and c.non_null_count > 1
    ]


def _high_cardinality_findings(columns: list[ColumnProfile]) -> list[Finding]:
    return [
        Finding(
            code="HIGH_CARDINALITY",
            severity="info",
            title="High-cardinality text column",
            description=(
                f"Column {c.name} has {c.unique_count} unique values out of {c.non_null_count}; "
                "it may be an identifier or free text."
            ),
            column=c.name,
            count=c.unique_count,
        )
        for c in columns
        if c.kind == "text"
        and c.text is not None
        and c.non_null_count >= HIGH_CARDINALITY_MIN_VALUES
        and c.text.unique_ratio >= HIGH_CARDINALITY_RATIO
    ]


def _negative_value_findings(dataframe: pd.DataFrame, columns: list[ColumnProfile]) -> list[Finding]:
    findings = []
    for c in columns:
        if c.kind != "numeric":
            continue
        values = pd.to_numeric(dataframe[c.name], errors="coerce")
        count = int((values < 0).sum())
        if count:
            findings.append(
                Finding(
                    code="NEGATIVE_VALUES",
                    severity="info",
                    title="Negative values present",
                    description=f"{count} negative values found in {c.name}. Verify they are expected.",
                    column=c.name,
                    count=count,
                )
            )
    return findings


def _infinite_count(series: pd.Series) -> int:
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    return int(np.isinf(values).sum())


def _infinite_value_findings(infinite_counts: dict[str, int]) -> list[Finding]:
    return [
        Finding(
            code="INFINITE_VALUES",
            severity="warning",
            title="Infinite values detected",
            description=f"{count} infinite values found in {name}; they are excluded from statistics.",
            column=name,
            count=count,
        )
        for name, count in infinite_counts.items()
        if count > 0
    ]


def _mixed_type_findings(mixed: dict[str, dict[str, int]]) -> list[Finding]:
    findings = []
    for name, breakdown in mixed.items():
        total = sum(breakdown.values())
        numbers = breakdown.get("number", 0)
        parts = ", ".join(f"{count} {category}" for category, count in sorted(breakdown.items()))
        if numbers / total >= MIXED_NUMERIC_SHARE:
            non_numeric = total - numbers
            findings.append(
                Finding(
                    code="MIXED_TYPES",
                    severity="warning",
                    title="Mostly numeric column contains non-numeric values",
                    description=f"Column {name} is mostly numeric but has {non_numeric} non-numeric values ({parts}).",
                    column=name,
                    count=non_numeric,
                )
            )
        else:
            findings.append(
                Finding(
                    code="MIXED_TYPES",
                    severity="info",
                    title="Column contains multiple value types",
                    description=f"Column {name} contains values of several types ({parts}).",
                    column=name,
                )
            )
    return findings


def _outlier_findings(outliers: list[OutlierSummary]) -> list[Finding]:
    return [
        Finding(
            code="OUTLIERS",
            severity="info",
            title="Potential outliers (IQR method)",
            description=(
                f"{o.outlier_count} values ({o.outlier_percentage}%) in {o.column} fall outside "
                f"[{o.lower_bound}, {o.upper_bound}]."
            ),
            column=o.column,
            count=o.outlier_count,
        )
        for o in outliers
    ]


def _base_name(name: str) -> str:
    """`price.1` -> `price` (reverses the duplicate-header renaming)."""
    base, sep, suffix = name.rpartition(".")
    return base if sep and suffix.isdigit() else name


def _ratio(part: int, whole: int) -> float:
    return min(1.0, part / whole) if whole else 0.0
