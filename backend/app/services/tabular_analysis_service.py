"""Deterministic profiling of a DataFrame: column stats, IQR outliers and preview.

Every number returned here is calculated by pandas/numpy. The LLM never computes facts.
"""

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from app.schemas.analysis import (
    ColumnKind,
    ColumnProfile,
    DatasetInfo,
    NumericStats,
    OutlierSummary,
    TabularAnalysis,
    TextStats,
    ValueCount,
)
from app.services.data_quality_service import evaluate_quality
from app.utils.json_safe import safe_float, to_json_safe

PREVIEW_ROWS = 5
TOP_VALUES = 5
# IQR is meaningless on tiny samples, so require a minimum number of finite values.
MIN_OUTLIER_SAMPLE = 8
IQR_MULTIPLIER = 1.5


def analyze_dataframe(
    dataframe: pd.DataFrame, duplicate_column_names: list[str] | None = None
) -> TabularAnalysis:
    columns = [profile_column(str(name), dataframe[name]) for name in dataframe.columns]
    outliers = detect_outliers(dataframe, columns)
    quality = evaluate_quality(dataframe, columns, outliers, duplicate_column_names or [])
    return TabularAnalysis(
        dataset=DatasetInfo(
            rows=int(dataframe.shape[0]),
            columns=int(dataframe.shape[1]),
            column_names=[str(name) for name in dataframe.columns],
        ),
        columns=columns,
        quality=quality,
        outliers=outliers,
        preview=build_preview(dataframe),
    )


# ---------------------------------------------------------------- columns


def column_kind(series: pd.Series) -> ColumnKind:
    if series.notna().sum() == 0:
        return "empty"
    if ptypes.is_bool_dtype(series):
        return "boolean"
    if ptypes.is_numeric_dtype(series):
        return "numeric"
    if ptypes.is_datetime64_any_dtype(series):
        return "datetime"
    non_null = series.dropna()
    if non_null.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        return "boolean"
    return "text"


def finite_numeric_values(series: pd.Series) -> pd.Series:
    """Numeric values with NaN and +/-Infinity removed, as float64."""
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    return values[np.isfinite(values)]


def profile_column(name: str, series: pd.Series) -> ColumnProfile:
    total = int(len(series))
    non_null = int(series.notna().sum())
    unique = int(series.nunique(dropna=True))
    kind = column_kind(series)

    profile = ColumnProfile(
        name=name,
        dtype=str(series.dtype),
        kind=kind,
        non_null_count=non_null,
        null_count=total - non_null,
        null_percentage=_percentage(total - non_null, total),
        unique_count=unique,
    )
    if kind == "numeric":
        profile.numeric = _numeric_stats(series)
    elif kind in ("text", "boolean"):
        profile.text = _text_stats(series, unique, non_null)
    return profile


def _numeric_stats(series: pd.Series) -> NumericStats:
    values = finite_numeric_values(series)
    if values.empty:
        return NumericStats()
    return NumericStats(
        min=safe_float(values.min()),
        max=safe_float(values.max()),
        mean=safe_float(values.mean()),
        median=safe_float(values.median()),
        std=safe_float(values.std(ddof=1)) if len(values) > 1 else None,
    )


def _text_stats(series: pd.Series, unique: int, non_null: int) -> TextStats:
    counts = series.dropna().value_counts().head(TOP_VALUES)
    return TextStats(
        unique_ratio=round(unique / non_null, 4) if non_null else 0.0,
        top_values=[
            ValueCount(value=to_json_safe(value, max_text_length=100), count=int(count))
            for value, count in counts.items()
        ],
    )


# ---------------------------------------------------------------- outliers


def detect_outliers(
    dataframe: pd.DataFrame, columns: list[ColumnProfile]
) -> list[OutlierSummary]:
    """IQR method: outlier if value < Q1 - 1.5*IQR or value > Q3 + 1.5*IQR.

    Only finite values are considered. Columns with too few values or a zero IQR
    (constant / near-constant data) are skipped. Only columns with outliers are returned.
    """
    results: list[OutlierSummary] = []
    for profile in columns:
        if profile.kind != "numeric":
            continue
        summary = iqr_outliers(profile.name, dataframe[profile.name])
        if summary is not None and summary.outlier_count > 0:
            results.append(summary)
    return results


def iqr_outliers(name: str, series: pd.Series) -> OutlierSummary | None:
    values = finite_numeric_values(series)
    if len(values) < MIN_OUTLIER_SAMPLE:
        return None
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        return None
    lower = q1 - IQR_MULTIPLIER * iqr
    upper = q3 + IQR_MULTIPLIER * iqr
    count = int(((values < lower) | (values > upper)).sum())
    return OutlierSummary(
        column=name,
        outlier_count=count,
        outlier_percentage=_percentage(count, len(values)),
        lower_bound=safe_float(lower) or 0.0,
        upper_bound=safe_float(upper) or 0.0,
    )


# ---------------------------------------------------------------- preview


def build_preview(dataframe: pd.DataFrame, rows: int = PREVIEW_ROWS) -> list[dict[str, object]]:
    head = dataframe.head(rows)
    names = [str(name) for name in head.columns]
    return [
        {name: to_json_safe(value) for name, value in zip(names, row)}
        for row in head.itertuples(index=False, name=None)
    ]


def _percentage(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 2) if whole else 0.0
