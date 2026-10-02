"""Pydantic schemas for deterministic dataset analysis and AI interpretation."""

from typing import Any, Literal

from pydantic import BaseModel, Field

Severity = Literal["info", "warning", "critical"]
ColumnKind = Literal["numeric", "boolean", "datetime", "text", "empty"]
AIStatus = Literal["completed", "failed", "skipped"]

# JSON-safe scalar used in previews and top values.
JSONScalar = str | int | float | bool | None


class DatasetInfo(BaseModel):
    rows: int
    columns: int
    column_names: list[str]


class NumericStats(BaseModel):
    """Computed over finite values only (NaN and +/-Infinity excluded)."""

    min: float | None = None
    max: float | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None


class ValueCount(BaseModel):
    value: JSONScalar
    count: int


class TextStats(BaseModel):
    unique_ratio: float = Field(description="unique non-null values / non-null values")
    top_values: list[ValueCount] = Field(default_factory=list)


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    kind: ColumnKind
    non_null_count: int
    null_count: int
    null_percentage: float
    unique_count: int
    numeric: NumericStats | None = None
    text: TextStats | None = None


class Finding(BaseModel):
    code: str
    severity: Severity
    title: str
    description: str
    column: str | None = None
    count: int | None = None


class ScorePenalty(BaseModel):
    category: str
    max_penalty: float
    penalty: float
    detail: str


class QualityReport(BaseModel):
    score: int = Field(ge=0, le=100)
    penalties: list[ScorePenalty]
    findings: list[Finding]


class OutlierSummary(BaseModel):
    column: str
    outlier_count: int
    outlier_percentage: float
    lower_bound: float
    upper_bound: float


class TabularAnalysis(BaseModel):
    """Deterministic analysis result, independent of file format and AI."""

    dataset: DatasetInfo
    columns: list[ColumnProfile]
    quality: QualityReport
    outliers: list[OutlierSummary]
    preview: list[dict[str, Any]]


class AIInterpretation(BaseModel):
    requested: bool
    status: AIStatus
    model: str | None = None
    summary: str | None = None
    key_insights: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    error: str | None = None
