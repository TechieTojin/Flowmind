"""Pydantic schemas for the file analysis endpoint."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.analysis import (
    AIInterpretation,
    ColumnProfile,
    DatasetInfo,
    OutlierSummary,
    QualityReport,
)

FileType = Literal["csv", "xlsx"]


class FileMetadata(BaseModel):
    filename: str
    file_type: FileType
    size_bytes: int
    analyzed_sheet: str | None = None
    available_sheets: list[str] = Field(default_factory=list)


class FileAnalysisResponse(BaseModel):
    file: FileMetadata
    dataset: DatasetInfo
    columns: list[ColumnProfile]
    quality: QualityReport
    outliers: list[OutlierSummary]
    preview: list[dict[str, Any]]
    ai: AIInterpretation
