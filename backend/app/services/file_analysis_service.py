"""Orchestrates parse -> deterministic analysis -> optional AI interpretation."""

from app.schemas.files import FileAnalysisResponse, FileMetadata
from app.services.analysis_ai_service import AnalysisAIService
from app.services.file_service import parse_file
from app.services.tabular_analysis_service import analyze_dataframe


def analyze_file(
    filename: str, content: bytes, use_ai: bool, ai_service: AnalysisAIService
) -> FileAnalysisResponse:
    """Raises FileValidationError for invalid uploads; AI failures never raise."""
    table = parse_file(filename, content)
    analysis = analyze_dataframe(table.dataframe, table.duplicate_column_names)
    ai = ai_service.interpret(analysis, table.file_type) if use_ai else AnalysisAIService.skipped()

    return FileAnalysisResponse(
        file=FileMetadata(
            filename=filename,
            file_type=table.file_type,
            size_bytes=len(content),
            analyzed_sheet=table.analyzed_sheet,
            available_sheets=table.available_sheets,
        ),
        dataset=analysis.dataset,
        columns=analysis.columns,
        quality=analysis.quality,
        outliers=analysis.outliers,
        preview=analysis.preview,
        ai=ai,
    )
