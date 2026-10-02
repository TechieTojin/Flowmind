import io
from collections.abc import Iterator
from typing import Any

import pytest
from openpyxl import Workbook

from app.main import app
from app.schemas.analysis import AIInterpretation, TabularAnalysis
from app.services.analysis_ai_service import get_analysis_ai_service
from tests.asgi_client import ASGIClient


def make_csv(rows: list[list[Any]]) -> bytes:
    lines = [",".join("" if value is None else str(value) for value in row) for row in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


def make_xlsx(sheets: dict[str, list[list[Any]]]) -> bytes:
    workbook = Workbook()
    workbook.remove(workbook.active)
    for title, rows in sheets.items():
        worksheet = workbook.create_sheet(title)
        for row in rows:
            worksheet.append(row)
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


class FakeAIService:
    """Stands in for AnalysisAIService; records whether it was called."""

    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def interpret(self, analysis: TabularAnalysis, file_type: str) -> AIInterpretation:
        self.calls += 1
        if self.fail:
            return AIInterpretation(requested=True, status="failed", error="Ollama unavailable")
        return AIInterpretation(
            requested=True,
            status="completed",
            model="fake",
            summary=f"{analysis.dataset.rows} rows analyzed.",
            key_insights=["insight"],
            recommended_actions=["action"],
        )


@pytest.fixture
def fake_ai() -> Iterator[FakeAIService]:
    fake = FakeAIService()
    app.dependency_overrides[get_analysis_ai_service] = lambda: fake
    yield fake
    app.dependency_overrides.pop(get_analysis_ai_service, None)


@pytest.fixture
def client() -> ASGIClient:
    return ASGIClient(app)
