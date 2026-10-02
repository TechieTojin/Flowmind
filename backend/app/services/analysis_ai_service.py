"""AI interpretation of a deterministic dataset analysis via the existing Ollama service.

Qwen only receives a compact, already-computed summary (never raw rows) and is asked
to interpret it. Any AI failure is reported in the result; it never raises.
"""

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, ValidationError, field_validator

from app.schemas.analysis import AIInterpretation, TabularAnalysis
from app.services.ollama_service import OllamaError, OllamaService, get_ollama_service

logger = logging.getLogger(__name__)

MAX_PROMPT_COLUMNS = 40
MAX_PROMPT_FINDINGS = 25
MAX_PROMPT_OUTLIERS = 15
MAX_ITEMS = 3
MAX_ITEM_LENGTH = 400
# qwen3:8b on CPU generates roughly 2-3 tokens/s; bound the answer so it fits the timeout.
MAX_RESPONSE_TOKENS = 400

SYSTEM_PROMPT = """You are FlowMind, a data analyst. You receive a JSON summary of a tabular \
file that was produced by deterministic Python analysis. Interpret it for a business user.

Rules:
- Use ONLY the supplied analysis. Do not invent statistics, numbers, columns or values.
- Do not recalculate anything; quote numbers exactly as given when you mention them.
- Do not claim facts the summary does not support. If something is uncertain, say so.
- Clearly separate data-quality findings from possible business interpretation \
(prefix business interpretation with "Possibly:").
- Be concise: summary at most 2 sentences; at most 3 key insights and 3 recommended actions, \
each one short sentence.
- Respond with JSON only, matching: \
{"summary": string, "key_insights": [string], "recommended_actions": [string]}"""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_insights": {"type": "array", "items": {"type": "string"}},
        "recommended_actions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "key_insights", "recommended_actions"],
}

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class _AIPayload(BaseModel):
    summary: str
    key_insights: list[str] = []
    recommended_actions: list[str] = []

    @field_validator("summary")
    @classmethod
    def _summary_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("summary is empty")
        return value.strip()[:1500]

    @field_validator("key_insights", "recommended_actions", mode="before")
    @classmethod
    def _clean_list(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        items = [str(item).strip()[:MAX_ITEM_LENGTH] for item in value if str(item).strip()]
        return items[:MAX_ITEMS]


class AIResponseParseError(ValueError):
    pass


def build_analysis_summary(analysis: TabularAnalysis, file_type: str) -> dict[str, Any]:
    """Compact, LLM-friendly view of the analysis. Contains no raw data rows."""
    columns = []
    for c in analysis.columns[:MAX_PROMPT_COLUMNS]:
        entry: dict[str, Any] = {
            "name": c.name,
            "kind": c.kind,
            "null_pct": c.null_percentage,
            "unique": c.unique_count,
        }
        if c.numeric is not None:
            entry["stats"] = c.numeric.model_dump(exclude_none=True)
        if c.text is not None:
            entry["top_values"] = [
                {"value": str(v.value)[:40], "count": v.count} for v in c.text.top_values[:3]
            ]
        columns.append(entry)

    return {
        "file_type": file_type,
        "rows": analysis.dataset.rows,
        "columns": analysis.dataset.columns,
        "columns_omitted": max(0, len(analysis.columns) - MAX_PROMPT_COLUMNS),
        "quality_score": analysis.quality.score,
        "score_penalties": [
            {"category": p.category, "penalty": p.penalty}
            for p in analysis.quality.penalties
            if p.penalty > 0
        ],
        "findings": [
            {"severity": f.severity, "code": f.code, "column": f.column, "description": f.description}
            for f in analysis.quality.findings[:MAX_PROMPT_FINDINGS]
        ],
        "outliers": [o.model_dump() for o in analysis.outliers[:MAX_PROMPT_OUTLIERS]],
        "column_summaries": columns,
    }


def parse_ai_response(raw: str) -> _AIPayload:
    """Parse model output into the expected structure, tolerating fences and extra text."""
    text = _CODE_FENCE.sub("", raw.strip())
    candidates = [text]
    match = _JSON_OBJECT.search(text)
    if match and match.group(0) != text:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            try:
                return _AIPayload.model_validate(data)
            except ValidationError as exc:
                raise AIResponseParseError("AI response did not match the expected structure.") from exc
    raise AIResponseParseError("AI response was not valid JSON.")


class AnalysisAIService:
    def __init__(self, ollama: OllamaService) -> None:
        self.ollama = ollama

    def interpret(self, analysis: TabularAnalysis, file_type: str) -> AIInterpretation:
        summary = build_analysis_summary(analysis, file_type)
        prompt = "Analysis summary:\n" + json.dumps(summary, separators=(",", ":"), ensure_ascii=False)
        try:
            raw = self.ollama.chat(
                prompt,
                system=SYSTEM_PROMPT,
                response_format=RESPONSE_SCHEMA,
                temperature=0.2,
                max_tokens=MAX_RESPONSE_TOKENS,
            )
            payload = parse_ai_response(raw)
        except (OllamaError, AIResponseParseError) as exc:
            return self._failed(str(exc))
        except Exception:  # never let AI problems fail the deterministic analysis
            logger.exception("Unexpected error during AI interpretation")
            return self._failed("Unexpected error during AI interpretation.")

        return AIInterpretation(
            requested=True,
            status="completed",
            model=self.ollama.model,
            summary=payload.summary,
            key_insights=payload.key_insights,
            recommended_actions=payload.recommended_actions,
        )

    def _failed(self, error: str) -> AIInterpretation:
        return AIInterpretation(requested=True, status="failed", model=self.ollama.model, error=error)

    @staticmethod
    def skipped() -> AIInterpretation:
        return AIInterpretation(requested=False, status="skipped")


def get_analysis_ai_service() -> AnalysisAIService:
    return AnalysisAIService(get_ollama_service())
