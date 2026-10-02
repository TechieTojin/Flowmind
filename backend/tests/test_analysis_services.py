"""Unit tests for the deterministic analysis, quality score and AI response parsing."""

import numpy as np
import pandas as pd
import pytest

from app.schemas.analysis import ColumnProfile, ScorePenalty
from app.services.analysis_ai_service import (
    AIResponseParseError,
    AnalysisAIService,
    build_analysis_summary,
    parse_ai_response,
)
from app.services.data_quality_service import compute_penalties, score_from_penalties
from app.services.file_service import parse_file
from app.services.ollama_service import OllamaTimeoutError
from app.services.tabular_analysis_service import analyze_dataframe, iqr_outliers
from tests.conftest import make_csv


def _codes(analysis, code):
    return [f for f in analysis.quality.findings if f.code == code]


def test_missing_value_detection():
    df = pd.DataFrame({"price": [1.0, None, 3.0, None], "name": ["a", "b", None, "d"]})
    analysis = analyze_dataframe(df)
    missing = {f.column: f for f in _codes(analysis, "MISSING_VALUES")}
    assert missing["price"].count == 2 and missing["price"].severity == "critical"  # 50%
    assert missing["name"].count == 1 and missing["name"].severity == "warning"  # 25%
    price = next(c for c in analysis.columns if c.name == "price")
    assert price.null_count == 2 and price.null_percentage == 50.0


def test_duplicate_row_detection():
    df = pd.DataFrame({"a": [1, 1, 2, 1], "b": ["x", "x", "y", "x"]})
    analysis = analyze_dataframe(df)
    (finding,) = _codes(analysis, "DUPLICATE_ROWS")
    assert finding.count == 2 and finding.severity == "critical"  # 2 of 4 = 50%
    assert analysis.quality.score < 100


def test_iqr_outlier_detection():
    series = pd.Series([10, 11, 12, 13, 12, 11, 10, 12, 13, 1000])
    result = iqr_outliers("amount", series)
    assert result is not None
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    assert result.lower_bound == pytest.approx(q1 - 1.5 * (q3 - q1))
    assert result.upper_bound == pytest.approx(q3 + 1.5 * (q3 - q1))
    assert result.outlier_count == 1
    assert result.outlier_percentage == 10.0

    analysis = analyze_dataframe(pd.DataFrame({"amount": series}))
    assert [o.column for o in analysis.outliers] == ["amount"]


@pytest.mark.parametrize(
    "values",
    [
        [5] * 20,  # constant -> IQR 0
        [1, 2, 3],  # too small
        [np.nan] * 10,  # empty
        [np.inf, -np.inf, 1, 2],  # mostly infinite
    ],
)
def test_iqr_outliers_skip_unsuitable_columns(values):
    assert iqr_outliers("x", pd.Series(values, dtype="float64")) is None


def test_quality_score_bounds():
    clean = analyze_dataframe(pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}))
    assert clean.quality.score == 100

    terrible = pd.DataFrame(
        {
            "a": [None, None, None, None],
            "b": [np.inf, np.inf, np.inf, np.inf],
            "c": ["1", "2", "3", "oops"],
            "d": [None, 1.0, None, 1.0],
        }
    )
    score = analyze_dataframe(terrible, ["c"]).quality.score
    assert 0 <= score <= 100


def test_score_clamped_with_extreme_penalties():
    columns = [
        ColumnProfile(name="a", dtype="object", kind="text", non_null_count=0,
                      null_count=10, null_percentage=100.0, unique_count=0)
    ]
    penalties = compute_penalties(
        columns=columns, rows=10, duplicate_rows=100, infinite_columns=50,
        mixed_type_columns=50, duplicate_name_columns=50,
    )
    # Ratios are capped at 1, so no category exceeds its weight.
    assert all(p.penalty <= p.max_penalty for p in penalties)
    assert 0 <= score_from_penalties(penalties) <= 100

    overflow = [ScorePenalty(category="x", max_penalty=100, penalty=150, detail="")]
    assert score_from_penalties(overflow) == 0
    bonus = [ScorePenalty(category="x", max_penalty=100, penalty=-50, detail="")]
    assert score_from_penalties(bonus) == 100


def test_informational_findings_do_not_reduce_score():
    df = pd.DataFrame(
        {
            "id": [f"id-{i}" for i in range(30)],  # high cardinality
            "const": ["same"] * 30,  # constant
            "delta": list(range(-15, 15)),  # negative values
        }
    )
    analysis = analyze_dataframe(df)
    codes = {f.code for f in analysis.quality.findings}
    assert {"HIGH_CARDINALITY", "CONSTANT_COLUMN", "NEGATIVE_VALUES"} <= codes
    assert all(f.severity == "info" for f in analysis.quality.findings)
    assert analysis.quality.score == 100


def test_other_quality_checks():
    df = pd.DataFrame(
        {
            "empty": [None] * 4,
            "amount": ["10", "20", "30", "n/a-ish"],
            "inf": [1.0, np.inf, 2.0, 3.0],
        }
    )
    analysis = analyze_dataframe(df, duplicate_column_names=["amount"])
    codes = {f.code: f for f in analysis.quality.findings}
    assert codes["EMPTY_COLUMN"].column == "empty"
    assert codes["MIXED_TYPES"].severity == "warning" and codes["MIXED_TYPES"].count == 1
    assert codes["INFINITE_VALUES"].count == 1
    assert codes["DUPLICATE_COLUMN_NAMES"].column == "amount"


def test_csv_duplicate_headers_and_bom():
    content = "﻿price,price,name\n1,2,a\n3,4,b\n".encode("utf-8")
    table = parse_file("dup.csv", content)
    assert list(table.dataframe.columns) == ["price", "price.1", "name"]
    assert table.duplicate_column_names == ["price"]


def test_ai_summary_is_compact_and_has_no_raw_rows():
    rows = [["secret_col", "n"]] + [[f"row-value-{i}", i] for i in range(500)]
    analysis = analyze_dataframe(parse_file("d.csv", make_csv(rows)).dataframe)
    summary = build_analysis_summary(analysis, "csv")
    text = str(summary)
    assert summary["rows"] == 500
    assert "preview" not in summary
    assert sum(f"row-value-{i}'" in text for i in range(500)) <= 3  # top values only


@pytest.mark.parametrize(
    "raw",
    [
        '{"summary": "ok", "key_insights": ["a"], "recommended_actions": ["b"]}',
        '```json\n{"summary": "ok", "key_insights": ["a"], "recommended_actions": ["b"]}\n```',
        'Here you go: {"summary": "ok", "key_insights": "a", "recommended_actions": ["b"]} thanks',
    ],
)
def test_parse_ai_response_is_robust(raw):
    payload = parse_ai_response(raw)
    assert payload.summary == "ok"
    assert payload.key_insights == ["a"] and payload.recommended_actions == ["b"]


@pytest.mark.parametrize("raw", ["not json", '{"key_insights": []}', '{"summary": "  "}'])
def test_parse_ai_response_rejects_invalid(raw):
    with pytest.raises(AIResponseParseError):
        parse_ai_response(raw)


class _StubOllama:
    model = "qwen3:8b"

    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    def chat(self, message, **kwargs):
        self.calls.append((message, kwargs))
        if self.error:
            raise self.error
        return self.result


def test_ai_service_success_and_failure_never_raise():
    analysis = analyze_dataframe(pd.DataFrame({"a": [1, 2, 3]}))

    ok = AnalysisAIService(_StubOllama('{"summary": "fine", "key_insights": [], "recommended_actions": []}'))
    result = ok.interpret(analysis, "csv")
    assert result.status == "completed" and result.summary == "fine"
    message, kwargs = ok.ollama.calls[0]
    assert "Do not invent statistics" in kwargs["system"]
    assert kwargs["max_tokens"] > 0
    assert kwargs["response_format"]["required"] == ["summary", "key_insights", "recommended_actions"]

    timeout = AnalysisAIService(_StubOllama(error=OllamaTimeoutError("Ollama did not respond")))
    result = timeout.interpret(analysis, "csv")
    assert result.status == "failed" and result.error == "Ollama did not respond"

    garbage = AnalysisAIService(_StubOllama("I cannot do that"))
    assert garbage.interpret(analysis, "csv").status == "failed"

    crash = AnalysisAIService(_StubOllama(error=RuntimeError("boom")))
    result = crash.interpret(analysis, "csv")
    assert result.status == "failed" and "boom" not in result.error
