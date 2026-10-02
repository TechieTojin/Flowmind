"""Endpoint tests for POST /api/files/analyze and preservation of existing routes."""

import json
import math

from tests.conftest import make_csv, make_xlsx

ANALYZE = "/api/files/analyze"

SALES_ROWS = [
    ["order_id", "region", "price", "quantity"],
    [1, "North", 10.5, 2],
    [2, "South", 12.0, 1],
    [3, "North", None, 5],
    [4, "East", 9.75, 3],
    [5, "West", 11.25, 4],
    [6, "South", 10.0, 2],
]


def _assert_strict_json(body: bytes) -> None:
    """Fails if the body contains NaN/Infinity (invalid in strict JSON)."""

    def reject(constant: str) -> None:
        raise AssertionError(f"non-standard JSON constant: {constant}")

    json.loads(body, parse_constant=reject)


def test_valid_csv_analysis(client, fake_ai):
    response = client.post_file(ANALYZE, "sales.csv", make_csv(SALES_ROWS), {"use_ai": "true"})
    assert response.status_code == 200, response.body
    _assert_strict_json(response.body)
    data = response.json()

    assert data["file"] == {
        "filename": "sales.csv",
        "file_type": "csv",
        "size_bytes": len(make_csv(SALES_ROWS)),
        "analyzed_sheet": None,
        "available_sheets": [],
    }
    assert data["dataset"] == {
        "rows": 6,
        "columns": 4,
        "column_names": ["order_id", "region", "price", "quantity"],
    }
    price = next(c for c in data["columns"] if c["name"] == "price")
    assert price["kind"] == "numeric"
    assert price["null_count"] == 1 and price["non_null_count"] == 5
    assert price["numeric"]["min"] == 9.75 and price["numeric"]["max"] == 12.0
    region = next(c for c in data["columns"] if c["name"] == "region")
    assert region["kind"] == "text" and region["text"]["top_values"][0]["count"] == 2
    assert len(data["preview"]) == 5
    assert data["preview"][2]["price"] is None
    assert data["ai"]["status"] == "completed" and fake_ai.calls == 1


def test_valid_xlsx_analysis_uses_first_non_empty_sheet(client, fake_ai):
    content = make_xlsx({"Cover": [], "Data": SALES_ROWS, "Notes": [["just a note"]]})
    response = client.post_file(ANALYZE, "sales.xlsx", content, {"use_ai": "false"})
    assert response.status_code == 200, response.body
    data = response.json()

    assert data["file"]["file_type"] == "xlsx"
    assert data["file"]["analyzed_sheet"] == "Data"
    assert data["file"]["available_sheets"] == ["Cover", "Data", "Notes"]
    assert data["dataset"]["rows"] == 6
    assert data["dataset"]["column_names"] == ["order_id", "region", "price", "quantity"]


def test_xlsx_formulas_are_not_evaluated(client, fake_ai):
    content = make_xlsx({"Sheet": [["a", "b"], [1, "=A2*1000"], [2, "=SUM(A1:A3)"]]})
    response = client.post_file(ANALYZE, "f.xlsx", content, {"use_ai": "false"})
    assert response.status_code == 200, response.body
    # openpyxl-written files have no cached values, so formulas yield empty cells.
    b = next(c for c in response.json()["columns"] if c["name"] == "b")
    assert b["kind"] == "empty"


def test_unsupported_extension_returns_415(client, fake_ai):
    response = client.post_file(ANALYZE, "notes.txt", b"hello")
    assert response.status_code == 415
    assert "Only .csv and .xlsx" in response.json()["detail"]


def test_empty_file_returns_422(client, fake_ai):
    response = client.post_file(ANALYZE, "empty.csv", b"")
    assert response.status_code == 422
    assert response.json()["detail"] == "The uploaded file is empty."


def test_header_only_csv_returns_422(client, fake_ai):
    response = client.post_file(ANALYZE, "h.csv", b"a,b,c\n")
    assert response.status_code == 422
    assert "no data rows" in response.json()["detail"]


def test_malformed_xlsx_returns_422(client, fake_ai):
    response = client.post_file(ANALYZE, "broken.xlsx", b"this is not a workbook")
    assert response.status_code == 422
    response = client.post_file(ANALYZE, "broken.xlsx", b"PK\x03\x04garbage-zip-content")
    assert response.status_code == 422
    assert "Traceback" not in response.body.decode()


def test_malformed_csv_returns_422(client, fake_ai):
    content = b'a,b\n1,2\n3,4,5,6\n"unterminated,7\n'
    response = client.post_file(ANALYZE, "bad.csv", content)
    assert response.status_code == 422


def test_binary_content_with_csv_extension_returns_422(client, fake_ai):
    response = client.post_file(ANALYZE, "fake.csv", make_xlsx({"S": [["a"], [1]]}))
    assert response.status_code == 422


def test_workbook_without_usable_sheet_returns_422(client, fake_ai):
    response = client.post_file(ANALYZE, "blank.xlsx", make_xlsx({"A": [], "B": []}))
    assert response.status_code == 422
    assert "non-empty worksheet" in response.json()["detail"]


def test_oversized_upload_returns_413(client, fake_ai):
    content = b"a,b\n" + b"1,2\n" * (5 * 1024 * 1024 + 10)  # just over 20 MB
    response = client.post_file(ANALYZE, "big.csv", content)
    assert response.status_code == 413
    assert fake_ai.calls == 0


def test_use_ai_false_does_not_contact_ollama(client, monkeypatch):
    """No dependency override: the real service is used, but must never be called."""

    def explode(*args, **kwargs):
        raise AssertionError("Ollama must not be contacted when use_ai=false")

    monkeypatch.setattr("app.services.ollama_service.requests.request", explode)
    response = client.post_file(ANALYZE, "sales.csv", make_csv(SALES_ROWS), {"use_ai": "false"})
    assert response.status_code == 200, response.body
    assert response.json()["ai"] == {
        "requested": False,
        "status": "skipped",
        "model": None,
        "summary": None,
        "key_insights": [],
        "recommended_actions": [],
        "error": None,
    }


def test_ai_failure_does_not_fail_request(client, monkeypatch):
    """Real AI service, with Ollama unreachable: analysis still succeeds."""
    import requests

    def unreachable(*args, **kwargs):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr("app.services.ollama_service.requests.request", unreachable)
    response = client.post_file(ANALYZE, "sales.csv", make_csv(SALES_ROWS), {"use_ai": "true"})
    assert response.status_code == 200, response.body
    ai = response.json()["ai"]
    assert ai["requested"] is True and ai["status"] == "failed"
    assert "Could not connect to Ollama" in ai["error"]
    assert response.json()["dataset"]["rows"] == 6


def test_infinite_values_are_json_safe(client, fake_ai):
    rows = [["x"], *[[v] for v in [1, 2, "inf", 3, "-inf", 4, 5, 6, 7, 8]]]
    response = client.post_file(ANALYZE, "inf.csv", make_csv(rows), {"use_ai": "false"})
    assert response.status_code == 200, response.body
    _assert_strict_json(response.body)
    data = response.json()
    codes = {f["code"] for f in data["quality"]["findings"]}
    assert "INFINITE_VALUES" in codes
    x = data["columns"][0]
    assert x["numeric"]["max"] == 8 and math.isfinite(x["numeric"]["mean"])
    assert data["preview"][2]["x"] is None  # "inf" rendered as null, not Infinity


def test_health_still_works(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "FlowMind API"}


def test_existing_ai_routes_remain_available(client, monkeypatch):
    import requests

    def unreachable(*args, **kwargs):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr("app.services.ollama_service.requests.request", unreachable)

    status = client.get("/api/ai/status")
    assert status.status_code == 200
    assert status.json()["ollama_reachable"] is False

    chat = client.post_json("/api/ai/chat", {"message": "Hello"})
    assert chat.status_code == 503

    paths = client.get("/openapi.json").json()["paths"]
    assert {"/api/health", "/api/ai/status", "/api/ai/chat", "/api/files/analyze"} <= set(paths)
