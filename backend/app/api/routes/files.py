import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.schemas.files import FileAnalysisResponse
from app.services.analysis_ai_service import AnalysisAIService, get_analysis_ai_service
from app.services.file_analysis_service import analyze_file
from app.services.file_service import (
    FileValidationError,
    detect_file_type,
    read_upload_limited,
    sanitize_filename,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/files", tags=["files"])


@router.post("/analyze", response_model=FileAnalysisResponse)
async def analyze_upload(
    file: UploadFile = File(..., description="A .csv or .xlsx file (max 20 MB)"),
    use_ai: bool = Form(True),
    ai_service: AnalysisAIService = Depends(get_analysis_ai_service),
) -> FileAnalysisResponse:
    filename = sanitize_filename(file.filename)
    try:
        detect_file_type(filename)  # reject unsupported types before reading the body
        content = await read_upload_limited(file, get_settings().max_upload_bytes)
        # Parsing, analysis and Ollama calls are blocking: keep them off the event loop.
        return await run_in_threadpool(analyze_file, filename, content, use_ai, ai_service)
    except FileValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except Exception as exc:
        logger.exception("Unexpected error while analyzing an uploaded file")
        raise HTTPException(status_code=500, detail="The file could not be analyzed.") from exc
    finally:
        await file.close()
