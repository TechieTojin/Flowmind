from fastapi import APIRouter, Depends, HTTPException

from app.schemas.ai import AIStatusResponse, ChatRequest, ChatResponse
from app.services.ollama_service import OllamaError, OllamaService, get_ollama_service

router = APIRouter(prefix="/ai", tags=["ai"])


# Sync handlers: FastAPI runs them in a threadpool, so blocking Ollama calls
# don't stall the event loop.
@router.get("/status", response_model=AIStatusResponse)
def ai_status(service: OllamaService = Depends(get_ollama_service)) -> AIStatusResponse:
    try:
        models = service.list_models()
    except OllamaError as exc:
        return AIStatusResponse(
            ollama_reachable=False,
            base_url=service.base_url,
            model=service.model,
            model_available=False,
            error=exc.message,
        )

    available = service.is_model_available(models)
    return AIStatusResponse(
        ollama_reachable=True,
        base_url=service.base_url,
        model=service.model,
        model_available=available,
        installed_models=models,
        error=None if available else f"Model '{service.model}' is not installed in Ollama",
    )


@router.post("/chat", response_model=ChatResponse)
def ai_chat(
    request: ChatRequest, service: OllamaService = Depends(get_ollama_service)
) -> ChatResponse:
    try:
        reply = service.chat(request.message)
    except OllamaError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    return ChatResponse(model=service.model, response=reply)
