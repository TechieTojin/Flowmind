"""Pydantic schemas for the AI endpoints."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000, description="User message")


class ChatResponse(BaseModel):
    model: str
    response: str


class AIStatusResponse(BaseModel):
    ollama_reachable: bool
    base_url: str
    model: str
    model_available: bool
    installed_models: list[str] = Field(default_factory=list)
    error: str | None = None
