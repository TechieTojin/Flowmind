from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.middleware import reject_oversized_uploads
from app.api.routes import ai, files, health
from app.config import get_settings

settings = get_settings()

app = FastAPI(title=settings.app_name)

app.middleware("http")(reject_oversized_uploads)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(ai.router, prefix="/api")
app.include_router(files.router, prefix="/api")
