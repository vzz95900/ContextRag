"""Health check endpoint."""

from fastapi import APIRouter

from app.core.config import settings
from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        version="0.1.0",
        vector_store=settings.vector_store_provider,
        llm_provider=settings.llm_provider,
    )
