"""FastAPI dependency injection — shared service instances."""

from __future__ import annotations

from functools import lru_cache

from app.core.config import Settings


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""
    return Settings()
