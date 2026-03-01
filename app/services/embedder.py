"""Embedding generation — supports Gemini, OpenAI, and local models."""

from __future__ import annotations

import logging
import time
from typing import List

from app.core.config import settings

logger = logging.getLogger(__name__)


def _retry_on_429(func, *args, max_retries: int = 3, **kwargs):
    """Retry with exponential backoff on 429 rate-limit errors."""
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
            if is_rate_limit and attempt < max_retries:
                wait = 2 ** attempt * 15  # 15s, 30s, 60s
                logger.warning("Embedding rate limited (attempt %d/%d). Waiting %ds...", attempt + 1, max_retries, wait)
                time.sleep(wait)
            else:
                raise


def _embed_gemini(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using Google Gemini API (google.genai) with retry."""
    from google import genai

    client = genai.Client(api_key=settings.gemini_api_key)

    embeddings: List[List[float]] = []
    # Batch embedding (up to 100 texts per call)
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = _retry_on_429(
            client.models.embed_content,
            model=settings.embedding_model,
            contents=batch,
        )
        embeddings.extend([e.values for e in result.embeddings])
    return embeddings


def _embed_gemini_query(text: str) -> List[float]:
    """Embed a single query using Gemini with retry."""
    from google import genai

    client = genai.Client(api_key=settings.gemini_api_key)

    result = _retry_on_429(
        client.models.embed_content,
        model=settings.embedding_model,
        contents=text,
    )
    return result.embeddings[0].values


def _embed_openai(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using OpenAI API."""
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(model=settings.embedding_model, input=texts)
    return [item.embedding for item in response.data]


_local_embedding_model = None


def _embed_local(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using a local sentence-transformers model (cached)."""
    from sentence_transformers import SentenceTransformer

    global _local_embedding_model
    if _local_embedding_model is None:
        logger.info("Loading local embedding model: %s", settings.embedding_model)
        _local_embedding_model = SentenceTransformer(settings.embedding_model)
    return _local_embedding_model.encode(texts, show_progress_bar=False).tolist()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of texts using the configured provider.

    Returns a list of embedding vectors.
    """
    provider = settings.embedding_provider.lower()
    if provider == "gemini":
        return _embed_gemini(texts)
    elif provider == "openai":
        return _embed_openai(texts)
    elif provider == "local":
        return _embed_local(texts)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


def embed_query(text: str) -> List[float]:
    """
    Embed a single query string.

    Uses retrieval_query task type for Gemini (improves search quality).
    For other providers, delegates to embed_texts.
    """
    provider = settings.embedding_provider.lower()
    if provider == "gemini":
        return _embed_gemini_query(text)
    return embed_texts([text])[0]
