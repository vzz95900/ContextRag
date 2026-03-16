"""LLM chain — grounded answer generation with citations (Gemini-first)."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a precise research assistant. Answer the user's question using
ONLY the provided context. Follow these rules strictly:

1. If the context contains the answer, respond clearly and cite sources
   as [Source: filename, page X].
2. If the context is insufficient, say "I don't have enough information
   in the provided documents to answer this question."
3. NEVER fabricate information not present in the context.
4. Quote directly from sources when possible.
5. Be concise but thorough.
"""

# Max characters per chunk to keep token usage low
_MAX_CHUNK_CHARS = 1500


def _format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a context block for the prompt."""
    parts: List[str] = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        filename = meta.get("filename", chunk.get("filename", "unknown"))
        page = meta.get("page_num", meta.get("page", "?"))
        text = chunk["text"][:_MAX_CHUNK_CHARS]
        parts.append(
            f"[Chunk {i}] (Source: {filename}, Page {page})\n{text}"
        )
    return "\n\n---\n\n".join(parts)


def _retry_on_429(func, *args, max_retries: int = 1, **kwargs):
    """Retry a function call with exponential backoff on 429 rate-limit errors."""
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
            if is_rate_limit and attempt < max_retries:
                wait = 2 ** attempt * 15  # 15s, 30s, 60s
                logger.warning("Rate limited (attempt %d/%d). Waiting %ds...", attempt + 1, max_retries, wait)
                time.sleep(wait)
            else:
                raise


def _call_gemini(contents, system_instruction, temperature, max_output_tokens, model):
    """Single Gemini API call (separated for retry wrapping)."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key)
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
    )
    return response.text


def _generate_gemini(context: str, query: str, history: List[Dict[str, str]] | None = None) -> str:
    """Generate answer using Google Gemini (google.genai) with retry."""
    from google.genai import types

    # Build conversation contents
    contents: list = []
    if history:
        for msg in history:
            contents.append(types.Content(
                role=msg.get("role", "user"),
                parts=[types.Part.from_text(text=msg.get("parts", [msg.get("content", "")])[0]
                       if isinstance(msg.get("parts"), list)
                       else msg.get("content", ""))],
            ))

    user_message = f"Context:\n{context}\n\nQuestion: {query}"
    contents.append(types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    ))

    return _retry_on_429(
        _call_gemini,
        contents,
        SYSTEM_PROMPT,
        settings.llm_temperature,
        settings.llm_max_tokens,
        settings.llm_model,
    )


def _generate_openai(context: str, query: str, history: List[Dict[str, str]] | None = None) -> str:
    """Generate answer using OpenAI."""
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    return response.choices[0].message.content or ""


def _generate_huggingface(context: str, query: str, history: List[Dict[str, str]] | None = None) -> str:
    """Generate answer using Hugging Face Serverless Inference API."""
    import requests as _requests

    headers = {
        "Authorization": f"Bearer {settings.hf_token}",
        "Content-Type": "application/json"
    }

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})

    url = "https://router.huggingface.co/v1/chat/completions"

    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
    }

    resp = _requests.post(url, headers=headers, json=payload, timeout=120)

    if resp.status_code != 200:
        raise ValueError(f"Hugging Face API Error: {resp.status_code} - {resp.text}")

    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _generate_ollama(context: str, query: str, history: List[Dict[str, str]] | None = None) -> str:
    """Generate answer using a local Ollama model."""
    import requests as _requests

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if not content and isinstance(msg.get("parts"), list):
                content = msg["parts"][0]
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})

    url = f"{settings.ollama_base_url}/api/chat"
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": settings.llm_temperature,
            "num_predict": settings.llm_max_tokens,
        },
    }

    logger.info("Calling Ollama at %s with model %s ...", url, settings.llm_model)
    resp = _requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "")


def generate_answer(
    query: str,
    chunks: List[Dict[str, Any]],
    history: List[Dict[str, str]] | None = None,
) -> str:
    """
    Generate a grounded answer from retrieved chunks.

    Enforces source citation and refuses to answer if context is insufficient.
    """
    if not chunks:
        return (
            "I don't have enough information in the provided documents "
            "to answer this question. Please upload relevant PDFs first."
        )

    context = _format_context(chunks)
    provider = settings.llm_provider.lower()

    if provider == "gemini":
        return _generate_gemini(context, query, history)
    elif provider == "openai":
        return _generate_openai(context, query, history)
    elif provider == "huggingface":
        return _generate_huggingface(context, query, history)
    elif provider == "ollama":
        return _generate_ollama(context, query, history)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
