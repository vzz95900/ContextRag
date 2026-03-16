"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration — values are read from .env automatically."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── API Keys ────────────────────────────────────────────
    openai_api_key: str = ""
    gemini_api_key: str = ""
    hf_token: str = ""

    # ── LLM ─────────────────────────────────────────────────
    llm_provider: str = "huggingface"  # gemini | openai | ollama | huggingface
    llm_model: str = "Qwen/Qwen2.5-72B-Instruct"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    ollama_base_url: str = "http://localhost:11434"

    # ── Embeddings ──────────────────────────────────────────
    embedding_provider: str = "gemini"  # gemini | openai | local
    embedding_model: str = "gemini-embedding-001"

    # ── Vector Store ────────────────────────────────────────
    vector_store_provider: str = "chromadb"  # chromadb | faiss
    chroma_persist_dir: str = "./data/chromadb"
    faiss_index_path: str = "./data/faiss_index"

    # ── Chunking ────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── Retrieval ───────────────────────────────────────────
    retrieval_top_k: int = 10
    rerank_top_n: int = 3
    enable_hybrid_search: bool = True
    enable_reranker: bool = True

    # ── API ─────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: List[str] = ["http://localhost:8501", "http://localhost:8502", "http://localhost:8000"]

    # ── Paths ───────────────────────────────────────────────
    upload_dir: str = "./data/uploads"

    def ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        for d in (self.upload_dir, self.chroma_persist_dir):
            Path(d).mkdir(parents=True, exist_ok=True)


# Singleton — import this everywhere
settings = Settings()
