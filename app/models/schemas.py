"""Pydantic request / response schemas for the API."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ── Chat ────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Incoming chat message from the user."""

    query: str = Field(..., min_length=1, max_length=5000, description="User question")
    session_id: Optional[str] = Field(None, description="Conversation session ID")
    filters: Optional[dict] = Field(None, description="Metadata filters (e.g. doc_id, page)")
    history: Optional[List[dict]] = Field(None, description="Past messages in the conversation")


class ChatHistoryInfo(BaseModel):
    """Metadata about a past chat session."""
    session_id: str
    title: str
    updated_at: str
    doc_id: Optional[str] = None


class ChatHistoryListResponse(BaseModel):
    """List of past chat sessions."""
    sessions: List[ChatHistoryInfo] = []


class ChatSessionResponse(BaseModel):
    """A full chat session with messages."""
    session_id: str
    title: str
    messages: List[dict]
    doc_id: Optional[str] = None


class SourceCitation(BaseModel):
    """A single source chunk backing the answer."""

    document: str
    page: int
    chunk_index: int
    text: str
    score: float


class ChatResponse(BaseModel):
    """Response returned to the user."""

    answer: str
    sources: List[SourceCitation] = []
    session_id: str
    model: str
    latency_ms: float
    retrieval_mode: str = "top_k"


# ── Documents ───────────────────────────────────────────────


class DocumentInfo(BaseModel):
    """Metadata about an indexed document."""

    doc_id: str
    filename: str
    page_count: int
    chunk_count: int
    indexed_at: datetime


class DocumentListResponse(BaseModel):
    """List of all indexed documents."""

    documents: List[DocumentInfo] = []
    total: int


class UploadResponse(BaseModel):
    """Result after uploading & ingesting a PDF."""

    doc_id: str
    filename: str
    page_count: int
    chunk_count: int
    message: str = "Document indexed successfully"


# ── Health ──────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    vector_store: str = ""
    llm_provider: str = ""
