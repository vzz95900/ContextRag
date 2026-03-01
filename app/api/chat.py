"""Chat endpoint — query the RAG pipeline."""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.models.schemas import ChatRequest, ChatResponse, SourceCitation
from app.services.llm_chain import generate_answer
from app.services.reranker import rerank
from app.services.retriever import retrieve

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Process a user query through the full RAG pipeline."""
    start = time.perf_counter()

    try:
        # 1. Retrieve candidates
        t0 = time.perf_counter()
        candidates = retrieve(
            query=req.query,
            filters=req.filters,
        )
        logger.info("Retrieve: %.1fs (%d candidates)", time.perf_counter() - t0, len(candidates))

        # 2. Rerank
        t1 = time.perf_counter()
        top_chunks = rerank(query=req.query, candidates=candidates)
        logger.info("Rerank: %.1fs (%d chunks)", time.perf_counter() - t1, len(top_chunks))

        # 3. Generate grounded answer
        t2 = time.perf_counter()
        answer = generate_answer(query=req.query, chunks=top_chunks)
        logger.info("LLM generate: %.1fs", time.perf_counter() - t2)

        # 4. Build source citations
        sources = [
            SourceCitation(
                document=c.get("metadata", {}).get("filename", "unknown"),
                page=c.get("metadata", {}).get("page_num", 0),
                chunk_index=c.get("metadata", {}).get("chunk_index", 0),
                text=c["text"][:300],  # truncate for response
                score=round(c.get("rerank_score", c.get("score", 0.0)), 4),
            )
            for c in top_chunks
        ]

        elapsed = (time.perf_counter() - start) * 1000

        return ChatResponse(
            answer=answer,
            sources=sources,
            session_id=req.session_id or str(uuid.uuid4()),
            model=f"{settings.llm_provider}/{settings.llm_model}",
            latency_ms=round(elapsed, 1),
        )

    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            raise HTTPException(
                status_code=429,
                detail=(
                    "Rate limit reached. Please wait 1-2 minutes and try again."
                ),
            )
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")
