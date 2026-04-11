"""Chat endpoint — query the RAG pipeline."""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, HTTPException, Query

from app.core.config import settings
from app.models.schemas import ChatRequest, ChatResponse, SourceCitation, ChatHistoryListResponse, ChatSessionResponse
from app.services.llm_chain import generate_answer
from app.services.reranker import rerank
from app.services.retriever import retrieve
from app.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Process a user query through the full RAG pipeline."""
    start = time.perf_counter()

    try:
        # 1. Retrieve candidates
        t0 = time.perf_counter()
        
        # Heuristic Context Expansion for Retrieval
        # Short conversational intents (e.g. "Answer in detail") lack semantic keywords.
        # We prepend the previous user question to inject topic grounding without extra LLM latency.
        retrieval_query = req.query
        if req.history and len(req.query.split()) < 20:
            last_user_msgs = [m.get("content", "") for m in req.history if m.get("role") == "user"]
            if last_user_msgs:
                retrieval_query = f"{last_user_msgs[-1]} {req.query}"
                logger.info(f"Expanded retrieval query: '{retrieval_query}'")
                
        candidates = retrieve(
            query=retrieval_query,
            filters=req.filters,
        )
        logger.info("Retrieve: %.1fs (%d candidates)", time.perf_counter() - t0, len(candidates))

        # 2. Rerank
        t1 = time.perf_counter()
        top_chunks = rerank(query=retrieval_query, candidates=candidates)
        logger.info("Rerank: %.1fs (%d chunks)", time.perf_counter() - t1, len(top_chunks))

        # 3. Generate grounded answer
        t2 = time.perf_counter()
        answer = generate_answer(query=req.query, chunks=top_chunks, history=req.history)
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
        
        # If the LLM explicitly refused to answer due to lack of context, hide the citations
        if "I don't see the answer" in answer or "I don't have enough information" in answer:
            sources = []

        elapsed = (time.perf_counter() - start) * 1000
        session_id = req.session_id or str(uuid.uuid4())
        model_name = f"{settings.llm_provider}/{settings.llm_model}"

        # 5. Save chat history to vector store
        title = req.query[:50]
        if req.history and len(req.history) > 0:
            title = req.history[0].get("content", req.query)[:50]
            
        updated_history = list(req.history) if req.history else []
        updated_history.append({"role": "user", "content": req.query})
        updated_history.append({
            "role": "assistant",
            "content": answer,
            "sources": [s.model_dump() for s in sources],
            "latency": round(elapsed, 1),
            "model": model_name
        })
        
        doc_scope_id = None
        if req.filters and isinstance(req.filters, dict):
            doc_scope_id = req.filters.get("doc_id")

        try:
            store = get_vector_store()
            store.save_chat(session_id, title, updated_history, doc_id=doc_scope_id)
        except Exception as e:
            logger.error(f"Failed to save chat history: {e}")

        return ChatResponse(
            answer=answer,
            sources=sources,
            session_id=session_id,
            model=model_name,
            latency_ms=round(elapsed, 1),
            retrieval_mode="optimized" if settings.enable_optimizer else "top_k",
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
        
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chats", response_model=ChatHistoryListResponse)
def list_chats(doc_id: str | None = Query(default=None, description="Filter chat sessions by selected document scope")):
    """Return all past chat sessions."""
    try:
        store = get_vector_store()
        sessions = store.list_chats(doc_id=doc_id)
        return ChatHistoryListResponse(sessions=sessions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch chats: {e}")

@router.get("/chats/{session_id}", response_model=ChatSessionResponse)
def get_chat(session_id: str):
    """Retrieve a specific chat history."""
    try:
        store = get_vector_store()
        chat = store.get_chat(session_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat session not found")
        return ChatSessionResponse(**chat)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch chat: {e}")

@router.delete("/chats/{session_id}")
def delete_chat(session_id: str):
    """Delete a specific chat history."""
    try:
        store = get_vector_store()
        store.delete_chat(session_id)
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete chat: {e}")
