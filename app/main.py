"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.api import chat, health, upload, auth
from app.core.config import settings

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


import asyncio

async def periodic_guest_cleanup():
    while True:
        try:
            await asyncio.sleep(3600)  # Clean up every hour
            from app.services.auth import cleanup_old_guest_sessions
            cleanup_old_guest_sessions()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logging.getLogger("app.main").error(f"Periodic guest cleanup failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle hook."""
    # ── Startup ──
    settings.ensure_dirs()
    
    # Initialize SQLite User DB and clean up old sessions
    from app.services.auth import init_db, cleanup_old_guest_sessions
    init_db()
    try:
        cleanup_old_guest_sessions()
    except Exception as e:
        logging.getLogger("app.main").error(f"Failed to clean up old guest sessions on start: {e}")

    # Eagerly initialize vector store to avoid cold start lag on first search
    from app.services.vector_store import get_vector_store
    get_vector_store()
    
    # Eagerly load cross-encoder reranker model to avoid cold start lag on first request
    if settings.enable_reranker:
        from app.services.reranker import _get_cross_encoder
        _get_cross_encoder()
        
    cleanup_task = asyncio.create_task(periodic_guest_cleanup())
    
    yield
    # ── Shutdown ──
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="Context-Aware RAG Engine",
    description="Semantic search & QA over PDF collections with source-grounded answers.",
    version="0.1.0",
    lifespan=lifespan,
)

# ── CORS ────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ──────────────────────────────────────────────────
app.include_router(health.router, tags=["Health"])
app.include_router(auth.router, prefix="/api", tags=["Auth"])
app.include_router(upload.router, prefix="/api", tags=["Documents"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])


# ── Frontend ────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def serve_root():
    """Serve the main chat page."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Return an empty response for browser favicon requests."""
    return Response(status_code=204)


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
