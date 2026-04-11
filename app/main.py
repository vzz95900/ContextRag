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

from app.api import chat, health, upload
from app.core.config import settings

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle hook."""
    # ── Startup ──
    settings.ensure_dirs()
    yield
    # ── Shutdown ──


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
