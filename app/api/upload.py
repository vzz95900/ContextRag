"""PDF upload & ingestion endpoint."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import settings
from app.models.schemas import DocumentInfo, DocumentListResponse, UploadResponse
from app.services.chunker import chunk_pages
from app.services.embedder import embed_texts
from app.services.pdf_parser import generate_doc_id, parse_pdf
from app.services.vector_store import get_vector_store

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF, parse it, chunk it, embed it, and index it."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Save uploaded file
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    filepath = upload_dir / file.filename

    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # 1. Parse PDF
        pages = parse_pdf(str(filepath))
        if not pages:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

        # 2. Chunk
        chunks = chunk_pages(pages)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No text could be extracted from this PDF. "
                    "It may be a scanned/image-based document. "
                    "Try uploading a text-based PDF (exported from Word/Google Docs) for best results."
                ),
            )

        # 3. Embed
        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts)

        # 4. Index
        store = get_vector_store()
        store.add_chunks(chunks, embeddings)

        doc_id = generate_doc_id(str(filepath))
        return UploadResponse(
            doc_id=doc_id,
            filename=file.filename,
            page_count=len(pages),
            chunk_count=len(chunks),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@router.get("/documents", response_model=DocumentListResponse)
def list_documents():
    """List all indexed documents."""
    store = get_vector_store()
    docs = store.list_documents()
    return DocumentListResponse(
        documents=[
            DocumentInfo(
                doc_id=d["doc_id"],
                filename=d["filename"],
                page_count=d["page_count"],
                chunk_count=d["chunk_count"],
                indexed_at=datetime.now(timezone.utc),
            )
            for d in docs
        ],
        total=len(docs),
    )


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    """Remove a document from the vector store."""
    store = get_vector_store()
    deleted = store.delete_document(doc_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"message": f"Deleted {deleted} chunks for document {doc_id}"}
