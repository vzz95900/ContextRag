"""Bulk-ingest all PDFs from data/pdfs/ into the vector store.

Usage:
    cd d:\Pjts\CntextAware
    .venv\Scripts\python scripts\ingest_pdfs.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so `app.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings  # noqa: E402
from app.services.chunker import chunk_pages  # noqa: E402
from app.services.embedder import embed_texts  # noqa: E402
from app.services.pdf_parser import parse_pdf, generate_doc_id  # noqa: E402
from app.services.vector_store import get_vector_store  # noqa: E402


PDF_FOLDER = PROJECT_ROOT / "data" / "pdfs"


def ingest_all():
    """Find every PDF in data/pdfs/ and index it."""
    settings.ensure_dirs()
    PDF_FOLDER.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️  No PDFs found in {PDF_FOLDER}")
        print("   Drop your PDF files there and run this script again.")
        return

    store = get_vector_store()
    # Grab already-indexed doc IDs so we can skip duplicates
    existing_docs = {d["doc_id"] for d in store.list_documents()}

    total = len(pdf_files)
    indexed = 0
    skipped = 0
    failed = 0

    print(f"\n📂 Found {total} PDF(s) in {PDF_FOLDER}\n")

    for i, pdf_path in enumerate(pdf_files, 1):
        name = pdf_path.name
        doc_id = generate_doc_id(str(pdf_path))

        if doc_id in existing_docs:
            print(f"  [{i}/{total}] ⏭️  {name} — already indexed, skipping")
            skipped += 1
            continue

        print(f"  [{i}/{total}] 📄 {name}")
        t0 = time.time()

        try:
            # 1. Parse
            pages = parse_pdf(str(pdf_path))
            if not pages:
                print(f"           ⚠️  No text extracted — skipping")
                failed += 1
                continue

            # 2. Chunk
            chunks = chunk_pages(pages)
            if not chunks:
                print(f"           ⚠️  No chunks produced — skipping")
                failed += 1
                continue

            # 3. Embed
            texts = [c.text for c in chunks]
            embeddings = embed_texts(texts)

            # 4. Store
            store.add_chunks(chunks, embeddings)

            elapsed = time.time() - t0
            print(f"           ✅ {len(pages)} pages · {len(chunks)} chunks · {elapsed:.1f}s")
            indexed += 1

        except Exception as e:
            print(f"           ❌ Failed: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"✅ Indexed: {indexed}  ⏭️ Skipped: {skipped}  ❌ Failed: {failed}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    ingest_all()
