"""PDF text extraction with metadata."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """Extracted text from a single PDF page."""

    doc_id: str
    filename: str
    page_num: int  # 1-indexed
    text: str
    metadata: dict = field(default_factory=dict)


def generate_doc_id(filepath: str) -> str:
    """Deterministic document ID from file content hash."""
    h = hashlib.sha256(Path(filepath).read_bytes()).hexdigest()[:16]
    return h


def _ocr_page(page: fitz.Page) -> str:
    """Run OCR on a page that has no extractable text.

    Uses Ollama vision model if LLM_PROVIDER=ollama, otherwise Gemini Vision.
    Returns extracted text or empty string on failure.
    """
    from app.core.config import settings

    provider = settings.llm_provider.lower()

    if provider == "ollama":
        return _ocr_page_ollama(page)
    else:
        return _ocr_page_gemini(page)


def _ocr_page_ollama(page: fitz.Page) -> str:
    """OCR via Ollama vision model (llama3.2-vision or llava)."""
    try:
        import base64
        import requests as _requests
        from app.core.config import settings

        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        b64_image = base64.b64encode(img_bytes).decode("utf-8")

        url = f"{settings.ollama_base_url}/api/chat"
        payload = {
            "model": "llama3.2-vision",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Extract ALL text from this image exactly as it appears. "
                        "Preserve the original structure, headings, bullet points, "
                        "and formatting. Return only the extracted text, nothing else."
                    ),
                    "images": [b64_image],
                }
            ],
            "stream": False,
        }

        logger.info("Attempting Ollama vision OCR...")
        resp = _requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return (data.get("message", {}).get("content", "") or "").strip()
    except Exception as e:
        logger.warning(f"Ollama vision OCR failed (is llama3.2-vision pulled?): {e}")
        logger.info("Skipping OCR for this page. Upload a text-based PDF for best results.")
        return ""


def _ocr_page_gemini(page: fitz.Page) -> str:
    """OCR via Gemini Vision API (used when LLM_PROVIDER=gemini)."""
    try:
        import base64
        from google import genai
        from google.genai import types
        from app.core.config import settings

        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")

        client = genai.Client(api_key=settings.gemini_api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                        types.Part.from_text(
                            text="Extract ALL text from this image exactly as it appears. "
                            "Preserve the original structure, headings, bullet points, "
                            "and formatting. Return only the extracted text, nothing else."
                        ),
                    ],
                )
            ],
            config=types.GenerateContentConfig(temperature=0.0),
        )
        return (response.text or "").strip()
    except Exception as e:
        logger.warning(f"Gemini Vision OCR failed for page: {e}")
        return ""


def parse_pdf(filepath: str) -> List[PageContent]:
    """
    Extract text from every page of a PDF.

    Falls back to OCR (pytesseract) for scanned / image-based pages.
    Returns a list of PageContent objects, one per page,
    with metadata attached for downstream use.
    """
    filepath = str(filepath)
    doc_id = generate_doc_id(filepath)
    filename = Path(filepath).name
    pages: List[PageContent] = []

    doc = fitz.open(filepath)
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if not text or not text.strip():
            # Fallback: try extracting from blocks for scanned-like pages
            blocks = page.get_text("blocks")
            text = "\n".join(b[4] for b in blocks if b[6] == 0)  # type: ignore[index]

        # If still no text, try OCR
        if not text or not text.strip():
            logger.info(f"No text on page {page_num} of {filename}, attempting OCR...")
            text = _ocr_page(page)

        pages.append(
            PageContent(
                doc_id=doc_id,
                filename=filename,
                page_num=page_num,
                text=text.strip(),
                metadata={
                    "total_pages": len(doc),
                    "width": page.rect.width,
                    "height": page.rect.height,
                },
            )
        )
    doc.close()

    return pages
