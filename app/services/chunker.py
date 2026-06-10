"""Text chunking with metadata preservation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import tiktoken

from app.core.config import settings
from app.services.pdf_parser import PageContent


@dataclass
class Chunk:
    """A single text chunk with full provenance metadata."""

    chunk_id: str
    doc_id: str
    filename: str
    page_num: int
    chunk_index: int
    text: str
    token_count: int
    metadata: dict = field(default_factory=dict)


def _count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens using tiktoken."""
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


def _split_text_recursive(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: List[str] | None = None,
) -> List[str]:
    """
    Recursively split text using a hierarchy of separators.
    Falls back to character-level splitting as a last resort.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    final_chunks: List[str] = []

    # Find the best separator that exists in the text
    separator = separators[-1]
    remaining_separators = []
    for i, sep in enumerate(separators):
        if sep in text:
            separator = sep
            remaining_separators = separators[i + 1 :]
            break

    # Split using the chosen separator
    splits = text.split(separator) if separator else list(text)

    current_chunk: List[str] = []
    current_length = 0

    for piece in splits:
        piece_len = len(piece)
        if current_length + piece_len + len(separator) > chunk_size and current_chunk:
            # Emit current chunk
            chunk_text = separator.join(current_chunk).strip()
            if chunk_text:
                # If still too large, recurse with smaller separators
                if len(chunk_text) > chunk_size and remaining_separators:
                    final_chunks.extend(
                        _split_text_recursive(chunk_text, chunk_size, chunk_overlap, remaining_separators)
                    )
                else:
                    final_chunks.append(chunk_text)

            # Keep overlap from the end of current chunk
            overlap_chunks: List[str] = []
            overlap_len = 0
            for item in reversed(current_chunk):
                if overlap_len + len(item) > chunk_overlap:
                    break
                overlap_chunks.insert(0, item)
                overlap_len += len(item)

            current_chunk = overlap_chunks
            current_length = overlap_len

        current_chunk.append(piece)
        current_length += piece_len + len(separator)

    # Emit remaining
    remaining = separator.join(current_chunk).strip()
    if remaining:
        if len(remaining) > chunk_size and remaining_separators:
            final_chunks.extend(
                _split_text_recursive(remaining, chunk_size, chunk_overlap, remaining_separators)
            )
        else:
            final_chunks.append(remaining)

    return final_chunks


def chunk_pages(pages: List[PageContent]) -> List[Chunk]:
    """
    Split extracted pages into overlapping chunks with metadata.

    Uses settings.chunk_size and settings.chunk_overlap (in characters,
    roughly ~4 chars per token).
    """
    chunk_size = settings.chunk_size * 4  # approximate char count
    chunk_overlap = settings.chunk_overlap * 4
    all_chunks: List[Chunk] = []
    global_index = 0

    for page in pages:
        if not page.text:
            continue

        text_splits = _split_text_recursive(page.text, chunk_size, chunk_overlap)

        for split_text in text_splits:
            token_count = _count_tokens(split_text)
            chunk = Chunk(
                chunk_id=f"{page.doc_id}_chunk_{global_index}",
                doc_id=page.doc_id,
                filename=page.filename,
                page_num=page.page_num,
                chunk_index=global_index,
                text=split_text,
                token_count=token_count,
                metadata={
                    **page.metadata,
                    "source": page.filename,
                    "page": page.page_num,
                },
            )
            all_chunks.append(chunk)
            global_index += 1

    return all_chunks
