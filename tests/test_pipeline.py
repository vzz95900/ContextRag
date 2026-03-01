"""Basic tests for the RAG pipeline components."""

from app.core.config import Settings


def test_settings_defaults():
    """Verify default settings load correctly."""
    s = Settings(gemini_api_key="test-key")
    assert s.llm_provider == "gemini"
    assert s.embedding_provider == "gemini"
    assert s.chunk_size == 512
    assert s.chunk_overlap == 64
    assert s.retrieval_top_k == 20
    assert s.rerank_top_n == 5


def test_chunker_splits_text():
    """Verify text chunking produces expected output."""
    from app.services.chunker import _split_text_recursive

    text = "Hello world. " * 200  # ~2600 chars
    chunks = _split_text_recursive(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 600  # allow some tolerance


def test_format_context():
    """Verify context formatting for LLM prompt."""
    from app.services.llm_chain import _format_context

    chunks = [
        {"text": "Sample text", "metadata": {"filename": "test.pdf", "page_num": 1}},
        {"text": "More text", "metadata": {"filename": "test.pdf", "page_num": 2}},
    ]
    ctx = _format_context(chunks)
    assert "test.pdf" in ctx
    assert "Page 1" in ctx
    assert "Sample text" in ctx
