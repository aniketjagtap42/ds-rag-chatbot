# test_ingestion.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE: Tests the ingestion pipeline — loader and chunker.
#          Verifies documents load correctly and chunks are valid.
#
# RUN: pytest test_ingestion.py -v
# ─────────────────────────────────────────────────────────────────────

import pytest                              # pytest framework — provides assert, fixtures, test runner
from ingestion.loader import load_document     # function under test
from ingestion.chunker import chunk_documents  # function under test


def test_load_document_returns_content():
    """Document loads and returns non-empty content."""
    docs = load_document()
    assert len(docs) > 0, "Loader returned empty list — check data_path in settings"
    assert len(docs[0].page_content) > 0, "Document loaded but page_content is empty"


def test_load_document_has_metadata():
    """Each document has a source in metadata."""
    docs = load_document()
    for doc in docs:
        # metadata should contain 'source' key with the file path
        assert "source" in doc.metadata, f"Document missing 'source' in metadata: {doc.metadata}"


def test_chunk_documents_returns_chunks():
    """Chunking produces a non-empty list of chunks."""
    docs = load_document()
    chunks = chunk_documents(docs)
    assert len(chunks) > 0, "Chunker returned 0 chunks — document may have no text"


def test_chunk_size_within_limit():
    """No chunk exceeds the configured chunk_size significantly."""
    from config.settings import settings
    docs = load_document()
    chunks = chunk_documents(docs)

    # allow 10% tolerance — splitter may slightly exceed chunk_size
    # at natural boundaries like paragraph ends
    max_allowed = settings.chunk_size * 1.1

    oversized = [
        c for c in chunks
        if len(c.page_content) > max_allowed
    ]
    assert len(oversized) == 0, (
        f"{len(oversized)} chunks exceed size limit of {max_allowed:.0f} chars"
    )


def test_chunks_inherit_metadata():
    """Each chunk inherits source metadata from parent document."""
    docs = load_document()
    chunks = chunk_documents(docs)
    for chunk in chunks:
        assert "source" in chunk.metadata, "Chunk missing source metadata"


def test_chunk_documents_rejects_empty_input():
    """chunk_documents raises ValueError on empty input."""
    with pytest.raises(ValueError, match="empty documents list"):
        chunk_documents([])