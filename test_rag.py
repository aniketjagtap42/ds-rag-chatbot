# test_rag.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE: Tests the RAG chain — vectorstore + LLM generation.
#          Verifies retrieval returns results and chain produces answers.
#
# RUN: pytest test_rag.py -v
# NOTE: Requires chroma_db/ to exist (run build_index.py first)
#       Requires GROQ_API_KEY in .env (makes real API calls)
# ─────────────────────────────────────────────────────────────────────

import pytest
from retrieval.vectorstore import load_vectorstore, get_retriever
from generation.llm_client import build_rag_chain


@pytest.fixture(scope="module")
def rag_chain():
    """
    Build the RAG chain once and reuse across all tests in this file.
    scope="module" → fixture created once per file, not once per test.
    Saves 10-15 seconds of model loading time.
    """
    db = load_vectorstore()            # load ChromaDB from disk
    retriever = get_retriever(db)      # wrap as LangChain retriever
    chain = build_rag_chain(retriever) # assemble LCEL pipeline
    return chain


@pytest.fixture(scope="module")
def retriever():
    """Retriever fixture reused across retrieval tests."""
    db = load_vectorstore()
    return get_retriever(db)


def test_vectorstore_loads():
    """ChromaDB loads without error and contains vectors."""
    db = load_vectorstore()
    count = db._collection.count()
    assert count > 0, (
        "Vector store is empty. Run 'python build_index.py' first."
    )


def test_retriever_returns_results(retriever):
    """Retriever returns top_k chunks for a valid DS question."""
    from config.settings import settings
    docs = retriever.invoke("What is a lambda function?")
    assert len(docs) == settings.top_k, (
        f"Expected {settings.top_k} chunks, got {len(docs)}"
    )


def test_retriever_chunks_have_content(retriever):
    """Every retrieved chunk has non-empty page_content."""
    docs = retriever.invoke("What is pandas DataFrame?")
    for doc in docs:
        assert len(doc.page_content) > 0, "Retrieved chunk has empty content"


def test_rag_chain_returns_string(rag_chain):
    """RAG chain returns a non-empty string answer."""
    answer = rag_chain.invoke("What is list comprehension?")
    assert isinstance(answer, str), f"Expected str, got {type(answer)}"
    assert len(answer) > 10, "Answer is suspiciously short"


def test_rag_chain_in_scope_question(rag_chain):
    """RAG chain answers a valid DS question."""
    answer = rag_chain.invoke("What is overfitting in machine learning?")
    assert isinstance(answer, str)
    assert len(answer) > 20