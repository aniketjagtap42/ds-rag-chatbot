# test_agent.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE: Tests the RouterAgent — scope check, query rewriting, memory.
#
# RUN: pytest test_agent.py -v
# NOTE: Makes real Groq API calls — requires GROQ_API_KEY in .env
# ─────────────────────────────────────────────────────────────────────

import pytest
from retrieval.vectorstore import load_vectorstore, get_retriever
from generation.llm_client import build_rag_chain
from agents.router_agent import RouterAgent, session_store


@pytest.fixture(scope="module")
def agent():
    """Build RouterAgent once and reuse across all tests."""
    db = load_vectorstore()
    retriever = get_retriever(db)
    chain = build_rag_chain(retriever)
    return RouterAgent(chain, retriever)


def test_in_scope_question(agent):
    """DS/Python question is answered (not rejected)."""
    result = agent.run("What is a lambda function?", session_id="test_scope")
    assert result["in_scope"] is True, "Valid DS question incorrectly rejected"
    assert len(result["answer"]) > 10


def test_out_of_scope_question(agent):
    """Non-DS question is rejected with in_scope=False."""
    result = agent.run("What is the weather in Pune today?", session_id="test_scope2")
    assert result["in_scope"] is False, "Off-topic question should be rejected"
    assert result["sources"] == [], "Out-of-scope should return empty sources"


def test_response_has_required_keys(agent):
    """Response dict always contains all required keys."""
    result = agent.run("What is pandas?", session_id="test_keys")
    required_keys = {"answer", "sources", "in_scope", "rewritten_query"}
    assert required_keys.issubset(result.keys()), (
        f"Response missing keys: {required_keys - result.keys()}"
    )


def test_session_memory_saves(agent):
    """Session history is saved after each exchange."""
    sid = "test_memory_save"
    agent.run("What is a lambda function?", session_id=sid)
    assert sid in session_store, "Session was not saved to session_store"
    assert len(session_store[sid]) >= 2, "Expected at least 2 messages in history"


def test_session_memory_followup(agent):
    """Follow-up question is rewritten using session context."""
    sid = "test_followup"
    agent.run("What is a lambda function?", session_id=sid)
    result = agent.run("Give me a code example of that", session_id=sid)

    # rewritten query should NOT still say "that" — it should be resolved
    assert "that" not in result["rewritten_query"].lower() or \
           "lambda" in result["rewritten_query"].lower(), (
        f"Query not rewritten properly: {result['rewritten_query']}"
    )