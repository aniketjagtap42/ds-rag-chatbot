# test_api.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE: Integration tests — calls the running API via HTTP.
#          Tests real end-to-end behaviour through the full stack.
#
# RUN: pytest test_api.py -v
# REQUIRES: API must be running at localhost:8000
#           Start with: docker-compose up  OR  uvicorn api.main:app
# ─────────────────────────────────────────────────────────────────────

import pytest
import requests    # HTTP client to call the live API

# ── Base URL of the running API ───────────────────────────────────────
BASE_URL = "http://localhost:8000"


@pytest.fixture(autouse=True)
def check_api_running():
    """
    Skip all tests if API is not running.
    autouse=True → runs before every test in this file automatically.
    Gives a clear message instead of confusing ConnectionRefusedError.
    """
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code not in (200, 503):
            pytest.skip("API is not running at localhost:8000")
    except requests.ConnectionError:
        pytest.skip("API is not running — start with: docker-compose up")


def test_health_endpoint():
    """Health endpoint returns 200 when everything is loaded."""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["checks"]["agent"] is True
    assert data["checks"]["vectorstore"] is True


def test_root_endpoint():
    """Root endpoint returns status ok."""
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_in_scope_question():
    """Valid DS question returns an answer with sources."""
    response = requests.post(f"{BASE_URL}/chat", json={
        "question": "What is list comprehension?",
        "session_id": "test_api_1"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["in_scope"] is True
    assert len(data["answer"]) > 10
    assert isinstance(data["sources"], list)


def test_chat_out_of_scope_question():
    """Off-topic question returns in_scope=False."""
    response = requests.post(f"{BASE_URL}/chat", json={
        "question": "What is the weather in Pune today?",
        "session_id": "test_api_2"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["in_scope"] is False


def test_chat_empty_question_rejected():
    """Empty question returns 422 validation error."""
    response = requests.post(f"{BASE_URL}/chat", json={
        "question": "",
        "session_id": "test_api_3"
    })
    # 422 = Unprocessable Entity — Pydantic validation failed
    assert response.status_code == 422


def test_chat_question_too_long():
    """Question over 1000 chars returns 422 validation error."""
    response = requests.post(f"{BASE_URL}/chat", json={
        "question": "a" * 1001,   # 1001 chars — over the 1000 limit
        "session_id": "test_api_4"
    })
    assert response.status_code == 422


def test_clear_session():
    """DELETE /session clears the session successfully."""
    # first create a session by sending a message
    requests.post(f"{BASE_URL}/chat", json={
        "question": "What is Python?",
        "session_id": "test_clear_session"
    })
    # then delete it
    response = requests.delete(f"{BASE_URL}/session/test_clear_session")
    assert response.status_code == 200
    assert "cleared" in response.json()["message"].lower()


def test_response_has_all_fields():
    """Chat response always contains all required fields."""
    response = requests.post(f"{BASE_URL}/chat", json={
        "question": "What is pandas?",
        "session_id": "test_fields"
    })
    data = response.json()
    required = {"answer", "sources", "in_scope", "session_id", "rewritten_query"}
    assert required.issubset(data.keys()), (
        f"Response missing fields: {required - data.keys()}"
    )