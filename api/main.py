# api/main.py
# ─────────────────────────────────────────────────────
# FastAPI application entry point.
# Defines all API endpoints and middleware.
#
# STARTUP FLOW:
# 1. lifespan() runs on startup
# 2. Loads vector store from chroma_db/
# 3. Builds retriever + RAG chain + RouterAgent
# 4. All endpoints are now ready to handle requests
#
# ENDPOINTS:
# GET  /          → health check (simple)
# GET  /health    → detailed health check
# POST /chat      → main chat endpoint
# DELETE /session/{session_id} → clear chat history
# GET  /docs      → auto-generated API documentation
# ─────────────────────────────────────────────────────

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from retrieval.vectorstore import load_vectorstore, get_retriever
from generation.llm_client import build_rag_chain
from agents.router_agent import RouterAgent, session_store
from config.settings import settings
from config.logger import get_logger

logger = get_logger(__name__)

# ── Global state ──────────────────────────────────────
# Loaded once on startup, reused for every request
# Loading on every request would be too slow (2-3 seconds)
db = None
agent = None


# ── Lifespan ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on application startup and shutdown.

    STARTUP (before yield):
    - Load vector store from disk
    - Build retriever, RAG chain, agent
    - App is ready to handle requests

    SHUTDOWN (after yield):
    - Clean up resources if needed
    - Currently nothing to clean up

    WHY LIFESPAN INSTEAD OF @app.on_event:
    lifespan is the modern FastAPI approach
    @app.on_event("startup") is deprecated
    """
    global db, agent

    logger.info("=" * 50)
    logger.info("DS Notes Assistant starting up")
    logger.info("=" * 50)

    try:
        # ── Load components in order ───────────────────
        logger.info("Step 1/4: Loading vector store")
        db = load_vectorstore()

        logger.info("Step 2/4: Creating retriever")
        retriever = get_retriever(db)

        logger.info("Step 3/4: Building RAG chain")
        rag_chain = build_rag_chain(retriever)

        logger.info("Step 4/4: Initialising RouterAgent")
        agent = RouterAgent(rag_chain, retriever)

        logger.info("=" * 50)
        logger.info("DS Notes Assistant is READY")
        logger.info(f"Allowed origins: {settings.allowed_origins}")
        logger.info("=" * 50)

    except FileNotFoundError as e:
        # ── chroma_db not found ────────────────────────
        # Clear message telling developer what to do
        logger.error(f"Startup failed: {e}")
        logger.error("Run 'python build_index.py' first")
        raise

    except Exception as e:
        logger.error(f"Startup failed unexpectedly: {e}", exc_info=True)
        raise

    # ── App runs here ──────────────────────────────────
    yield

    # ── Shutdown ───────────────────────────────────────
    logger.info("DS Notes Assistant shutting down")


# ── FastAPI App ───────────────────────────────────────
app = FastAPI(
    title="DS Notes Assistant",
    description=(
        "RAG chatbot that answers questions from "
        "your Data Science and Python study notes"
    ),
    version="1.0.0",
    lifespan=lifespan
)


# ── CORS Middleware ───────────────────────────────────
# FIXED: was allow_origins=["*"] — security risk
# Now reads from settings.allowed_origins
# Set in .env: ALLOWED_ORIGINS=http://localhost:3000
# Multiple origins separated by comma in .env
allowed_origins = [
    origin.strip()
    for origin in settings.allowed_origins.split(",")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    # ── Allow all methods and headers ─────────────────
    # Locked down origins is enough for this project
    # In banking/healthcare you'd restrict methods too
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
    allow_credentials=True
)


# ── Request Logging Middleware ────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log every incoming request and its response time.

    This middleware wraps EVERY request automatically.
    No need to add logging to each endpoint individually.

    Logs:
        → method, path, status code, response time
    """
    start_time = time.time()

    # ── Process the request ────────────────────────────
    response = await call_next(request)

    elapsed = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} | "
        f"status: {response.status_code} | "
        f"took: {elapsed:.3f}s"
    )

    return response


# ── Request / Response Models ─────────────────────────
class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    sources: list
    in_scope: bool
    session_id: str
    rewritten_query: str


# ── Endpoints ─────────────────────────────────────────

@app.get("/")
def root():
    """Simple root endpoint — confirms API is running."""
    return {
        "message": "DS Notes Assistant is running",
        "status": "ok",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    """
    Detailed health check endpoint.

    Used by:
    - Docker health checks
    - Load balancers
    - Monitoring tools
    - Developers verifying deployment

    Returns agent_ready so you know if startup completed.
    """
    return {
        "status": "healthy",
        "agent_ready": agent is not None,
        "version": "1.0.0"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint — processes question and returns answer.

    Flow:
    1. Validate request
    2. Pass to RouterAgent
    3. Return structured response

    All RAG logic is in RouterAgent, not here.
    API layer only handles HTTP concerns.
    """
    # ── Validate agent is ready ────────────────────────
    # Could be not ready if startup is still in progress
    if agent is None:
        logger.warning("Chat request received but agent not ready")
        raise HTTPException(
            status_code=503,
            detail="Server is still starting up. Please try again."
        )

    # ── Validate input ─────────────────────────────────
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    # ── Enforce question length limit ─────────────────
    # Prevents abuse and prompt injection attempts
    if len(request.question) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Question too long. Maximum 1000 characters."
        )

    logger.info(
        f"Chat request | "
        f"session: {request.session_id} | "
        f"question: '{request.question[:80]}'"
    )

    # ── Call agent with error handling ────────────────
    # FIXED: was no try/except — raw errors reached user
    # Now all errors return clean message
    try:
        result = agent.run(
            question=request.question,
            session_id=request.session_id
        )

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            in_scope=result["in_scope"],
            session_id=request.session_id,
            rewritten_query=result["rewritten_query"]
        )

    except Exception as e:
        # ── Log full error internally ──────────────────
        # exc_info=True captures the full stack trace
        # in logs — but user never sees it
        logger.error(
            f"Chat endpoint failed | "
            f"session: {request.session_id} | "
            f"error: {e}",
            exc_info=True
        )
        # ── Return clean error to user ─────────────────
        # Never expose internal details to users
        raise HTTPException(
            status_code=500,
            detail="Something went wrong. Please try again."
        )


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """
    Clear conversation history for a session.

    Called when user clicks 'New Chat' in the UI.
    Removes all messages for that session_id.
    """
    if session_id in session_store:
        del session_store[session_id]
        logger.info(f"Session cleared: {session_id}")
        return {
            "message": f"Session {session_id} cleared successfully"
        }

    logger.warning(f"Clear requested for unknown session: {session_id}")
    return {
        "message": f"Session {session_id} not found"
    }