# api/main.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE: FastAPI application entry point.
#          Defines all HTTP endpoints, middleware, and startup logic.
#
# WHAT THIS FILE IS RESPONSIBLE FOR:
#   - Starting up and loading all components (vectorstore, agent)
#   - Defining HTTP routes (/chat, /health, /session, /ui)
#   - Input validation (length, empty check)
#   - Request logging (every request logged with timing)
#   - Request ID tracking (unique ID per request for debugging)
#   - CORS policy (which frontends can call this API)
#   - Clean error responses (never expose internal errors to users)
#
# WHAT THIS FILE IS NOT RESPONSIBLE FOR:
#   - RAG logic (that lives in agents/router_agent.py)
#   - Embedding logic (ingestion/embedder.py)
#   - Prompt building (generation/prompt_builder.py)
#   - This separation is intentional — API layer = HTTP concerns only
#
# STARTUP FLOW:
#   lifespan() runs → load vectorstore → build retriever
#   → build RAG chain → init RouterAgent → app ready
#
# ENDPOINTS:
#   GET    /                      → simple liveness check
#   GET    /health                → deep health check (used by K8s/Docker)
#   GET    /ui                    → serves chat_ui.html in browser
#   POST   /chat                  → main chat endpoint
#   DELETE /session/{session_id}  → clear conversation history
#   GET    /docs                  → auto-generated Swagger UI (FastAPI built-in)
# ─────────────────────────────────────────────────────────────────────

import os                                # os.path → build absolute path to chat_ui.html
import time                              # time.time() → measure request duration
import uuid                              # uuid.uuid4() → generate unique request IDs
import asyncio                           # asyncio.to_thread → run sync agent.run() without blocking event loop
from contextlib import asynccontextmanager  # @asynccontextmanager → modern FastAPI lifespan pattern

from fastapi import FastAPI, HTTPException, Request      # FastAPI core — app, errors, request object
from fastapi.middleware.cors import CORSMiddleware       # CORS middleware — controls which frontends can call API
from fastapi.responses import JSONResponse, FileResponse # JSONResponse → custom status codes | FileResponse → serve HTML file
from pydantic import BaseModel, Field, validator         # BaseModel → schemas | Field → validation rules

from retrieval.vectorstore import load_vectorstore, get_retriever  # loads ChromaDB and creates retriever
from generation.llm_client import build_rag_chain                  # assembles the LCEL RAG chain
from agents.router_agent import RouterAgent, session_store         # agent handles all RAG logic
from config.settings import settings                               # centralised config from .env
from config.logger import get_logger                               # structured JSON logger

# ── Module-level logger ───────────────────────────────────────────────
# __name__ = "api.main" → every log from this file tagged with this name
# makes filtering in CloudWatch easy: filter where module = "api.main"
logger = get_logger(__name__)

# ── Global singletons ─────────────────────────────────────────────────
# These are loaded ONCE at startup and reused for every request.
# WHY global? Loading the vectorstore and agent takes 2-4 seconds.
# Reloading on every request would make every /chat call 5+ seconds.
# These are set in lifespan() below and read in the /chat endpoint.
db = None      # ChromaDB vectorstore instance
agent = None   # RouterAgent instance (contains rag_chain + retriever)


# ─────────────────────────────────────────────────────────────────────
# LIFESPAN — Startup and Shutdown
# ─────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup (before yield) and shutdown (after yield).

    WHY lifespan instead of @app.on_event("startup")?
    @app.on_event is deprecated in FastAPI 0.95+.
    lifespan is the modern approach — cleaner, supports both
    startup and shutdown in one function, and works with pytest.

    STARTUP (code before yield):
        Loads all heavy components once.
        If any component fails → app crashes immediately with clear error.
        WHY crash? Better to fail fast at startup than to serve broken
        requests for hours before someone notices.

    SHUTDOWN (code after yield):
        Runs when server receives SIGTERM (e.g. Kubernetes pod shutdown).
        Good place to flush logs, close DB connections, etc.
    """
    global db, agent   # declare global so assignment affects module-level variables

    logger.info("=" * 60)
    logger.info("DS Notes Assistant — STARTING UP")
    logger.info("=" * 60)

    try:
        # ── STARTUP STEP 1: Load vector store ─────────────────────────
        # Reads the ChromaDB index from disk (chroma_db/ folder).
        # If chroma_db/ doesn't exist → FileNotFoundError → app exits.
        # Fix: run `python build_index.py` first to build the index.
        logger.info("Step 1/4: Loading vector store from disk...")
        db = load_vectorstore()

        # ── STARTUP STEP 2: Create retriever ──────────────────────────
        # Wraps the vectorstore as a LangChain retriever.
        # Retriever accepts a query string and returns top-k chunks.
        logger.info("Step 2/4: Creating retriever...")
        retriever = get_retriever(db)

        # ── STARTUP STEP 3: Build RAG chain ───────────────────────────
        # Assembles the full LCEL pipeline:
        #   retriever → format_docs → prompt → LLM → StrOutputParser
        logger.info("Step 3/4: Building RAG chain...")
        rag_chain = build_rag_chain(retriever)

        # ── STARTUP STEP 4: Initialise RouterAgent ────────────────────
        # RouterAgent wraps rag_chain and retriever with:
        #   scope checking, query rewriting, session memory
        logger.info("Step 4/4: Initialising RouterAgent...")
        agent = RouterAgent(rag_chain, retriever)

        logger.info("=" * 60)
        logger.info("DS Notes Assistant — READY TO SERVE REQUESTS")
        logger.info(f"Allowed CORS origins : {settings.allowed_origins}")
        logger.info(f"LLM model            : {settings.llm_model}")
        logger.info(f"Embedding model      : {settings.embedding_model}")
        logger.info("=" * 60)

    except FileNotFoundError as e:
        # ── chroma_db/ folder missing ─────────────────────────────────
        # Most common startup failure — developer forgot build_index.py
        logger.critical(f"Vector store not found: {e}")
        logger.critical("ACTION REQUIRED: run 'python build_index.py' first")
        raise

    except Exception as e:
        # ── Any other startup failure ─────────────────────────────────
        # exc_info=True → includes full Python traceback in the log
        logger.critical(f"Startup failed unexpectedly: {e}", exc_info=True)
        raise

    # ── App runs here ─────────────────────────────────────────────────
    # Everything above yield = startup
    # Everything below yield = shutdown
    yield

    # ── SHUTDOWN ──────────────────────────────────────────────────────
    # Runs when Kubernetes sends SIGTERM or docker stop is called.
    logger.info("DS Notes Assistant — SHUTTING DOWN")


# ─────────────────────────────────────────────────────────────────────
# FASTAPI APP INSTANCE
# ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DS Notes Assistant",
    description=(
        "RAG chatbot that answers questions from "
        "your Data Science and Python study notes. "
        "Powered by LangChain + Groq + ChromaDB."
    ),
    version="1.0.0",
    lifespan=lifespan
)


# ─────────────────────────────────────────────────────────────────────
# MIDDLEWARE 1: CORS
# ─────────────────────────────────────────────────────────────────────

# ── Parse allowed origins from settings ───────────────────────────────
# settings.allowed_origins is a comma-separated string from .env
# e.g. "http://localhost:3000,https://myapp.com"
# .split(",") → list → .strip() → removes accidental spaces
# WHY not use ["*"]? That allows ANY website to call your API —
# a Cross-Site Request Forgery (CSRF) security risk.
allowed_origins = [
    origin.strip()
    for origin in settings.allowed_origins.split(",")  # split into list
    if origin.strip()                                   # remove empty strings
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,                              # whitelisted frontends only
    allow_methods=["GET", "POST", "DELETE"],                    # only methods this API uses
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
    allow_credentials=True,
)


# ─────────────────────────────────────────────────────────────────────
# MIDDLEWARE 2: Request ID + Timing Logger
# ─────────────────────────────────────────────────────────────────────

@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """
    Runs for EVERY incoming HTTP request automatically.

    WHAT IT DOES:
    1. Generates a unique request_id for this request
    2. Times how long the request takes end to end
    3. Logs method, path, status code, duration after completion
    4. Attaches request_id to response header for client correlation

    WHY request_id matters in production:
    When a user says "my request failed at 2:47pm", you grep
    CloudWatch logs for that request_id and see every log line
    for that exact request — scope check, retrieval, generation.
    Without it, finding one request in thousands is near impossible.
    """
    # generate short unique ID — 8 chars is enough at our scale
    request_id = str(uuid.uuid4())[:8]

    # store on request state so downstream handlers can access it
    request.state.request_id = request_id

    start_time = time.time()

    response = await call_next(request)   # process request through all handlers

    elapsed_ms = int((time.time() - start_time) * 1000)   # milliseconds

    logger.info(
        f"request_id={request_id} | "
        f"{request.method} {request.url.path} | "
        f"status={response.status_code} | "
        f"duration={elapsed_ms}ms"
    )

    # attach to response so frontend can log and correlate it
    response.headers["X-Request-ID"] = request_id

    return response


# ─────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS (Pydantic)
# ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """
    Validates and parses the incoming /chat request body.

    WHY Pydantic Field validators instead of manual if/else?
    FastAPI automatically validates before your handler runs.
    Invalid input → 422 Unprocessable Entity with clear message.
    No manual validation code needed in the route handler.
    """

    question: str = Field(
        ...,                  # required — no default
        min_length=1,         # rejects empty string
        max_length=1000,      # SECURITY: prevents token abuse
        description="The question to ask the DS Notes Assistant",
        example="What is a lambda function in Python?"
    )

    session_id: str = Field(
        default="default",
        max_length=64,
        pattern=r"^[a-zA-Z0-9_-]+$",  # SECURITY: blocks injection via session_id
        description="Unique session identifier for conversation memory",
        example="user_123"
    )

    @validator("question")
    def strip_question(cls, v):
        # strip whitespace so "  Python?  " = "Python?"
        # also ensures min_length=1 catches whitespace-only input
        return v.strip()


class ChatResponse(BaseModel):
    """
    Defines the structure of every /chat response.

    WHY define a response model?
    1. Auto-documents in /docs Swagger UI
    2. FastAPI validates before sending — catches bugs early
    3. Frontend always knows exactly what fields to expect
    """
    answer: str           # LLM-generated answer grounded in notes
    sources: list         # retrieved chunk previews with source filename
    in_scope: bool        # True = answered | False = rejected as off-topic
    session_id: str       # echoed back to confirm which session was used
    rewritten_query: str  # the query used for vector search (for debugging)


# ─────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """
    Simple liveness check — confirms the process is alive.
    No dependencies checked. Returns immediately.
    Used by load balancers to confirm the server is up.
    """
    return {
        "message": "DS Notes Assistant is running",
        "status":  "ok",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    """
    Deep health check — checks all critical dependencies.

    Returns 200 if everything is ready, 503 if not.

    WHY 503 when not ready?
    Kubernetes readiness probe checks HTTP status code.
    200 → K8s sends traffic to this pod.
    503 → K8s holds traffic back until pod is ready.
    Returning 200 with agent_ready=False would send real
    user traffic to a broken pod.
    """
    checks = {
        "vectorstore": db is not None,    # ChromaDB loaded from disk
        "agent":       agent is not None, # RouterAgent initialised
    }
    all_healthy = all(checks.values())

    return JSONResponse(
        status_code=200 if all_healthy else 503,
        content={
            "status":  "healthy" if all_healthy else "degraded",
            "checks":  checks,
            "version": "1.0.0"
        }
    )


@app.get("/ui")
def serve_ui():
    """
    Serve chat_ui.html directly from the API server.

    WHY serve HTML from the API?
    When you open chat_ui.html as a local file (file://),
    the browser blocks fetch() calls to http://127.0.0.1:8000
    due to CORS policy — the origins don't match.

    Serving from the API means both HTML and API share the
    same origin (http://127.0.0.1:8000) → no CORS issue at all.

    Access at: http://127.0.0.1:8000/ui
    """
    # ── Build absolute path to chat_ui.html ───────────────────────────
    # __file__ = absolute path to THIS file (api/main.py)
    # dirname(__file__) = the api/ folder
    # join(..., "..", "chat_ui.html") = one level up → project root
    # abspath() resolves the ".." → clean absolute path
    # WHY absolute path? Relative paths break depending on where
    # uvicorn is launched from. Absolute path always works.
    ui_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),  # api/ folder
            "..",                        # go up to project root
            "chat_ui.html"               # the HTML file
        )
    )

    # ── Verify the file exists before trying to serve it ──────────────
    # Gives a clear 404 with the exact path if file is missing,
    # instead of a confusing internal server error.
    if not os.path.exists(ui_path):
        logger.error(f"chat_ui.html not found at: {ui_path}")
        raise HTTPException(
            status_code=404,
            detail=f"UI file not found. Expected at: {ui_path}"
        )

    logger.info(f"Serving UI from: {ui_path}")
    return FileResponse(ui_path)   # sends the HTML file as HTTP response


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint — the core of the entire application.

    WHY async def?
    FastAPI runs async routes on the event loop.
    agent.run() is sync/blocking so it runs in asyncio.to_thread()
    — this frees the event loop to handle concurrent requests.

    Flow:
        1. Pydantic validates ChatRequest (before this runs)
        2. Check agent is ready (503 if still starting up)
        3. Call agent.run() in thread pool (non-blocking)
        4. Return structured ChatResponse
        5. Unexpected errors → 500 with clean message
    """
    if agent is None:
        logger.warning("Chat request received before agent is ready")
        raise HTTPException(
            status_code=503,
            detail="Server is still starting up. Please try again."
        )

    logger.info(
        f"Chat request | "
        f"session={request.session_id} | "
        f"question='{request.question[:80]}'"
    )

    try:
        # ── Run sync agent in thread pool ─────────────────────────────
        # WHY asyncio.to_thread?
        # agent.run() blocks while waiting for Groq API response.
        # Running it directly in async would freeze the event loop —
        # no other request could be handled until this one finishes.
        # to_thread() offloads it to a thread pool, keeping the
        # event loop free for concurrent requests.
        result = await asyncio.to_thread(
            agent.run,
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

    except HTTPException:
        # re-raise intentional HTTP errors unchanged
        # without this, the broad except below would wrap them as 500
        raise

    except Exception as e:
        # log full traceback internally — never expose to user
        # traceback reveals internal code structure → security risk
        logger.error(
            f"Chat endpoint error | "
            f"session={request.session_id} | "
            f"error={e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Something went wrong. Please try again."
        )


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """
    Clear conversation history for a specific session.

    Called when user clicks 'New Chat' in the UI.
    WHY expose as API? Frontend needs to reset server-side memory.
    Without this, old history persists and breaks query rewriting
    for the next conversation.

    NOTE: When session storage moves to Redis, this calls
    redis_client.delete(f"session:{session_id}") instead.
    """
    if session_id in session_store:
        del session_store[session_id]
        logger.info(f"Session cleared | session_id={session_id}")
        return {"message": f"Session '{session_id}' cleared successfully"}

    # return 200 even when not found — end state is the same either way
    # returning 404 forces client to handle an error that isn't really one
    logger.warning(f"Clear requested for non-existent session: {session_id}")
    return {"message": f"Session '{session_id}' not found — nothing to clear"}