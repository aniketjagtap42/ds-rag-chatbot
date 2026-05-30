# Dockerfile
# ─────────────────────────────────────────────────────────────────────
# PURPOSE: Build the ds_rag_chatbot container image.
#
# MULTI-STAGE BUILD — WHY?
#   Stage 1 (builder): installs all dependencies including heavy packages
#   Stage 2 (runtime): copies only what's needed to run the app
#   Result: final image is smaller and has no build tools installed
#
# SIZE REDUCTION: 8.82 GB → ~2 GB
#   The main cause of the large image is PyTorch.
#   Default PyTorch includes CUDA GPU libraries (~4 GB extra).
#   We install CPU-only PyTorch instead — we don't need GPU
#   because our embedding model runs on CPU in this setup.
#
# HOW TO BUILD:
#   docker-compose build
#   or: docker build -t ds-rag-api:latest .
#
# HOW TO RUN LOCALLY:
#   docker-compose up
# ─────────────────────────────────────────────────────────────────────


# ═════════════════════════════════════════════════════════════════════
# STAGE 1 — builder
# Install all Python dependencies in this stage.
# This stage is NOT part of the final image.
# ═════════════════════════════════════════════════════════════════════
FROM python:3.10-slim AS builder

# ── Working directory for build stage ────────────────────────────────
# All commands from here run inside /app inside the container
WORKDIR /app

# ── Copy only requirements first ─────────────────────────────────────
# WHY copy requirements before code?
# Docker caches each layer. If requirements.txt hasn't changed,
# Docker skips re-running pip install on the next build.
# This makes rebuilds after code changes much faster.
COPY requirements.txt .

# ── Install CPU-only PyTorch FIRST ───────────────────────────────────
# WHY install torch separately before requirements.txt?
# pip would install the default torch (with CUDA) if we don't specify.
# Default torch = ~4 GB. CPU-only torch = ~700 MB.
# --index-url points pip to the CPU-only PyTorch build.
# This MUST come before pip install -r requirements.txt
# otherwise requirements.txt might trigger the full CUDA torch.
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    --index-url https://download.pytorch.org/whl/cpu

# ── Install all remaining dependencies ───────────────────────────────
# --no-cache-dir → don't save pip download cache inside the image
#                  reduces image size by ~200MB
RUN pip install --no-cache-dir -r requirements.txt


# ═════════════════════════════════════════════════════════════════════
# STAGE 2 — runtime (final image)
# Only copy what's needed to run the app.
# Build tools, pip cache, and test files stay in stage 1.
# ═════════════════════════════════════════════════════════════════════
FROM python:3.10-slim

# ── Working directory ─────────────────────────────────────────────────
WORKDIR /app

# ── Copy installed packages from builder stage ───────────────────────
# Copies Python packages installed in stage 1 into the runtime image.
# This is how multi-stage builds work — packages built in stage 1,
# only the results copied to stage 2.
COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /usr/local/bin /usr/local/bin

# ── Copy application source code only ────────────────────────────────
# Copy each folder explicitly — do NOT use COPY . .
# WHY? COPY . . would copy venv/, chroma_db/, data/, .env
# which bloats the image and bakes secrets into it.
COPY agents/     agents/
COPY api/        api/
COPY config/     config/
COPY generation/ generation/
COPY ingestion/  ingestion/
COPY retrieval/  retrieval/

# ── Copy root-level files ─────────────────────────────────────────────
# build_index.py → needed to run ingestion inside container
# chat_ui.html   → served by /ui endpoint in FastAPI
COPY build_index.py .
COPY chat_ui.html .
# ── Copy vector index and data for AWS deployment ─────────────────────
# WHY? ECS Fargate has no volume mounts like local docker-compose.
# chroma_db/ contains the 108 vectors — needed for retrieval
# data/ contains DS_Complete_Notes.docx — needed if re-indexing
COPY chroma_db/ chroma_db/
COPY data/ data/
# ── Create directories and set permissions BEFORE switching user ──────
# WHY create /tmp/logs here?
# The non-root appuser cannot create directories in /app.
# /tmp is always writable by any user in Linux containers.
# We create the logs directory as root here so appuser can write to it.
# WHY /tmp/logs instead of /app/logs?
# In ECS Fargate there is no volume mount for logs/ like local docker-compose.
# /tmp is always available and writable — safe fallback for cloud deployments.
RUN mkdir -p /tmp/logs && \
    chmod 777 /tmp/logs

# ── Security: run as non-root user ───────────────────────────────────
# WHY? Running as root inside a container is a security risk.
# If an attacker escapes the container, they get root on the host.
# Best practice: always run production containers as a non-root user.
# docker-compose.yml overrides this with user: "0" for local dev.
# FIXED: moved directory creation BEFORE USER switch so root creates them
RUN adduser --disabled-password --gecos "" --uid 1001 appuser && \
    chown -R appuser:appuser /tmp/logs
USER appuser

# ── Tell Docker which port this container uses ────────────────────────
# Documentation only — does not actually open the port.
# The actual port mapping is in docker-compose.yml (ports: 8000:8000)
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────
# Docker and Kubernetes call GET /health to check if container is healthy.
# --start-period=60s → wait 60s before first check (model loading takes time)
# --interval=30s     → check every 30 seconds after that
# --timeout=10s      → fail if no response in 10 seconds
# --retries=3        → mark unhealthy only after 3 consecutive failures
# If unhealthy → Docker restarts the container automatically
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Start command ─────────────────────────────────────────────────────
# uvicorn starts the FastAPI app
# --host 0.0.0.0 → listen on ALL interfaces (required in Docker/ECS)
# --port 8000    → matches EXPOSE above and security group rule
# --workers 1    → single worker (avoid multiprocessing permission issues)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]