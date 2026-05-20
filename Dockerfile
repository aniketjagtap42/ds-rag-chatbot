# ── Base Image ──────────────────────────────────────────────
# We use Python 3.10 slim (lightweight, no unnecessary OS packages)
# This matches your local Python version exactly
FROM python:3.10-slim

# ── Working Directory ────────────────────────────────────────
# All commands from here run inside /app folder inside the container
WORKDIR /app

# ── Install Dependencies ─────────────────────────────────────
# Copy requirements FIRST (before code) — Docker cache trick:
# If requirements.txt didn't change, Docker skips re-installing
# This makes rebuilds much faster during development
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy Project Code ────────────────────────────────────────
# Now copy everything else (code changes don't invalidate pip cache)
COPY . .

# ── Expose Port ──────────────────────────────────────────────
# Tell Docker this container listens on port 8000
EXPOSE 8000

# ── Start Command ────────────────────────────────────────────
# This runs when container starts
# --host 0.0.0.0 is critical — without it, API is unreachable from outside container
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]