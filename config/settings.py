# config/settings.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE: Single source of truth for all application configuration.
#          Every configurable value lives here — never hardcoded.
#
# HOW IT WORKS:
#   Pydantic's BaseSettings reads values in this priority order:
#     1. Environment variables (e.g. export GROQ_API_KEY=xxx)
#     2. .env file (GROQ_API_KEY=xxx in the .env file)
#     3. Default values defined here in the class
#
# WHY CENTRALISED CONFIG?
#   - Change model name → edit .env, not 5 different files
#   - No secrets hardcoded in source code
#   - Different values for dev/staging/prod via different .env files
#   - Type validation — if chunk_size="abc" in .env, app crashes
#     immediately at startup with a clear error, not silently later
#
# HOW TO USE IN ANY FILE:
#   from config.settings import settings
#   print(settings.groq_api_key)   ← reads from .env automatically
#
# YOUR .env FILE SHOULD CONTAIN:
#   GROQ_API_KEY=gsk_your_key_here
#   EMBEDDING_MODEL=all-MiniLM-L6-v2
#   LLM_MODEL=llama-3.3-70b-versatile
#   CHUNK_SIZE=500
#   CHUNK_OVERLAP=50
#   TOP_K=4
#   CHROMA_DIR=./chroma_db
#   DATA_PATH=./data/DS_Complete_Notes.docx
#   ALLOWED_ORIGINS=http://localhost:3000
#   LOG_LEVEL=INFO
# ─────────────────────────────────────────────────────────────────────

from pydantic_settings import BaseSettings, SettingsConfigDict  # FIXED: added SettingsConfigDict import
from pydantic import Field                                       # Field → validation rules and descriptions


class Settings(BaseSettings):
    """
    All application settings loaded from environment / .env file.

    Pydantic validates every field on startup.
    If a required field (no default) is missing → crash immediately.
    If a field has wrong type → crash immediately with clear message.
    WHY crash immediately? Better than silently using wrong values
    for hours in production before someone notices.
    """

    # ── LLM / API Keys ────────────────────────────────────────────────
    # Required field — no default value, no fallback.
    # If GROQ_API_KEY is missing from .env → app crashes at startup
    # with: "groq_api_key field required"
    # This is intentional — app cannot work without this key.
    groq_api_key: str = Field(
        ...,                               # ... = required, no default
        description="Groq API key for LLM inference"
        # Set in .env: GROQ_API_KEY=gsk_xxx
        # NEVER hardcode this value in source code
        # NEVER commit .env to GitHub
    )

    # ── AUDIT FIX: removed openai_api_key ────────────────────────────
    # WHY removed? This project uses HuggingFace local embeddings,
    # not OpenAI embeddings. The openai_api_key was set to 'dummy'
    # which is confusing and misleading. Removing it entirely
    # prevents accidental use and cleans up the config.

    # ── Embedding Model ───────────────────────────────────────────────
    # AUDIT FIX: was 'text-embedding-ada-002' (an OpenAI model)
    # but embedder.py was using 'all-MiniLM-L6-v2' (HuggingFace).
    # Now aligned — settings.embedding_model matches actual usage.
    # WHY configurable? To swap embedding models without code changes.
    # IMPORTANT: if you change this, delete chroma_db/ and re-run
    # build_index.py — old vectors were created with the old model
    # and are incompatible with a different model's vector space.
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description=(
            "HuggingFace embedding model name. "
            "WARNING: changing this requires rebuilding the vector index."
        )
    )

    # ── LLM Model ─────────────────────────────────────────────────────
    # Groq model identifier — must match a model available on Groq API.
    # llama-3.3-70b-versatile is fast, high quality, and free tier.
    # Other options: llama-3.1-8b-instant (faster, cheaper, less accurate)
    llm_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq LLM model identifier"
    )

    # ── Chunking Parameters ───────────────────────────────────────────
    # These directly affect retrieval quality.
    # Tune by running experiments and measuring answer accuracy.
    # See chunker.py for explanation of what these values mean.
    chunk_size: int = Field(
        default=500,
        ge=100,     # ge = greater than or equal — minimum 100 chars
        le=2000,    # le = less than or equal — maximum 2000 chars
        description="Maximum characters per document chunk"
    )

    chunk_overlap: int = Field(
        default=50,
        ge=0,       # overlap can be 0 (no overlap)
        le=500,     # but not larger than chunk_size
        description="Character overlap between consecutive chunks"
    )

    # ── Retrieval Parameters ──────────────────────────────────────────
    # top_k = how many chunks to retrieve per query
    # Higher → more context for LLM but slower and more noise
    # Lower → faster but may miss relevant content
    # 4 is the sweet spot for most RAG use cases
    top_k: int = Field(
        default=4,
        ge=1,       # must retrieve at least 1 chunk
        le=20,      # retrieving more than 20 is almost never useful
        description="Number of chunks to retrieve per query"
    )

    # ── File Paths ────────────────────────────────────────────────────
    # These are relative paths from project root.
    # In Docker, these resolve relative to WORKDIR (/app).
    # chroma_db/ is mounted as a volume so it persists across restarts.
    chroma_dir: str = Field(
        default="./chroma_db",
        description="Directory where ChromaDB vector index is stored"
    )

    data_path: str = Field(
        default="./data/DS_Complete_Notes.docx",
        description="Path to the source .docx document for ingestion"
    )

    # ── CORS Origins ──────────────────────────────────────────────────
    # AUDIT FIX: was "*" — allows ANY website to call the API.
    # Now defaults to localhost:3000 for local development.
    # For production set in .env:
    #   ALLOWED_ORIGINS=https://yourfrontend.com
    # Multiple origins (comma-separated):
    #   ALLOWED_ORIGINS=http://localhost:3000,https://yourfrontend.com
    # WHY not "*"? Any website could call your API using a user's
    # browser session — a Cross-Site Request Forgery (CSRF) risk.
    allowed_origins: str = Field(
        default="http://localhost:3000",
        description=(
            "Comma-separated list of allowed CORS origins. "
            "Never use * in production."
        )
    )

    # ── Logging ───────────────────────────────────────────────────────
    # Controls minimum log level for the entire application.
    # INFO  → normal production setting (INFO + WARNING + ERROR + CRITICAL)
    # DEBUG → development setting (everything including DEBUG messages)
    # Set in .env: LOG_LEVEL=DEBUG for verbose output during development
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL"
    )

    # ── Pydantic V2 config ────────────────────────────────────────────
    # FIXED: replaced deprecated class Config with model_config
    # WHY: Pydantic V2 deprecated the inner class Config pattern.
    # SettingsConfigDict is the modern replacement — same functionality,
    # no deprecation warning.
    # IMPORTANT: model_config must be INSIDE the class, not outside.
    # The previous version had it outside the class — that's why
    # SettingsConfigDict was not recognized.
    model_config = SettingsConfigDict(
        env_file=".env",             # read from .env file in project root
        env_file_encoding="utf-8",   # handle special characters in .env values
        case_sensitive=False         # GROQ_API_KEY and groq_api_key both work
    )


# ── Single shared instance ────────────────────────────────────────────
# WHY module-level singleton?
# Python caches module imports — Settings() is called exactly once.
# Every file that does `from config.settings import settings`
# gets the SAME instance, not a new one.
# This means .env is read exactly once at startup — efficient.
settings = Settings()