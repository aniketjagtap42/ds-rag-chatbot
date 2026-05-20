# ingestion/embedder.py
# ─────────────────────────────────────────────────────
# Creates the embedding model used to convert text
# chunks into vectors for similarity search.
#
# WHAT IS AN EMBEDDING:
# "What is Python?" → [0.23, -0.81, 0.45, ...]  (384 numbers)
# Similar sentences → similar vectors
# This is how ChromaDB finds relevant chunks
# ─────────────────────────────────────────────────────

from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import settings
from config.logger import get_logger

logger = get_logger(__name__)


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load and return the HuggingFace embedding model.

    Model runs locally — no API cost, no internet needed
    after first download. Downloads once, cached forever.

    Returns:
        HuggingFaceEmbeddings instance ready to use
    """
    logger.info(f"Loading embedding model: {settings.embedding_model}")

    try:
        embeddings = HuggingFaceEmbeddings(
            # ── Model name from settings ──────────────
            # FIXED: was hardcoded 'all-MiniLM-L6-v2'
            # Now reads from settings → controlled from .env
            model_name=settings.embedding_model
        )
        logger.info("Embedding model loaded successfully")
        return embeddings

    except Exception as e:
        # ── Why log AND raise? ────────────────────────
        # logger.error → saves error to logs/app.log
        # raise        → tells the caller something went wrong
        # Without raise, app silently continues with no embeddings
        logger.error(f"Failed to load embedding model: {e}")
        raise