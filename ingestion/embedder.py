#ingestion/embedder.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE: Load the embedding model that converts text into vectors.
#          Every chunk of text and every user query goes through this
#          model before anything can be searched or stored.
#
# PIPELINE POSITION:
#   loader → chunker → [embedder.py] → vectorstore → retriever
#
# WHAT IS AN EMBEDDING (simple explanation):
#   Text  : "What is a lambda function?"
#   Vector: [0.23, -0.81, 0.45, 0.12, ...]  ← 384 numbers
#
#   Similar meaning → similar numbers → close together in vector space
#   This is how ChromaDB finds the most relevant chunks for a query.
#
# WHY all-MiniLM-L6-v2:
#   - Runs 100% locally — no API calls, no cost per embedding
#   - Downloads once on first use, cached in ~/.cache/huggingface/
#   - 384 dimensions — small and fast, good quality for general text
#   - Used by thousands of production RAG systems
#
# IMPORTANT — model name is now in settings.py:
#   Change EMBEDDING_MODEL in .env to swap models without touching code
# ─────────────────────────────────────────────────────────────────────
 
from langchain_huggingface import HuggingFaceEmbeddings  # wrapper that runs HuggingFace models locally
from config.settings import settings                      # centralised config → embedding_model from .env
from config.logger import get_logger                      # structured JSON logger → replaces print()
 
# ── Module-level logger ───────────────────────────────────────────────
# __name__ = "ingestion.embedder" → appears in every log line
# so you know exactly which file generated each log message
logger = get_logger(__name__)
 
 
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load and return the HuggingFace embedding model.
 
    Called in two places in the pipeline:
      1. build_index.py  → embeds document chunks and stores them
      2. vectorstore.py  → loads the same model to embed user queries
 
    WHY must both use the SAME model?
    Chunks were stored using model A's vector space.
    Queries must be searched in the same vector space.
    Using a different model = searching in a completely different
    coordinate system = garbage results.
 
    First call: downloads model from HuggingFace Hub (~90MB)
    Subsequent calls: loads from local cache instantly
 
    Returns:
        HuggingFaceEmbeddings : ready-to-use embedding model instance
 
    Raises:
        Exception : if model download fails or cache is corrupted
    """
 
    # ── Log which model we are loading ───────────────────────────────
    # WHY: If someone changes EMBEDDING_MODEL in .env, this log line
    # confirms which model actually got loaded at runtime.
    # Prevents silent mismatches between config and actual behaviour.
    logger.info(f"Loading embedding model: {settings.embedding_model}")
 
    try:
        embeddings = HuggingFaceEmbeddings(
            # ── model_name from settings ──────────────────────────────
            # AUDIT FIX: previously hardcoded as 'all-MiniLM-L6-v2'
            # Now reads from settings.embedding_model which reads from .env
            # This means: change .env → different model, zero code changes
            # Default value in settings.py: 'all-MiniLM-L6-v2'
            model_name=settings.embedding_model,
 
            # ── model_kwargs ──────────────────────────────────────────
            # cpu → forces CPU inference even if GPU is available
            # WHY: our Docker container has no GPU (CPU-only PyTorch)
            # Explicitly setting this prevents a confusing CUDA error
            # if the container ever runs on a GPU-enabled machine
            model_kwargs={"device": "cpu"},
 
            # ── encode_kwargs ─────────────────────────────────────────
            # normalize_embeddings=True → all vectors scaled to length 1
            # WHY: ChromaDB uses cosine similarity for search.
            # Normalised vectors make cosine similarity more accurate.
            # Always True when using cosine similarity as search type.
            encode_kwargs={"normalize_embeddings": True},
        )
 
        # ── Confirm model loaded successfully ─────────────────────────
        # This log line appearing means embeddings are ready to use.
        # If app crashes before this line, model loading was the issue.
        logger.info(
            f"Embedding model loaded successfully: {settings.embedding_model}"
        )
 
        return embeddings   # returned to vectorstore.py for building/loading the index
 
    except Exception as e:
        # ── Why log AND raise? ────────────────────────────────────────
        # logger.error → persists the error message to logs/app.log
        #                and to CloudWatch in production
        # raise         → tells the CALLER (build_index.py or
        #                 vectorstore.py) that this failed
        # WITHOUT raise: caller gets None back silently → crash later
        # WITH raise:    crash happens immediately with clear message
        logger.error(f"Failed to load embedding model '{settings.embedding_model}': {e}", exc_info=True)
        raise