# build_index.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE: One-time script to build the ChromaDB vector index.
#          Run this BEFORE starting the API for the first time.
#          Re-run whenever your source document changes.
#
# WHAT IT DOES:
#   1. Loads the .docx file from data/
#   2. Splits it into ~500 character chunks
#   3. Embeds each chunk using all-MiniLM-L6-v2
#   4. Saves all vectors to chroma_db/ folder on disk
#
# HOW TO RUN:
#   Locally  : python build_index.py
#   In Docker: docker-compose run --rm ds-rag-api python build_index.py
#
# WHEN TO RE-RUN:
#   - When you update DS_Complete_Notes.docx
#   - When you change CHUNK_SIZE or CHUNK_OVERLAP in .env
#   - When you change EMBEDDING_MODEL in .env
#   NOTE: changing embedding model requires deleting chroma_db/ first
# ─────────────────────────────────────────────────────────────────────

import sys                                        # sys.exit() → exit with error code on failure
import time                                       # time.time() → measure total build time

from ingestion.loader import load_document        # step 1: read .docx from disk
from ingestion.chunker import chunk_documents     # step 2: split into overlapping chunks
from retrieval.vectorstore import build_vectorstore  # step 3: embed + save to chroma_db/
from config.logger import get_logger              # structured logger → replaces print()

# ── Logger for this script ────────────────────────────────────────────
# __name__ = "__main__" when run as a script
# Logs appear in terminal AND logs/app.log
logger = get_logger(__name__)


def main():
    """
    Run the full ingestion pipeline to build the vector index.

    Exit codes:
        0 → success
        1 → failure (logged with full error details)
    """
    logger.info("=" * 60)
    logger.info("DS Notes Vector Index Builder — STARTING")
    logger.info("=" * 60)

    overall_start = time.time()   # track total time for all 3 steps

    try:
        # ── STEP 1: Load the source document ─────────────────────────
        # Reads settings.data_path (./data/DS_Complete_Notes.docx)
        # Raises FileNotFoundError if file doesn't exist
        logger.info("Step 1/3: Loading source document...")
        docs = load_document()
        logger.info(f"Step 1/3 complete | sections_loaded={len(docs)}")

        # ── STEP 2: Split into chunks ─────────────────────────────────
        # Uses chunk_size and chunk_overlap from settings
        # Raises ValueError if docs is empty
        logger.info("Step 2/3: Chunking document...")
        chunks = chunk_documents(docs)
        logger.info(f"Step 2/3 complete | chunks_created={len(chunks)}")

        # ── STEP 3: Embed and save to ChromaDB ───────────────────────
        # Most time-consuming step — embeds all chunks with MiniLM
        # Saves to chroma_db/ folder — persists between restarts
        logger.info("Step 3/3: Building vector store (this takes ~30-60s)...")
        build_vectorstore(chunks)
        logger.info("Step 3/3 complete | vector index saved to chroma_db/")

        # ── Done ──────────────────────────────────────────────────────
        total_elapsed = time.time() - overall_start
        logger.info("=" * 60)
        logger.info(f"Vector index built successfully | total_time={total_elapsed:.1f}s")
        logger.info("You can now start the API with: docker-compose up")
        logger.info("=" * 60)

    except FileNotFoundError as e:
        # ── Source document not found ─────────────────────────────────
        # Clear message: put your .docx in data/ folder first
        logger.error(f"Document not found: {e}")
        logger.error("Fix: place DS_Complete_Notes.docx in the data/ folder")
        sys.exit(1)   # exit code 1 = failure (CI/CD pipelines check this)

    except ValueError as e:
        # ── Empty document or chunks ──────────────────────────────────
        logger.error(f"Document content error: {e}")
        sys.exit(1)

    except Exception as e:
        # ── Unexpected failure ────────────────────────────────────────
        # exc_info=True → full traceback in log file for debugging
        logger.error(f"Index build failed unexpectedly: {e}", exc_info=True)
        sys.exit(1)


# ── Entry point ───────────────────────────────────────────────────────
# WHY if __name__ == "__main__"?
# Prevents this code running when the file is imported by tests.
# Only runs when executed directly: python build_index.py
if __name__ == "__main__":
    main()