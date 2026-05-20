# retrieval/vectorstore.py
# ─────────────────────────────────────────────────────
# Manages the ChromaDB vector store.
# Three responsibilities:
#   1. build_vectorstore()  → create index from chunks
#   2. load_vectorstore()   → load existing index
#   3. get_retriever()      → search the index
#
# PIPELINE POSITION:
# loader → chunker → embedder → [vectorstore] → retriever
# ─────────────────────────────────────────────────────

import os
import time
from langchain_chroma import Chroma
from ingestion.embedder import get_embeddings
from config.settings import settings
from config.logger import get_logger

logger = get_logger(__name__)


def build_vectorstore(chunks):
    """
    Build ChromaDB vector store from document chunks.

    Embeds every chunk and saves to settings.chroma_dir.
    Run this ONCE via build_index.py before starting API.
    Re-run only when source document changes.

    Args:
        chunks: list of Document chunks from chunker

    Returns:
        Chroma vectorstore instance
    """
    logger.info(f"Building vector store from {len(chunks)} chunks")
    logger.info(f"Saving to: {settings.chroma_dir}")

    # ── Track how long embedding takes ───────────────
    # Useful to know when optimising for production
    start_time = time.time()

    try:
        embeddings = get_embeddings()

        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            # ── persist_directory ─────────────────────
            # Saves vectors to disk so we don't re-embed
            # every time server restarts
            persist_directory=settings.chroma_dir
        )

        elapsed = time.time() - start_time
        logger.info(
            f"Vector store built successfully in {elapsed:.1f}s | "
            f"Saved to {settings.chroma_dir}/"
        )
        return db

    except Exception as e:
        logger.error(f"Failed to build vector store: {e}", exc_info=True)
        raise


def load_vectorstore():
    """
    Load existing ChromaDB vector store from disk.

    Called once on API startup via lifespan in main.py.
    Loads vectors from disk — no re-embedding needed.

    Raises:
        FileNotFoundError: if chroma_db/ folder missing
                          (means build_index.py not run yet)

    Returns:
        Chroma vectorstore instance
    """

    # ── Check chroma_db exists before loading ─────────
    # Most common mistake: running API before build_index.py
    # This gives a clear, actionable error instead of
    # a confusing ChromaDB internal error
    if not os.path.exists(settings.chroma_dir):
        error_msg = (
            f"Vector store not found at: {settings.chroma_dir}\n"
            f"Please run 'python build_index.py' first to build the index."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"Loading vector store from: {settings.chroma_dir}")

    start_time = time.time()

    try:
        embeddings = get_embeddings()

        db = Chroma(
            persist_directory=settings.chroma_dir,
            embedding_function=embeddings
        )

        elapsed = time.time() - start_time

        # ── Log collection size ────────────────────────
        # Tells you how many vectors are loaded
        # Useful to verify index is complete
        count = db._collection.count()
        logger.info(
            f"Vector store loaded in {elapsed:.1f}s | "
            f"{count} vectors in index"
        )

        return db

    except FileNotFoundError:
        raise

    except Exception as e:
        logger.error(f"Failed to load vector store: {e}", exc_info=True)
        raise


def get_retriever(db):
    """
    Create a retriever from the vector store.

    Retriever takes a query string, embeds it, and
    returns the top_k most similar chunks.

    Args:
        db: Chroma vectorstore instance

    Returns:
        Retriever instance ready to use in RAG chain
    """
    logger.info(
        f"Creating retriever | "
        f"search_type=similarity | "
        f"top_k={settings.top_k}"
    )

    try:
        retriever = db.as_retriever(
            # ── similarity search ─────────────────────
            # Cosine similarity between query vector
            # and stored chunk vectors
            search_type="similarity",
            search_kwargs={"k": settings.top_k}
        )
        logger.info("Retriever created successfully")
        return retriever

    except Exception as e:
        logger.error(f"Failed to create retriever: {e}", exc_info=True)
        raise