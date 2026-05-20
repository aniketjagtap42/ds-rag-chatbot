# ingestion/loader.py
# ─────────────────────────────────────────────────────
# Loads the source document from disk into memory.
# First step of the RAG pipeline.
#
# PIPELINE POSITION:
# [loader] → chunker → embedder → vectorstore → retriever
# ─────────────────────────────────────────────────────

import os
from langchain_community.document_loaders import Docx2txtLoader
from config.settings import settings
from config.logger import get_logger

logger = get_logger(__name__)


def load_document():
    """
    Load the source .docx document from disk.

    Reads settings.data_path to find the file.
    Returns list of Document objects with:
        .page_content → the actual text
        .metadata     → source filename, etc.

    Raises:
        FileNotFoundError: if document doesn't exist at path
        Exception: if file exists but cannot be read

    Returns:
        list of Document objects
    """

    # ── Step 1: Check file exists before trying to load ──
    # Without this check, Docx2txtLoader gives a confusing
    # error. This gives a clear, actionable message instead.
    if not os.path.exists(settings.data_path):
        error_msg = (
            f"Document not found at: {settings.data_path}\n"
            f"Please place your .docx file at that path and restart."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"Loading document from: {settings.data_path}")

    try:
        # ── Step 2: Load the document ─────────────────
        loader = Docx2txtLoader(settings.data_path)
        documents = loader.load()

        # ── Step 3: Validate we got content ───────────
        # File could exist but be empty or corrupted
        if not documents:
            error_msg = f"Document loaded but returned no content: {settings.data_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # ── Step 4: Log useful stats ───────────────────
        # These logs help debug chunking issues later
        total_chars = sum(len(d.page_content) for d in documents)
        logger.info(f"Loaded {len(documents)} section(s)")
        logger.info(f"Total characters: {total_chars:,}")

        return documents

    except FileNotFoundError:
        # ── Re-raise FileNotFoundError as-is ──────────
        # Already logged above, just re-raise
        raise

    except Exception as e:
        # ── Catch everything else ─────────────────────
        # Could be corrupted file, permission error, etc.
        logger.error(f"Failed to load document: {e}", exc_info=True)
        raise