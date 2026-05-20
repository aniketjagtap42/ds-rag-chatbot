# ingestion/chunker.py
# ─────────────────────────────────────────────────────
# Splits large documents into smaller chunks for
# embedding and retrieval.
#
# PIPELINE POSITION:
# loader → [chunker] → embedder → vectorstore → retriever
#
# WHY CHUNKING IS NEEDED:
# LLMs have a context window limit.
# We cannot send the entire document to the LLM.
# Instead we find the most relevant small pieces
# and send only those as context.
# ─────────────────────────────────────────────────────

from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import settings
from config.logger import get_logger

logger = get_logger(__name__)


def chunk_documents(documents):
    """
    Split documents into smaller overlapping chunks.

    Uses RecursiveCharacterTextSplitter which tries to
    split on natural boundaries in this priority order:
        1. Double newline (paragraph break)
        2. Single newline
        3. Period (sentence end)
        4. Space (word boundary)
        5. Character (last resort)

    This keeps sentences and paragraphs together
    instead of cutting mid-sentence.

    Args:
        documents: list of Document objects from loader

    Returns:
        list of smaller Document chunk objects
    """
    logger.info(
        f"Chunking {len(documents)} document(s) | "
        f"chunk_size={settings.chunk_size} | "
        f"chunk_overlap={settings.chunk_overlap}"
    )

    try:
        splitter = RecursiveCharacterTextSplitter(
            # ── chunk_size ────────────────────────────
            # Max characters per chunk
            # 500 chars ≈ 3-4 sentences ≈ one concept
            # Too small → loses context
            # Too large → retrieves irrelevant content
            chunk_size=settings.chunk_size,

            # ── chunk_overlap ─────────────────────────
            # Last 50 chars of chunk N repeated at
            # start of chunk N+1
            # Prevents losing concepts at boundaries
            chunk_overlap=settings.chunk_overlap,

            # ── separators ────────────────────────────
            # Try these in order to find split points
            separators=["\n\n", "\n", ".", " ", ""]
        )

        chunks = splitter.split_documents(documents)

        # ── Validate we got chunks ────────────────────
        if not chunks:
            logger.warning("Chunking returned 0 chunks — check document content")
            return chunks

        # ── Log useful stats ──────────────────────────
        avg_chunk_size = sum(
            len(c.page_content) for c in chunks
        ) // len(chunks)

        logger.info(f"Total chunks created: {len(chunks)}")
        logger.info(f"Average chunk size: {avg_chunk_size} characters")

        return chunks

    except Exception as e:
        logger.error(f"Chunking failed: {e}", exc_info=True)
        raise