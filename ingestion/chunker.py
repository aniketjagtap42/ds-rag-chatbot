#ingestion/chunker.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE: Split large documents into smaller overlapping chunks
#          so they can be embedded and searched efficiently.
#
# PIPELINE POSITION:
#   loader → [chunker.py] → embedder → vectorstore → retriever
#
# WHY CHUNKING IS NECESSARY:
#   Problem 1 — LLM context window:
#     A document might be 200 pages. You cannot send 200 pages
#     to an LLM. It has a token limit (e.g. 8,000 tokens).
#
#   Problem 2 — Retrieval quality:
#     If you embed the whole document as one vector, the vector
#     represents EVERYTHING — too vague to match specific questions.
#     Small chunks = focused meaning = better search results.
#
#   Solution:
#     Split into ~500 character chunks → embed each chunk separately
#     → when user asks a question, find the 4 most relevant chunks
#     → send only those 4 chunks to the LLM as context
#
# WHY OVERLAP:
#   Without overlap, a concept split across chunk boundary is lost.
#   With 50-char overlap, both neighbouring chunks contain
#   the boundary content → no information lost at edges.
# ─────────────────────────────────────────────────────────────────────
 
from langchain_text_splitters import RecursiveCharacterTextSplitter  # splits text at natural boundaries
from config.settings import settings   # chunk_size and chunk_overlap values come from .env
from config.logger import get_logger   # structured JSON logger → replaces print()
 
# ── Module-level logger ───────────────────────────────────────────────
# __name__ = "ingestion.chunker" → every log line from this file
# is tagged with this name, making CloudWatch filtering easy
logger = get_logger(__name__)
 
 
def chunk_documents(documents: list) -> list:
    """
    Split a list of Document objects into smaller overlapping chunks.
 
    Uses RecursiveCharacterTextSplitter which splits at the most
    natural boundary it can find, trying each separator in order:
        1. "\n\n"  → paragraph break (preferred — keeps ideas together)
        2. "\n"    → line break
        3. "."     → sentence end
        4. " "     → word boundary
        5. ""      → character split (last resort — avoids this if possible)
 
    WHY RecursiveCharacterTextSplitter over simple split?
    Simple split at every 500 chars cuts mid-sentence randomly.
    Recursive split finds the nearest natural break → readable chunks
    that preserve semantic meaning.
 
    Args:
        documents (list): list of Document objects from loader.py
                          each has .page_content and .metadata
 
    Returns:
        list[Document]: smaller chunk Documents, each inheriting
                        the parent's .metadata (source filename etc.)
 
    Raises:
        ValueError  : if documents list is empty
        Exception   : if splitting fails unexpectedly
    """
 
    # ── STEP 1: Validate input before processing ──────────────────────
    # WHY: If loader returned empty list and we don't check here,
    # the splitter silently returns [] and vectorstore gets empty DB.
    # Fail loudly here so the error points to the right place.
    if not documents:
        error_msg = "chunk_documents() received empty documents list — nothing to chunk"
        logger.error(error_msg)
        raise ValueError(error_msg)
 
    # ── STEP 2: Log chunking parameters before starting ───────────────
    # WHY: These values directly affect retrieval quality.
    # Logging them means you can see exactly what settings were used
    # when you later debug "why did it return the wrong chunk?"
    logger.info(
        f"Starting chunking | "
        f"documents={len(documents)} | "
        f"chunk_size={settings.chunk_size} | "   # max chars per chunk
        f"chunk_overlap={settings.chunk_overlap}"  # chars shared between neighbouring chunks
    )
 
    try:
        # ── STEP 3: Create the text splitter ─────────────────────────
        splitter = RecursiveCharacterTextSplitter(
 
            # ── chunk_size ────────────────────────────────────────────
            # Maximum number of characters per chunk.
            # 500 chars ≈ 3–4 sentences ≈ one focused concept.
            # Configured in settings.py → controlled from .env
            # Tuning guide:
            #   Too small (< 200) → chunks lose context, poor answers
            #   Too large (> 1000) → too much noise, weaker retrieval
            #   Sweet spot for educational notes: 400–600
            chunk_size=settings.chunk_size,
 
            # ── chunk_overlap ─────────────────────────────────────────
            # Number of characters from end of chunk N
            # that are REPEATED at the start of chunk N+1.
            # Prevents losing a concept that spans a chunk boundary.
            # Rule of thumb: 10% of chunk_size
            # 500 chunk_size → 50 overlap is a good default
            chunk_overlap=settings.chunk_overlap,
 
            # ── separators ────────────────────────────────────────────
            # Ordered list of split points to try.
            # Splitter tries "\n\n" first. If that produces a piece
            # larger than chunk_size, it tries "\n", then ".", etc.
            # This hierarchy keeps paragraphs and sentences intact.
            separators=["\n\n", "\n", ".", " ", ""],
 
            # ── length_function ───────────────────────────────────────
            # How to measure chunk size.
            # len = character count (default, matches chunk_size unit)
            # Alternative: use tiktoken to count tokens instead,
            # which is more accurate for LLM context window planning.
            # For our use case, character count is sufficient.
            length_function=len,
        )
 
        # ── STEP 4: Perform the split ─────────────────────────────────
        # split_documents preserves .metadata from parent documents
        # so each chunk knows which source file it came from.
        # This is what populates the 'sources' field in the API response.
        chunks = splitter.split_documents(documents)
 
        # ── STEP 5: Validate output ───────────────────────────────────
        # WHY: A document with only images or no extractable text
        # would produce 0 chunks. Warn loudly so it's obvious.
        if not chunks:
            logger.warning(
                "Chunking produced 0 chunks — "
                "document may have no extractable text content"
            )
            return chunks   # return empty list — caller handles this
 
        # ── STEP 6: Calculate and log statistics ──────────────────────
        # WHY: Helps you verify chunking worked as expected.
        # avg_chunk_size should be close to settings.chunk_size.
        # If it's much smaller, the separators found many split points.
        # If it's much larger, text has no natural break points.
        total_chars = sum(len(c.page_content) for c in chunks)  # total chars across all chunks
        avg_chunk_size = total_chars // len(chunks)              # integer division → whole number average
 
        logger.info(f"Chunking complete | total_chunks={len(chunks)}")
        logger.info(f"Average chunk size: {avg_chunk_size} characters")
        logger.info(
            f"Smallest chunk: {min(len(c.page_content) for c in chunks)} chars | "
            f"Largest chunk: {max(len(c.page_content) for c in chunks)} chars"
        )
 
        return chunks   # hand off to embedder.py next in the pipeline
 
    except ValueError:
        # ── Re-raise ValueError unchanged ────────────────────────────
        # Already logged above — preserve the original message
        raise
 
    except Exception as e:
        # ── Catch unexpected splitter errors ─────────────────────────
        # exc_info=True adds the full Python traceback to the log.
        # Essential for debugging errors you haven't seen before.
        logger.error(f"Chunking failed unexpectedly: {e}", exc_info=True)
        raise   # never silently swallow — always re-raise in production