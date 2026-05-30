# ingestion/loader.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE: Load the source .docx document from disk into memory.
#          This is the FIRST step of the entire RAG pipeline.
#
# PIPELINE POSITION:
#   [loader.py] → chunker.py → embedder.py → vectorstore.py → retriever
#
# WHAT THIS FILE DOES:
#   1. Checks the file exists at the configured path
#   2. Reads the .docx file using Docx2txtLoader
#   3. Validates the content is not empty
#   4. Returns a list of LangChain Document objects
#
# WHAT IS A DOCUMENT OBJECT:
#   A LangChain Document has two fields:
#     .page_content → the actual text string
#     .metadata     → dict with source filename, page number, etc.
# ─────────────────────────────────────────────────────────────────────
 
import os                                            # os.path.exists → check if file is on disk before loading
from langchain_community.document_loaders import Docx2txtLoader  # reads .docx files and returns Document objects
from config.settings import settings                 # centralised config → data_path comes from .env
from config.logger import get_logger                 # structured JSON logger → replaces print()
 
# ── Module-level logger ───────────────────────────────────────────────
# __name__ = "ingestion.loader" → appears in every log line from this file
# helps you instantly know WHICH file produced a log line in CloudWatch
logger = get_logger(__name__)
 
 
def load_document() -> list:
    """
    Load the source .docx document from disk.
 
    Reads settings.data_path to find the file.
    Returns a list of Document objects where each object has:
        .page_content → the raw text extracted from the docx
        .metadata     → dict containing 'source' key with filename
 
    Why return a list even for one file?
    LangChain's splitters expect a list of Documents,
    so keeping this as a list makes the pipeline consistent.
 
    Raises:
        FileNotFoundError : file does not exist at settings.data_path
        ValueError        : file exists but contains no readable content
        Exception         : any other read/parse failure
 
    Returns:
        list[Document] : one or more Document objects with page_content
    """
 
    # ── STEP 1: Validate file exists before attempting to load ────────
    # WHY: Docx2txtLoader raises a confusing generic OSError if the file
    # is missing. This check catches it early and gives a clear message
    # telling you exactly what path to fix. Saves debugging time.
    if not os.path.exists(settings.data_path):
 
        # Build a helpful error message with the exact path that failed
        error_msg = (
            f"Document not found at: {settings.data_path}\n"
            f"Fix: place your .docx file at that path and restart."
        )
 
        logger.error(error_msg)          # log it so it appears in CloudWatch / log file
        raise FileNotFoundError(error_msg)  # stop execution — nothing can proceed without the doc
 
    # ── STEP 2: Log that we are starting to load ──────────────────────
    # WHY: In production, startup logs help you verify each component
    # loaded successfully. If the app hangs here, this log tells you where.
    logger.info(f"Loading document from: {settings.data_path}")
 
    try:
        # ── STEP 3: Create the loader and read the file ───────────────
        # Docx2txtLoader extracts all text from the .docx file.
        # It handles paragraphs, tables, headings automatically.
        # Returns a list of Document objects (usually one per file).
        loader = Docx2txtLoader(settings.data_path)  # initialise loader with file path
        documents = loader.load()                     # actually reads the file from disk
 
        # ── STEP 4: Validate content was extracted ────────────────────
        # WHY: A .docx file could exist on disk but be empty, password-
        # protected, or corrupted. loader.load() would return [] silently.
        # We check and raise early rather than letting chunker get [] later.
        if not documents:
            error_msg = (
                f"Document loaded but returned no content: {settings.data_path}\n"
                f"Check: file may be empty, corrupted, or password-protected."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
 
        # ── STEP 5: Log statistics about what was loaded ──────────────
        # WHY: These numbers are useful for debugging chunking issues.
        # If total_chars is very low, the doc may not have parsed correctly.
        # If sections count is unexpected, metadata may be wrong.
        total_chars = sum(len(d.page_content) for d in documents)  # total character count across all sections
        logger.info(f"Successfully loaded {len(documents)} section(s)")
        logger.info(f"Total characters extracted: {total_chars:,}")   # :, adds thousands separator e.g. 45,231
 
        return documents   # hand off to chunker.py next in the pipeline
 
    except (FileNotFoundError, ValueError):
        # ── Re-raise known errors unchanged ──────────────────────────
        # WHY: These were already logged above with good messages.
        # Re-raising without modification preserves the original message.
        # We don't want to wrap them in a generic "Exception" message.
        raise
 
    except Exception as e:
        # ── Catch unexpected errors ───────────────────────────────────
        # WHY: Could be permission denied, corrupted zip inside docx,
        # encoding error, etc. exc_info=True adds the full Python
        # traceback to the log — essential for debugging unknown errors.
        logger.error(f"Unexpected error loading document: {e}", exc_info=True)
        raise   # always re-raise — never silently swallow exceptions in production
 