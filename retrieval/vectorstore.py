# retrieval/vectorstore.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE: Manages the ChromaDB vector store — the search engine
#          at the heart of the RAG pipeline.
#
# THREE RESPONSIBILITIES:
#   build_vectorstore() → takes chunks, embeds them, saves to disk
#                         run ONCE via build_index.py before starting API
#   load_vectorstore()  → loads existing index from disk at API startup
#                         no re-embedding — just reads what's saved
#   get_retriever()     → wraps the vectorstore as a LangChain retriever
#                         accepts a query string, returns top_k chunks
#
# PIPELINE POSITION:
#   loader → chunker → embedder → [vectorstore.py] → retriever → LLM
#
# WHAT IS A VECTOR STORE?
#   A specialised database that stores vectors (lists of numbers).
#   ChromaDB stores each chunk as a 384-dimensional vector.
#   When you query it, it finds the chunks whose vectors are
#   most similar to the query vector — cosine similarity search.
#
# WHY PERSIST TO DISK?
#   Embedding 500+ chunks takes 30-60 seconds and costs computation.
#   Saving to disk means we only embed ONCE (during build_index.py).
#   On every API restart, we just load the saved vectors instantly.
# ─────────────────────────────────────────────────────────────────────

import os                              # os.path.exists → check if chroma_db/ folder and files exist
import time                            # time.time() → measure how long embedding/loading takes

from langchain_chroma import Chroma    # LangChain wrapper around ChromaDB
from ingestion.embedder import get_embeddings   # loads the HuggingFace embedding model
from config.settings import settings   # chroma_dir, top_k from .env
from config.logger import get_logger   # structured JSON logger

# ── Module-level logger ───────────────────────────────────────────────
# __name__ = "retrieval.vectorstore"
# Every log line from this file tagged with this name
logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# FUNCTION 1 — Build vector store (run once via build_index.py)
# ─────────────────────────────────────────────────────────────────────

def build_vectorstore(chunks: list):
    """
    Embed all document chunks and save the vector index to disk.

    This is the HEAVY operation — embedding 500+ chunks takes time.
    Run via build_index.py ONCE before starting the API.
    Re-run only when the source document changes.

    What happens internally:
        1. get_embeddings() loads the HuggingFace model
        2. Each chunk's text is converted to a 384-dim vector
        3. All vectors + original text saved to chroma_db/ folder
        4. Next time load_vectorstore() is called, it reads from disk

    Args:
        chunks (list): Document chunks from chunker.py
                       each has .page_content and .metadata

    Returns:
        Chroma: vectorstore instance (also saved to disk)

    Raises:
        ValueError  : if chunks list is empty
        Exception   : if embedding or saving fails
    """
    # ── Validate input ────────────────────────────────────────────────
    # WHY: If chunks is empty, ChromaDB creates an empty index silently.
    # Every query then returns 0 results and the LLM says "not found."
    # Catch this here with a clear message pointing to the real problem.
    if not chunks:
        error_msg = (
            "build_vectorstore() received empty chunks list. "
            "Run loader → chunker pipeline first."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(
        f"Building vector store | "
        f"chunks={len(chunks)} | "
        f"destination={settings.chroma_dir}"
    )

    start_time = time.time()   # track total embedding time

    try:
        # ── Load embedding model ──────────────────────────────────────
        # all-MiniLM-L6-v2 converts each chunk text to 384-dim vector
        # This model is loaded from local cache after first download
        embeddings = get_embeddings()

        # ── Embed all chunks and save to disk ─────────────────────────
        # Chroma.from_documents() does three things:
        #   1. Calls embeddings.embed_documents(texts) → list of vectors
        #   2. Stores vectors + original text + metadata in ChromaDB
        #   3. Persists everything to settings.chroma_dir on disk
        db = Chroma.from_documents(
            documents=chunks,                       # list of Document objects to embed and store
            embedding=embeddings,                   # embedding model to use for vectorisation
            persist_directory=settings.chroma_dir   # folder where index is saved (chroma_db/)
        )

        # ── Verify chunks were actually stored ────────────────────────
        # WHY: from_documents() can succeed but store 0 vectors if
        # all chunks had empty content. db._collection.count() checks
        # the actual number of stored vectors in ChromaDB.
        stored_count = db._collection.count()
        if stored_count == 0:
            logger.warning(
                "Vector store built but contains 0 vectors — "
                "chunks may have had empty page_content"
            )
        else:
            elapsed = time.time() - start_time
            logger.info(
                f"Vector store built successfully | "
                f"vectors_stored={stored_count} | "
                f"elapsed={elapsed:.1f}s | "
                f"saved_to={settings.chroma_dir}/"
            )

        return db   # returned to build_index.py (usually not used further)

    except ValueError:
        raise   # re-raise our own ValueError from the empty check above

    except Exception as e:
        logger.error(f"Failed to build vector store | error={e}", exc_info=True)
        raise


# ─────────────────────────────────────────────────────────────────────
# FUNCTION 2 — Load vector store (called on every API startup)
# ─────────────────────────────────────────────────────────────────────

def load_vectorstore():
    """
    Load the existing ChromaDB vector index from disk.

    Called once in lifespan() at API startup (api/main.py).
    Does NOT re-embed anything — just reads the saved vectors.
    Much faster than build_vectorstore() — typically < 2 seconds.

    Common failure: chroma_db/ folder doesn't exist because
    build_index.py was never run. This gives a clear error message.

    Returns:
        Chroma: loaded vectorstore instance ready for querying

    Raises:
        FileNotFoundError : chroma_db/ folder or DB file missing
        Exception         : ChromaDB internal load failure
    """
    # ── CHECK 1: folder must exist ────────────────────────────────────
    # Most common cause: developer forgot to run build_index.py first.
    # Give an actionable error message, not a cryptic ChromaDB error.
    if not os.path.exists(settings.chroma_dir):
        error_msg = (
            f"Vector store directory not found: {settings.chroma_dir}\n"
            f"ACTION: run 'python build_index.py' first to build the index."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # ── CHECK 2: chroma.sqlite3 must exist inside the folder ──────────
    # WHY this extra check? The chroma_db/ folder can exist on disk
    # (e.g. created by git or Docker volume mount) but be empty if
    # build_index.py failed halfway through. An empty folder causes a
    # confusing ChromaDB internal error. This gives a clear message.
    db_file = os.path.join(settings.chroma_dir, "chroma.sqlite3")
    if not os.path.exists(db_file):
        error_msg = (
            f"Vector store folder exists but database file missing: {db_file}\n"
            f"ACTION: delete {settings.chroma_dir}/ and run 'python build_index.py' again."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"Loading vector store from: {settings.chroma_dir}")
    start_time = time.time()

    try:
        # ── Load embedding model ──────────────────────────────────────
        # MUST use the SAME model that was used in build_vectorstore().
        # WHY? Vectors were created in all-MiniLM-L6-v2's vector space.
        # Querying with a different model = searching wrong coordinate system
        # = completely wrong results. settings.embedding_model ensures
        # the same model is used in both build and load.
        embeddings = get_embeddings()

        # ── Load from disk ────────────────────────────────────────────
        # Chroma() with persist_directory reads the existing index.
        # No embedding happens here — just loading saved vectors.
        db = Chroma(
            persist_directory=settings.chroma_dir,    # read from this folder
            embedding_function=embeddings              # needed to embed future queries
        )

        # ── Verify the index has content ──────────────────────────────
        # WHY: An index with 0 vectors is technically valid but useless.
        # Every query returns empty results. Warn loudly at startup
        # rather than confusing users with "not found in notes" for everything.
        count = db._collection.count()   # number of vectors currently stored

        if count == 0:
            logger.warning(
                "Vector store loaded but contains 0 vectors. "
                "Delete chroma_db/ and run build_index.py again."
            )
        else:
            elapsed = time.time() - start_time
            logger.info(
                f"Vector store loaded successfully | "
                f"vectors={count} | "
                f"elapsed={elapsed:.1f}s"
            )

        return db   # returned to get_retriever() and stored in global db in main.py

    except FileNotFoundError:
        raise   # re-raise our own FileNotFoundError from the checks above

    except Exception as e:
        logger.error(f"Failed to load vector store | error={e}", exc_info=True)
        raise


# ─────────────────────────────────────────────────────────────────────
# FUNCTION 3 — Create retriever from loaded vector store
# ─────────────────────────────────────────────────────────────────────

def get_retriever(db):
    """
    Wrap the ChromaDB vectorstore as a LangChain retriever.

    A retriever has a simple interface: give it a query string,
    it returns the top_k most relevant Document chunks.

    This retriever is used in TWO places:
        1. Inside the LCEL chain (llm_client.py) — fills {context}
        2. In RouterAgent.run() — gets sources[] for API response

    Args:
        db (Chroma): loaded vectorstore from load_vectorstore()

    Returns:
        VectorStoreRetriever: callable retriever
                              call with retriever.invoke("question")

    Raises:
        Exception: if retriever creation fails
    """
    logger.info(
        f"Creating retriever | "
        f"search_type=similarity | "
        f"top_k={settings.top_k}"
    )

    try:
        retriever = db.as_retriever(
            # ── search_type: similarity ───────────────────────────────
            # "similarity" → pure cosine similarity search
            #   Returns the top_k chunks closest to the query vector.
            #   Fast and simple. Good for most RAG use cases.
            #
            # Alternative: "mmr" (Maximum Marginal Relevance)
            #   Balances relevance WITH diversity in results.
            #   Prevents returning 4 nearly identical chunks.
            #   Use when your chunks have a lot of repeated content.
            #   Switch by changing search_type="mmr" here.
            search_type="similarity",

            search_kwargs={
                # ── k: number of chunks to retrieve ──────────────────
                # top_k=4 → retrieves 4 most similar chunks per query
                # These 4 chunks become the {context} sent to the LLM.
                # Tuning guide:
                #   k=2 → fast, less context, may miss relevant content
                #   k=4 → balanced (current setting)
                #   k=8 → more context but slower and noisier
                # Controlled from .env: TOP_K=4
                "k": settings.top_k
            }
        )

        logger.info(
            f"Retriever created successfully | "
            f"top_k={settings.top_k}"
        )
        return retriever   # returned to main.py lifespan → passed to RouterAgent

    except Exception as e:
        logger.error(f"Failed to create retriever | error={e}", exc_info=True)
        raise