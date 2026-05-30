# generation/llm_client.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE: Builds and returns the complete RAG chain.
#          This is the core generation layer of the pipeline.
#
# PIPELINE POSITION:
#   loader → chunker → embedder → vectorstore → retriever → [llm_client]
#
# WHAT THIS FILE DOES:
#   get_llm()                  → creates and returns the Groq LLM instance
#   format_docs()              → joins retrieved chunks into one context string
#   _invoke_chain_with_retry() → calls chain.invoke() with automatic retry
#   build_rag_chain()          → assembles the full LCEL pipeline and returns it
#
# WHAT IS LCEL (LangChain Expression Language)?
#   A way to compose LangChain components using the | pipe operator.
#   Each | passes the output of one step as input to the next.
#   Similar to Unix pipes: cat file.txt | grep "python" | wc -l
#
# RAG CHAIN FLOW:
#   user query
#     → { context: retriever | format_docs,  input: passthrough }
#     → prompt template (fills {context} and {input} placeholders)
#     → Groq LLM (generates answer grounded in context)
#     → StrOutputParser (extracts plain string from LLM response object)
#     → answer string returned
# ─────────────────────────────────────────────────────────────────────

import time                                          # time.time() → measure chain execution time

from langchain_groq import ChatGroq                  # Groq-hosted LLM client — fast Llama inference
from langchain_core.output_parsers import StrOutputParser   # extracts plain text string from LLM response
from langchain_core.runnables import RunnablePassthrough    # passes input through unchanged to fill {input} slot

from tenacity import (
    retry,                         # @retry decorator — wraps function with automatic retry logic
    stop_after_attempt,            # stop retrying after N total attempts
    wait_exponential,              # wait 1s → 2s → 4s between retries (exponential backoff)
    retry_if_exception_type        # only retry on specific exception types
)

from generation.prompt_builder import build_prompt   # system prompt template with {context} and {input}
from config.settings import settings                 # LLM model name, API key, top_k from .env
from config.logger import get_logger                 # structured JSON logger

# ── Module-level logger ───────────────────────────────────────────────
# __name__ = "generation.llm_client"
# Every log line from this file is tagged with this name
logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# STEP 1 — LLM Initialisation
# ─────────────────────────────────────────────────────────────────────

def get_llm() -> ChatGroq:
    """
    Create and return the Groq LLM instance.

    Called once during build_rag_chain() at app startup.
    The returned LLM is reused for every /chat request —
    no need to reinitialise on each request.

    Returns:
        ChatGroq: configured LLM instance ready for inference

    Raises:
        Exception: if Groq client fails to initialise
                   (wrong API key, network issue, etc.)
    """
    logger.info(f"Initialising LLM | model={settings.llm_model}")

    try:
        llm = ChatGroq(
            # ── Model name from settings ──────────────────────────────
            # Read from settings.llm_model which reads from .env
            # Change LLM_MODEL in .env to swap models without code changes
            model=settings.llm_model,

            # ── temperature=0 ─────────────────────────────────────────
            # WHY 0? For factual document Q&A you want deterministic output.
            # Same question → same answer every time.
            # temperature=0   → deterministic, factual (correct for RAG)
            # temperature=0.7 → creative, varied (correct for creative writing)
            # temperature=1.0 → highly random (wrong for document Q&A)
            temperature=0,

            # ── API key from settings ─────────────────────────────────
            # Never hardcode — always read from .env via settings
            groq_api_key=settings.groq_api_key,

            # ── max_tokens ────────────────────────────────────────────
            # Maximum number of tokens in the LLM response.
            # WHY set explicitly? Without this, Groq uses its own default
            # which can vary between models and API versions.
            # 1024 tokens ≈ 750 words — enough for a detailed explanation.
            # Prevents unexpectedly long (and costly) responses.
            max_tokens=1024,
        )

        logger.info(f"LLM initialised successfully | model={settings.llm_model}")
        return llm

    except Exception as e:
        # Log full traceback — this is a startup failure, needs immediate attention
        logger.error(f"Failed to initialise LLM | error={e}", exc_info=True)
        raise   # re-raise → startup fails → clear error shown immediately


# ─────────────────────────────────────────────────────────────────────
# STEP 2 — Document Formatter
# ─────────────────────────────────────────────────────────────────────

def format_docs(docs) -> str:
    """
    Join retrieved Document chunks into a single context string.

    This function sits in the LCEL chain:
        retriever → format_docs → fills {context} in prompt

    WHY join with double newline?
    Single \n looks like a line break within one chunk.
    Double \n\n looks like a paragraph break between chunks.
    The LLM reads this as "here are several separate pieces of context"
    rather than one continuous block — better answer quality.

    Args:
        docs (list): Document objects returned by ChromaDB retriever
                     each has .page_content and .metadata

    Returns:
        str: all chunk texts joined into one context string
    """
    if not docs:
        # ── No chunks retrieved ───────────────────────────────────────
        # WHY warn not error? The chain can still run — the LLM will
        # see empty context and respond with "not found in notes."
        # That's the correct behaviour — better than crashing.
        logger.warning("format_docs received empty docs list — context will be empty")
        return ""   # empty string fills {context} → LLM says "not found"

    # ── Join all chunk texts with double newline separator ────────────
    formatted = "\n\n".join(doc.page_content for doc in docs)

    logger.info(
        f"Formatted {len(docs)} chunks | "
        f"total_context_chars={len(formatted)}"
    )
    return formatted


# ─────────────────────────────────────────────────────────────────────
# STEP 3 — Retry Wrapper for Chain Invocation
# ─────────────────────────────────────────────────────────────────────

@retry(
    # ── Retry up to 3 times total (1 original attempt + 2 retries) ───
    # WHY retry? Groq API occasionally returns:
    #   - 429 Too Many Requests (rate limit hit)
    #   - 503 Service Unavailable (temporary outage)
    #   - Connection timeout (network blip)
    # Without retry, these transient errors fail the user's request.
    # With retry, the second attempt usually succeeds transparently.
    stop=stop_after_attempt(3),

    # ── Exponential backoff between retries ───────────────────────────
    # Attempt 1 fails → wait 2 seconds → attempt 2
    # Attempt 2 fails → wait 4 seconds → attempt 3
    # WHY exponential? Gives the API time to recover.
    # Flat retry (0s) hammers the API and worsens rate limits.
    # multiplier=1, min=2, max=10 → waits: 2s, 4s (capped at 10s)
    wait=wait_exponential(multiplier=1, min=2, max=10),

    # ── Retry on all Exception types ─────────────────────────────────
    # Covers: ConnectionError, TimeoutError, Groq API errors, etc.
    retry=retry_if_exception_type(Exception)
)
def _invoke_chain_with_retry(chain, query: str) -> str:
    """
    Invoke the RAG chain with automatic retry on failure.

    WHY a separate function for this?
    The @retry decorator needs to wrap a discrete function.
    We don't want retry logic wrapping the chain BUILD step —
    only the chain INVOKE step (the actual Groq API call).
    Separating them makes the retry boundary explicit and clean.

    Args:
        chain : assembled LCEL RAG chain from build_rag_chain()
        query : standalone question string (already rewritten if needed)

    Returns:
        str: LLM-generated answer as plain text

    Raises:
        Exception: if all 3 attempts fail
    """
    # ── Log truncated query — avoid logging full text in production ───
    # WHY truncate? Log files shouldn't store full user queries for
    # privacy reasons. 80 chars gives enough context for debugging.
    display_query = query[:80] + "..." if len(query) > 80 else query
    logger.info(f"Invoking RAG chain | query='{display_query}'")

    start_time = time.time()   # track how long the Groq API call takes

    # ── The actual chain call ─────────────────────────────────────────
    # This is what @retry wraps — if this raises, tenacity retries it
    result = chain.invoke(query)

    elapsed = time.time() - start_time
    logger.info(
        f"RAG chain completed | "
        f"elapsed={elapsed:.2f}s | "
        f"response_chars={len(result)}"
    )

    return result   # plain string answer from StrOutputParser


# ─────────────────────────────────────────────────────────────────────
# STEP 4 — Build the Full RAG Chain
# ─────────────────────────────────────────────────────────────────────

def build_rag_chain(retriever):
    """
    Assemble and return the complete LCEL RAG pipeline.

    Called ONCE at app startup (in lifespan).
    The returned chain is stored in the RouterAgent and
    reused for every /chat request — never rebuilt per request.

    CHAIN ANATOMY:
    ┌─────────────────────────────────────────────────────┐
    │  Input: query string (e.g. "What is overfitting?")  │
    │                                                     │
    │  Step 1: Split into two branches simultaneously:    │
    │    context branch:                                  │
    │      retriever → finds top_k similar chunks         │
    │      format_docs → joins chunks into one string     │
    │      → fills {context} placeholder in prompt        │
    │    input branch:                                    │
    │      RunnablePassthrough → query passes unchanged   │
    │      → fills {input} placeholder in prompt          │
    │                                                     │
    │  Step 2: prompt → fills template with context+input │
    │  Step 3: llm → sends filled prompt to Groq API      │
    │  Step 4: StrOutputParser → extracts plain text      │
    │                                                     │
    │  Output: answer string                              │
    └─────────────────────────────────────────────────────┘

    WHY expose chain.invoke() through _invoke_chain_with_retry()?
    That function has the @retry decorator. When RouterAgent calls
    self.rag_chain.invoke(query) directly, there is no retry.
    The returned chain is wrapped so retries happen automatically.

    Args:
        retriever: ChromaDB retriever from vectorstore.py
                   configured with top_k from settings

    Returns:
        callable: invoke it with a query string to get an answer
                  e.g. chain.invoke("What is a lambda function?")

    Raises:
        Exception: if LLM or prompt initialisation fails
    """
    logger.info("Building RAG chain...")

    try:
        llm = get_llm()          # initialise Groq LLM
        prompt = build_prompt()  # load system prompt template

        # ── Assemble the LCEL pipeline ────────────────────────────────
        # The dict at the top creates two parallel branches.
        # Both branches run with the same input (the query string).
        # Their outputs fill the {context} and {input} slots in the prompt.
        raw_chain = (
            {
                # ── BRANCH 1: context ─────────────────────────────────
                # retriever.invoke(query) → list of Document chunks
                # format_docs(docs)       → single joined string
                # This string fills {context} in SYSTEM_PROMPT
                "context": retriever | format_docs,

                # ── BRANCH 2: input ───────────────────────────────────
                # RunnablePassthrough() passes the query through unchanged
                # This string fills {input} in the human message template
                "input": RunnablePassthrough()
            }
            | prompt             # ChatPromptTemplate fills {context} and {input}
            | llm                # ChatGroq sends the filled prompt to Groq API
            | StrOutputParser()  # extracts the text string from the ChatMessage response
        )

        # ── Wrap chain invocation with retry logic ─────────────────────
        # WHY wrap? raw_chain.invoke() has no retry on Groq failures.
        # This wrapper ensures retries happen transparently.
        # RouterAgent calls chain.invoke(query) which now has retry built in.
        class ChainWithRetry:
            """
            Thin wrapper that exposes .invoke() with automatic retry.
            Keeps the same interface as a raw LangChain chain.
            """
            def __init__(self, chain):
                self._chain = chain   # store the raw LCEL chain

            def invoke(self, query: str) -> str:
                # delegates to the retry-decorated function
                return _invoke_chain_with_retry(self._chain, query)

        chain = ChainWithRetry(raw_chain)

        logger.info("RAG chain built successfully")
        return chain   # returned to lifespan → stored in RouterAgent

    except Exception as e:
        logger.error(f"Failed to build RAG chain | error={e}", exc_info=True)
        raise   # re-raise → startup fails with clear error