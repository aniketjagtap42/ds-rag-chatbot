# generation/llm_client.py
# ─────────────────────────────────────────────────────
# Builds the RAG chain that connects retriever → LLM.
#
# PIPELINE POSITION:
# vectorstore → retriever → [llm_client] → answer
#
# WHAT THIS FILE DOES:
# 1. get_llm()         → creates Groq LLM instance
# 2. format_docs()     → joins retrieved chunks into text
# 3. build_rag_chain() → assembles the full RAG pipeline
# ─────────────────────────────────────────────────────

import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from generation.prompt_builder import build_prompt
from config.settings import settings
from config.logger import get_logger

logger = get_logger(__name__)


def get_llm() -> ChatGroq:
    """
    Create and return the Groq LLM instance.

    temperature=0 means deterministic responses.
    Same question → same answer every time.
    Better for factual Q&A over documents.
    temperature=1 would give creative/varied responses.

    Returns:
        ChatGroq LLM instance
    """
    logger.info(f"Initialising LLM: {settings.llm_model}")

    try:
        llm = ChatGroq(
            model=settings.llm_model,
            # ── temperature=0 ─────────────────────────
            # Deterministic output for factual answers
            # In production: never use high temperature
            # for document Q&A — too unpredictable
            temperature=0,
            groq_api_key=settings.groq_api_key
        )
        logger.info("LLM initialised successfully")
        return llm

    except Exception as e:
        logger.error(f"Failed to initialise LLM: {e}", exc_info=True)
        raise


def format_docs(docs) -> str:
    """
    Join retrieved Document chunks into a single string.

    This becomes the {context} in the prompt template.
    Double newline between chunks helps LLM distinguish
    where one chunk ends and another begins.

    Args:
        docs: list of Document objects from retriever

    Returns:
        Single string with all chunk content joined
    """
    formatted = "\n\n".join(doc.page_content for doc in docs)
    logger.info(f"Formatted {len(docs)} chunks into context")
    return formatted


@retry(
    # ── Retry up to 3 times total ─────────────────────
    stop=stop_after_attempt(3),

    # ── Exponential backoff ───────────────────────────
    # Wait 2s after attempt 1
    # Wait 4s after attempt 2
    # Max wait 10s
    # Gives Groq API time to recover from rate limits
    wait=wait_exponential(multiplier=1, min=2, max=10),

    # ── Only retry on these errors ────────────────────
    # Retrying on ALL errors could cause problems
    # e.g. retrying a bad API key 3 times is wasteful
    retry=retry_if_exception_type(Exception)
)
def _invoke_chain_with_retry(chain, query: str) -> str:
    """
    Internal function that invokes the chain with retry.

    Separated from build_rag_chain so retry decorator
    only wraps the actual LLM call, not the setup.

    Args:
        chain: assembled LangChain LCEL chain
        query: user question string

    Returns:
        LLM response as string
    """
    logger.info(f"Invoking RAG chain for query: '{query[:80]}...'")
    start_time = time.time()

    result = chain.invoke(query)

    elapsed = time.time() - start_time
    logger.info(f"RAG chain completed in {elapsed:.2f}s")

    return result


def build_rag_chain(retriever):
    """
    Assemble the complete RAG pipeline chain.

    Chain flow (read left to right):
    query
      → {context: retriever|format_docs, input: passthrough}
      → prompt template fills {context} and {input}
      → LLM generates response
      → StrOutputParser extracts text string
      → return answer

    This uses LangChain LCEL (Expression Language).
    The | pipe operator chains steps like Unix pipes.

    Args:
        retriever: Chroma retriever from vectorstore.py

    Returns:
        Callable chain — call with chain.invoke("question")
    """
    logger.info("Building RAG chain")

    try:
        llm = get_llm()
        prompt = build_prompt()

        # ── Assemble LCEL chain ────────────────────────
        chain = (
            {
                # ── context branch ────────────────────
                # retriever finds top_k chunks
                # format_docs joins them into one string
                # this fills {context} in prompt
                "context": retriever | format_docs,

                # ── input branch ──────────────────────
                # RunnablePassthrough passes query through
                # unchanged — fills {input} in prompt
                "input": RunnablePassthrough()
            }
            | prompt        # inject context + input into template
            | llm           # send filled prompt to Groq
            | StrOutputParser()  # extract text from response object
        )

        logger.info("RAG chain built successfully")
        return chain

    except Exception as e:
        logger.error(f"Failed to build RAG chain: {e}", exc_info=True)
        raise