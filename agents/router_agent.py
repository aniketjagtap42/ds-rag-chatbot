# agents/router_agent.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE: RouterAgent sits between the FastAPI layer and the RAG chain.
#          It adds three layers of intelligence before any LLM call:
#
# LAYER 1 — Scope Check:
#   Is this question related to DS/Python topics?
#   If NO → polite rejection returned immediately
#   WHY: saves Groq API tokens, improves user experience,
#        prevents the LLM from answering random off-topic questions
#
# LAYER 2 — Query Rewriting:
#   Is this a follow-up question with pronouns or references?
#   "Give me an example of that" → meaningless to ChromaDB
#   Rewrite → "Give me an example of a lambda function"
#   WHY: ChromaDB does vector similarity search — it needs a
#        complete standalone question to find the right chunks
#
# LAYER 3 — Session Memory:
#   Keep last 6 messages (3 exchanges) per session
#   WHY: gives context so follow-up questions can be rewritten
#
# FLOW:
#   question + session_id
#       ↓
#   get history → scope check → rewrite query
#       ↓
#   retrieve chunks → generate answer → save history
#       ↓
#   return {answer, sources, in_scope, rewritten_query}
#
# PRODUCTION NOTE on session_store:
#   Currently uses in-memory dict — data lost on restart/redeploy.
#   For multi-pod Kubernetes: replace with Redis session store.
#   See session/store.py for the Redis implementation.
# ─────────────────────────────────────────────────────────────────────

import time                              # time.time() → measure how long each step takes
import asyncio                           # asyncio.to_thread → run sync code without blocking FastAPI event loop

from tenacity import (                   # tenacity → automatic retry library for flaky external API calls
    retry,                               # @retry decorator — wraps a function with retry logic
    stop_after_attempt,                  # stop retrying after N attempts
    wait_exponential,                    # wait 1s, then 2s, then 4s between retries (exponential backoff)
    retry_if_exception_type              # only retry on specific exception types
)

from langchain_groq import ChatGroq                          # Groq-hosted LLM client (fast Llama inference)
from langchain_core.prompts import ChatPromptTemplate        # builds structured prompt templates
from langchain_core.output_parsers import StrOutputParser    # parses LLM response to plain string
from config.settings import settings                         # centralised config — model name, API key from .env
from config.logger import get_logger                         # structured JSON logger — replaces print()

# ── Module-level logger ───────────────────────────────────────────────
# __name__ = "agents.router_agent"
# Every log line from this file is tagged with this name
# Makes filtering in CloudWatch Logs Insights easy:
#   fields @message | filter @logStream like "router_agent"
logger = get_logger(__name__)

# ── In-memory session store ───────────────────────────────────────────
# Structure: { "session_id": [ {role, content}, {role, content}, ... ] }
# LIMITATION: this dict lives in one Python process.
#   - Lost on every restart or redeploy
#   - Not shared across multiple pods in Kubernetes
# PRODUCTION UPGRADE: replace get/save calls with Redis
#   (see session/store.py)
session_store = {}

# ── Max messages to keep per session ─────────────────────────────────
# 6 messages = 3 user + 3 assistant exchanges
# WHY cap it? Prevents the history dict growing unbounded in memory.
# More than 6 messages adds noise rather than useful context
# for the query rewriter.
MAX_HISTORY = 6


class RouterAgent:
    """
    Intelligent router that sits in front of the RAG chain.
    Handles scope checking, query rewriting, and session memory.
    """

    def __init__(self, rag_chain, retriever):
        """
        Initialise RouterAgent with the RAG chain and retriever.

        Both are created once at app startup (in lifespan) and
        injected here. This avoids reloading models on every request.

        Args:
            rag_chain : LCEL chain from generation/llm_client.py
                        handles retrieval + prompt + LLM + parsing
            retriever : ChromaDB retriever from retrieval/vectorstore.py
                        used separately to get sources for the API response
        """
        self.rag_chain = rag_chain    # full RAG pipeline — used in step 5 (generate answer)
        self.retriever = retriever    # vector store retriever — used in step 4 (get sources)

        # ── Dedicated LLM for routing tasks only ─────────────────────
        # WHY separate from rag_chain's LLM?
        # Scope check and query rewriting are lightweight tasks.
        # In the future you could swap this for a smaller/cheaper model
        # (e.g. llama-3.1-8b) without affecting answer quality.
        # temperature=0 → deterministic output for YES/NO classifier
        self.llm = ChatGroq(
            model=settings.llm_model,           # e.g. "llama-3.3-70b-versatile" from .env
            temperature=0,                       # 0 = no randomness → consistent scope check results
            groq_api_key=settings.groq_api_key  # API key from .env — never hardcoded
        )

        logger.info(
            f"RouterAgent initialised | "
            f"model={settings.llm_model}"
        )

    # ─────────────────────────────────────────────────────────────────
    # PRIVATE METHODS — Session Management
    # ─────────────────────────────────────────────────────────────────

    def _get_history(self, session_id: str) -> list:
        """
        Retrieve conversation history for a given session.

        Returns empty list for new sessions — this is the correct
        default because a new session has no history to rewrite against.

        Args:
            session_id (str): unique identifier for the conversation

        Returns:
            list of message dicts: [{role: str, content: str}, ...]
        """
        history = session_store.get(session_id, [])   # .get() with default [] avoids KeyError on new sessions

        logger.info(
            f"session={session_id} | "
            f"history_length={len(history)} messages"
        )
        return history

    def _save_to_history(
        self,
        session_id: str,
        user_msg: str,
        bot_msg: str
    ) -> None:
        """
        Append a user + assistant exchange to session history.
        Trims to last MAX_HISTORY messages after saving.

        Args:
            session_id (str): conversation identifier
            user_msg   (str): what the user asked
            bot_msg    (str): what the assistant replied
        """
        # ── Create session entry if first message ─────────────────────
        # WHY check first? Avoids KeyError on dict access below.
        if session_id not in session_store:
            session_store[session_id] = []
            logger.info(f"New session created | session={session_id}")

        # ── Append both sides of the exchange ────────────────────────
        # Both messages added together so history always has
        # complete exchanges (never half a conversation)
        session_store[session_id].extend([
            {"role": "user",      "content": user_msg},   # user turn
            {"role": "assistant", "content": bot_msg}     # assistant turn
        ])

        # ── Trim history to MAX_HISTORY ───────────────────────────────
        # [-MAX_HISTORY:] keeps only the LAST 6 messages
        # Older messages are discarded — prevents unbounded memory growth
        session_store[session_id] = (
            session_store[session_id][-MAX_HISTORY:]
        )

        logger.info(
            f"History saved | session={session_id} | "
            f"total_messages={len(session_store[session_id])}"
        )

    # ─────────────────────────────────────────────────────────────────
    # PRIVATE METHODS — Scope Check
    # ─────────────────────────────────────────────────────────────────

    @retry(
        # ── Retry up to 3 times total (1 original + 2 retries) ───────
        # WHY: Groq API occasionally returns 429 (rate limit) or
        # 503 (service unavailable). Without retry, user gets an error.
        # With retry, the second attempt usually succeeds transparently.
        stop=stop_after_attempt(3),

        # ── Wait between retries: 1s → 2s → 4s ──────────────────────
        # WHY exponential? Gives the API time to recover.
        # Flat retry (0s) hammers the API and makes rate limits worse.
        wait=wait_exponential(multiplier=1, min=1, max=8),

        # ── Only retry on Exception, not on logic errors ──────────────
        retry=retry_if_exception_type(Exception)
    )
    def _is_in_scope(self, question: str) -> bool:
        """
        Classify whether a question is related to DS/Python topics.

        Makes a lightweight LLM call with a strict YES/NO prompt.
        If classification fails after all retries → defaults to True
        (fail open: better to answer off-topic than reject valid question)

        Args:
            question (str): the user's raw question

        Returns:
            bool: True if in scope, False if out of scope
        """
        logger.info(f"Scope check | question='{question[:80]}'")
        start_time = time.time()   # track how long the Groq API call takes

        try:
            # ── Build the classifier prompt ───────────────────────────
            # System message sets strict YES/NO format
            # WHY strict? Prevents LLM from saying "Yes, I think..." 
            # which would break the == "YES" check below
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are a query classifier. "
                    "Reply ONLY with YES or NO. No other words."
                ),
                (
                    "human",
                    "Is this question related to Python, "
                    "Data Science, Machine Learning, Deep Learning, "
                    "Statistics, Pandas, NumPy, RAG, LLMs, Git, "
                    "or FastAPI?\n\nQuestion: {question}"
                )
            ])

            # ── Run the chain: prompt → LLM → string output ──────────
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({"question": question})

            # ── Parse result — strict uppercase comparison ─────────────
            # .strip() removes any trailing whitespace or newline
            # .upper() normalises "yes", "Yes", "YES" → all match
            in_scope = result.strip().upper() == "YES"

            elapsed = time.time() - start_time
            logger.info(
                f"Scope check complete | "
                f"result={'IN_SCOPE' if in_scope else 'OUT_OF_SCOPE'} | "
                f"elapsed={elapsed:.2f}s"
            )
            return in_scope

        except Exception as e:
            # ── Fail open — default to in scope ──────────────────────
            # WHY fail open? If Groq is down, we should not block
            # the user from getting answers. RAG chain may still work.
            # Rejecting all questions because scope check is down
            # is a worse user experience than allowing a few off-topic ones.
            logger.error(
                f"Scope check failed after retries | error={e} | "
                f"defaulting to IN_SCOPE",
                exc_info=True  # includes full Python traceback in log
            )
            return True   # safe fallback

    # ─────────────────────────────────────────────────────────────────
    # PRIVATE METHODS — Query Rewriting
    # ─────────────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),          # retry up to 3 times on Groq API failures
        wait=wait_exponential(multiplier=1, min=1, max=8),  # exponential backoff between retries
        retry=retry_if_exception_type(Exception)
    )
    def _rewrite_with_context(
        self,
        question: str,
        history: list
    ) -> str:
        """
        Rewrite a follow-up question into a standalone question.

        PROBLEM THIS SOLVES:
            Turn 1 → User: "What is a lambda function?"
            Turn 2 → User: "Can you give me an example of that?"

            ChromaDB receives "Can you give me an example of that?"
            "that" has no meaning without context → wrong chunks retrieved

            After rewrite: "Can you give me an example of a lambda function?"
            ChromaDB now finds the right chunks → correct answer returned

        If no history exists, returns original question unchanged.
        If rewrite fails → returns original question as safe fallback.

        Args:
            question (str): user's current question (may contain pronouns)
            history  (list): previous messages in this session

        Returns:
            str: standalone question safe for vector search
        """
        # ── No history = first message = no rewrite needed ────────────
        # WHY check? Avoids an unnecessary LLM call on every first message.
        # Also: if there's no history, there are no pronouns to resolve.
        if not history:
            logger.info("No history present — skipping query rewrite")
            return question   # return unchanged — already standalone

        logger.info(f"Rewriting query | original='{question[:80]}'")
        start_time = time.time()

        try:
            # ── Build conversation context ────────────────────────────
            # Only use last 4 messages (2 exchanges) as context
            # WHY only 4? More history adds noise, not signal.
            # The immediately previous exchange is almost always
            # enough to resolve any pronoun in the follow-up.
            history_text = "\n".join([
                f"{m['role'].upper()}: {m['content']}"   # "USER: ..." or "ASSISTANT: ..."
                for m in history[-4:]                     # last 4 messages only
            ])

            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You rewrite follow-up questions into complete "
                    "standalone questions. "
                    "Return ONLY the rewritten question, nothing else. "
                    "Do not explain. Do not add any preamble."
                ),
                (
                    "human",
                    "Conversation history:\n{history}\n\n"
                    "Follow-up question: {question}\n\n"
                    "Rewrite as a complete standalone question:"
                )
            ])

            chain = prompt | self.llm | StrOutputParser()
            rewritten = chain.invoke({
                "history": history_text,
                "question": question
            }).strip()   # .strip() removes leading/trailing whitespace from LLM output

            elapsed = time.time() - start_time
            logger.info(
                f"Query rewritten | "
                f"original='{question[:50]}' → "
                f"rewritten='{rewritten[:50]}' | "
                f"elapsed={elapsed:.2f}s"
            )
            return rewritten

        except Exception as e:
            # ── Safe fallback — use original question ─────────────────
            # WHY? If rewrite fails, the original question still works
            # for vector search — just without pronoun resolution.
            # User still gets an answer, just possibly less accurate.
            logger.error(
                f"Query rewrite failed after retries | error={e} | "
                f"falling back to original question",
                exc_info=True
            )
            return question   # original question returned as fallback

    # ─────────────────────────────────────────────────────────────────
    # PUBLIC METHOD — Main Entry Point
    # ─────────────────────────────────────────────────────────────────

    def run(self, question: str, session_id: str) -> dict:
        """
        Main method — processes a question and returns a full response.

        Called from api/main.py via asyncio.to_thread() because this
        method is synchronous (blocking) and must not run directly in
        FastAPI's async event loop — it would block all other requests.

        Full flow:
            1. Get session history
            2. Scope check → reject if off-topic
            3. Rewrite query if follow-up question
            4. Retrieve relevant chunks from ChromaDB
            5. Generate answer via RAG chain (LLM)
            6. Save exchange to session history
            7. Return structured response dict

        Args:
            question   (str): the user's question (already validated by FastAPI)
            session_id (str): unique conversation identifier

        Returns:
            dict: {
                "answer"          : str   — the LLM-generated answer
                "sources"         : list  — retrieved chunk previews with source filename
                "in_scope"        : bool  — True if question was answered, False if rejected
                "rewritten_query" : str   — the standalone query used for vector search
            }
        """
        logger.info(
            f"Request received | "
            f"session={session_id} | "
            f"question='{question[:80]}'"
        )
        overall_start = time.time()   # track total end-to-end time for this request

        # ── STEP 1: Get conversation history ──────────────────────────
        # Needed by rewriter (step 3) to resolve pronouns.
        # Empty list returned for new sessions — safe default.
        history = self._get_history(session_id)

        # ── STEP 2: Scope check ───────────────────────────────────────
        # Reject off-topic questions before spending tokens on RAG.
        # _is_in_scope has @retry so transient Groq failures are handled.
        if not self._is_in_scope(question):
            out_of_scope_msg = (
                "I am designed to answer Data Science and Python "
                "questions from your study notes only. "
                "Please ask me something related to those topics!"
            )
            # Save rejection to history so follow-up questions have context
            self._save_to_history(session_id, question, out_of_scope_msg)

            logger.info(
                f"Question rejected — out of scope | "
                f"session={session_id}"
            )
            return {
                "answer":          out_of_scope_msg,
                "sources":         [],      # no sources because no retrieval was done
                "in_scope":        False,   # tells the frontend to display differently
                "rewritten_query": question # no rewrite happened — return original
            }

        # ── STEP 3: Rewrite query for vector search ───────────────────
        # Resolves pronouns/references using recent history.
        # Returns original question unchanged if no history exists.
        # _rewrite_with_context has @retry for Groq resilience.
        standalone = self._rewrite_with_context(question, history)

        # ── STEP 4: Retrieve source chunks from ChromaDB ──────────────
        # WHY retrieve separately here when rag_chain also retrieves?
        # rag_chain retrieves internally but doesn't expose the docs.
        # We need the raw docs to return sources[] in the API response
        # so the frontend can show "this answer came from page X".
        try:
            source_docs = self.retriever.invoke(standalone)   # vector similarity search

            # ── Build sources list for API response ───────────────────
            # Truncate content to 200 chars — enough for UI preview,
            # not so much that the response payload becomes huge.
            sources = [
                {
                    "content": doc.page_content[:200],          # first 200 chars of the chunk
                    "source":  doc.metadata.get(                # filename from metadata
                        "source",
                        "DS_Complete_Notes.docx"                # fallback if metadata missing
                    )
                }
                for doc in source_docs
            ]
            logger.info(f"Retrieved {len(sources)} source chunks")

        except Exception as e:
            # ── Retrieval failure — continue without sources ──────────
            # WHY continue? rag_chain does its own internal retrieval.
            # If THIS call fails, we lose source attribution but the
            # answer generation (step 5) may still succeed.
            logger.error(f"Source retrieval failed | error={e}", exc_info=True)
            sources = []   # empty sources — answer can still be generated

        # ── STEP 5: Generate answer via RAG chain ─────────────────────
        # rag_chain internally: retrieves → formats → prompts → LLM → parses
        # Uses the rewritten standalone question for better retrieval.
        try:
            answer = self.rag_chain.invoke(standalone)   # full RAG pipeline execution
            logger.info("Answer generated successfully")

        except Exception as e:
            # ── Generation failure — return user-friendly error ────────
            # WHY not re-raise? The API should return a 200 with an
            # error message rather than a 500 crash.
            # The error is fully logged for debugging.
            logger.error(f"RAG chain generation failed | error={e}", exc_info=True)
            answer = (
                "I encountered an error while generating your answer. "
                "Please try again in a moment."
            )

        # ── STEP 6: Save exchange to session history ──────────────────
        # Save original question (not rewritten) so history reads
        # naturally — user sees their own words in context, not the
        # rewritten version.
        self._save_to_history(session_id, question, answer)

        # ── STEP 7: Log total time and return response ─────────────────
        total_elapsed = time.time() - overall_start
        logger.info(
            f"Request complete | "
            f"session={session_id} | "
            f"total_elapsed={total_elapsed:.2f}s | "
            f"sources={len(sources)}"
        )

        return {
            "answer":          answer,      # LLM-generated grounded answer
            "sources":         sources,     # list of retrieved chunk previews
            "in_scope":        True,        # question was answered (not rejected)
            "rewritten_query": standalone   # the query used for vector search (useful for debugging)
        }