# agents/router_agent.py
# ─────────────────────────────────────────────────────
# RouterAgent sits between the API and RAG chain.
# It adds three layers of intelligence:
#
# LAYER 1 — Scope Check:
#   Is this question related to DS/Python?
#   If NO → polite rejection, no LLM cost
#
# LAYER 2 — Query Rewriting:
#   Is this a follow-up question with pronouns?
#   "Give me an example of that" → meaningless to RAG
#   Rewrite → "Give me an example of a lambda function"
#
# LAYER 3 — Memory:
#   Keep last 6 messages per session
#   Gives context for follow-up questions
# ─────────────────────────────────────────────────────

import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import settings
from config.logger import get_logger

logger = get_logger(__name__)

# ── Session Store ─────────────────────────────────────
# In-memory dict: {session_id: [list of messages]}
# PRODUCTION NOTE: Replace with Redis for persistence
# Current limitation: history lost on server restart
session_store = {}

# Keep only last 6 messages (3 exchanges)
# Prevents context window overflow
MAX_HISTORY = 6


class RouterAgent:

    def __init__(self, rag_chain, retriever):
        """
        Initialise RouterAgent with RAG chain and retriever.

        Args:
            rag_chain: assembled LCEL chain from llm_client.py
            retriever: Chroma retriever from vectorstore.py
        """
        self.rag_chain = rag_chain
        self.retriever = retriever

        # ── Separate LLM for routing tasks ────────────
        # Used for scope check and query rewriting
        # Kept separate from RAG chain LLM so we can
        # use a faster/cheaper model for routing if needed
        self.llm = ChatGroq(
            model=settings.llm_model,
            temperature=0,
            groq_api_key=settings.groq_api_key
        )
        logger.info("RouterAgent initialised successfully")

    # ── Private: Session Management ───────────────────

    def _get_history(self, session_id: str) -> list:
        """Get conversation history for a session."""
        history = session_store.get(session_id, [])
        logger.info(
            f"Session {session_id} | "
            f"history length: {len(history)} messages"
        )
        return history

    def _save_to_history(
        self,
        session_id: str,
        user_msg: str,
        bot_msg: str
    ):
        """
        Save exchange to session history.
        Keeps only last MAX_HISTORY messages.
        """
        if session_id not in session_store:
            session_store[session_id] = []
            logger.info(f"New session created: {session_id}")

        session_store[session_id].extend([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": bot_msg}
        ])

        # ── Trim to MAX_HISTORY ────────────────────────
        # Prevents memory growing forever
        # Keeps most recent exchanges
        session_store[session_id] = (
            session_store[session_id][-MAX_HISTORY:]
        )
        logger.info(
            f"Session {session_id} | "
            f"saved exchange | "
            f"total messages: {len(session_store[session_id])}"
        )

    # ── Private: Scope Check ──────────────────────────

    def _is_in_scope(self, question: str) -> bool:
        """
        Check if question is related to DS/Python topics.

        Makes a separate LLM call with a simple YES/NO prompt.
        Prevents irrelevant questions reaching RAG chain.
        Saves LLM cost and improves user experience.

        Returns:
            True if in scope, False if out of scope
            Defaults to True on error (safe fallback)
        """
        logger.info(f"Scope check for: '{question[:80]}'")
        start_time = time.time()

        try:
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

            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({"question": question})
            in_scope = result.strip().upper() == "YES"

            elapsed = time.time() - start_time
            logger.info(
                f"Scope check result: {'IN SCOPE' if in_scope else 'OUT OF SCOPE'} "
                f"| took {elapsed:.2f}s"
            )
            return in_scope

        except Exception as e:
            # ── Safe fallback ──────────────────────────
            # If scope check fails, assume IN scope
            # Better to answer an off-topic question than
            # to reject a valid one due to a network error
            logger.error(
                f"Scope check failed: {e} | "
                f"defaulting to IN scope",
                exc_info=True
            )
            return True

    # ── Private: Query Rewriting ──────────────────────

    def _rewrite_with_context(
        self,
        question: str,
        history: list
    ) -> str:
        """
        Rewrite follow-up questions into standalone questions.

        PROBLEM THIS SOLVES:
        User: "What is a lambda function?"
        User: "Give me an example of that"

        ChromaDB cannot search for "that" — meaningless.
        Rewrite: "Give me an example of a lambda function"
        Now ChromaDB can find relevant chunks.

        Returns:
            Rewritten standalone question, or
            original question if no history or on error
        """
        # ── No history = no rewrite needed ────────────
        if not history:
            logger.info("No history — using original question")
            return question

        logger.info(f"Rewriting follow-up question: '{question[:80]}'")
        start_time = time.time()

        try:
            # ── Build history text ─────────────────────
            # Only use last 4 messages for context
            # More than that adds noise, not signal
            history_text = "\n".join([
                f"{m['role'].upper()}: {m['content']}"
                for m in history[-4:]
            ])

            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You rewrite follow-up questions into complete "
                    "standalone questions. "
                    "Return ONLY the rewritten question, nothing else."
                ),
                (
                    "human",
                    "Conversation history:\n{history}\n\n"
                    "Follow-up question: {question}\n\n"
                    "Rewrite as standalone question:"
                )
            ])

            chain = prompt | self.llm | StrOutputParser()
            rewritten = chain.invoke({
                "history": history_text,
                "question": question
            }).strip()

            elapsed = time.time() - start_time
            logger.info(
                f"Query rewritten in {elapsed:.2f}s | "
                f"'{question[:50]}' → '{rewritten[:50]}'"
            )
            return rewritten

        except Exception as e:
            # ── Safe fallback ──────────────────────────
            # If rewrite fails, use original question
            # RAG still works, just without context rewrite
            logger.error(
                f"Query rewrite failed: {e} | "
                f"using original question",
                exc_info=True
            )
            return question

    # ── Public: Main Entry Point ──────────────────────

    def run(self, question: str, session_id: str) -> dict:
        """
        Main method — processes question and returns answer.

        Flow:
        1. Get session history
        2. Scope check → reject if out of scope
        3. Rewrite query with context if follow-up
        4. Retrieve relevant chunks from ChromaDB
        5. Generate answer via RAG chain
        6. Save to session history
        7. Return answer + sources + metadata

        Args:
            question: user's question string
            session_id: unique identifier for conversation

        Returns:
            dict with keys:
                answer: str
                sources: list of dicts
                in_scope: bool
                rewritten_query: str
        """
        logger.info(
            f"Processing question | "
            f"session: {session_id} | "
            f"question: '{question[:80]}'"
        )
        start_time = time.time()

        # ── Step 1: Get session history ────────────────
        history = self._get_history(session_id)

        # ── Step 2: Scope check ────────────────────────
        if not self._is_in_scope(question):
            msg = (
                "I am designed to answer Data Science and Python "
                "questions from your study notes only. "
                "Please ask me something related to those topics!"
            )
            self._save_to_history(session_id, question, msg)
            logger.info(
                f"Out of scope question rejected | "
                f"session: {session_id}"
            )
            return {
                "answer": msg,
                "sources": [],
                "in_scope": False,
                "rewritten_query": question
            }

        # ── Step 3: Rewrite follow-up questions ───────
        standalone = self._rewrite_with_context(question, history)

        # ── Step 4: Retrieve source chunks ────────────
        # Run retriever separately to get sources for UI
        # (RAG chain runs retriever internally too)
        try:
            source_docs = self.retriever.invoke(standalone)
            sources = [
                {
                    "content": doc.page_content[:200],
                    "source": doc.metadata.get(
                        "source",
                        "DS_Complete_Notes.docx"
                    )
                }
                for doc in source_docs
            ]
            logger.info(f"Retrieved {len(sources)} source chunks")

        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            sources = []

        # ── Step 5: Generate answer ────────────────────
        try:
            answer = self.rag_chain.invoke(standalone)
            logger.info("Answer generated successfully")

        except Exception as e:
            logger.error(
                f"RAG chain failed: {e}",
                exc_info=True
            )
            answer = (
                "I encountered an error while generating your answer. "
                "Please try again in a moment."
            )

        # ── Step 6: Save to history ────────────────────
        self._save_to_history(session_id, question, answer)

        # ── Step 7: Return result ──────────────────────
        elapsed = time.time() - start_time
        logger.info(
            f"Request completed | "
            f"session: {session_id} | "
            f"total time: {elapsed:.2f}s"
        )

        return {
            "answer": answer,
            "sources": sources,
            "in_scope": True,
            "rewritten_query": standalone
        }