# generation/prompt_builder.py
# ─────────────────────────────────────────────────────
# Builds the system prompt for the RAG chatbot.
#
# WHY THE PROMPT IS SO IMPORTANT:
# The prompt is what controls LLM behaviour.
# Same retrieved context + different prompt =
# completely different quality of answers.
#
# GOLDEN RULES FOR RAG PROMPTS:
# 1. Tell LLM to ONLY use provided context
# 2. Tell LLM what to say when answer NOT in context
# 3. NEVER allow LLM to make up information
# 4. Keep tone consistent (beginner friendly here)
# ─────────────────────────────────────────────────────

from langchain_core.prompts import ChatPromptTemplate
from config.logger import get_logger

logger = get_logger(__name__)

# ── System Prompt ─────────────────────────────────────
# This is sent as the "system" role to the LLM.
# Sets behaviour, rules, and constraints.
# {context} gets filled with retrieved chunks at runtime.
SYSTEM_PROMPT = """You are DS Notes Assistant, an expert tutor \
for Data Science and Python concepts.

You ONLY answer questions using the CONTEXT provided below.
The context is extracted from the student's own study notes.

Rules:
1. If the answer IS in the context — explain it clearly and simply.
2. If the answer is NOT in the context — say exactly:
   "I could not find this topic in your notes. \
   Please check your study material directly."
3. NEVER make up or assume information not present in the context.
4. If the context contains a code example — always include it.
5. Keep all explanations beginner friendly.
6. Use bullet points or numbered steps where it helps clarity.

CONTEXT:
{context}
"""


def build_prompt() -> ChatPromptTemplate:
    """
    Build and return the chat prompt template.

    Template has two slots filled at runtime:
        {context} → retrieved chunks from ChromaDB
        {input}   → user's question

    Returns:
        ChatPromptTemplate ready to use in RAG chain
    """
    logger.info("Building prompt template")

    try:
        prompt = ChatPromptTemplate.from_messages([
            # ── System message ────────────────────────
            # Sets LLM behaviour and rules
            # Filled with retrieved context at runtime
            ("system", SYSTEM_PROMPT),

            # ── Human message ─────────────────────────
            # The user's actual question
            # Passed through from router_agent.py
            ("human", "{input}")
        ])

        logger.info("Prompt template built successfully")
        return prompt

    except Exception as e:
        logger.error(f"Failed to build prompt template: {e}", exc_info=True)
        raise