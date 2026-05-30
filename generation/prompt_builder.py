# generation/prompt_builder.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE: Defines the system prompt and builds the ChatPromptTemplate
#          used by the RAG chain to generate grounded answers.
#
# WHY THE PROMPT IS THE MOST IMPORTANT PART OF RAG:
#   The same retrieved context + different prompt =
#   completely different answer quality.
#
#   A bad prompt → LLM ignores context and hallucinates
#   A good prompt → LLM ONLY uses context, admits when unsure
#
# GOLDEN RULES FOR RAG PROMPTS:
#   1. Explicitly tell LLM to ONLY use the provided context
#   2. Tell LLM exactly what to say when answer is NOT in context
#   3. NEVER allow open-ended generation — every claim needs context
#   4. Set tone and format expectations explicitly
#
# PROMPT VERSIONING:
#   PROMPT_VERSION is tracked here so when you run MLflow experiments
#   (trying different prompts), you can log which version was used
#   and correlate it with retrieval quality metrics.
# ─────────────────────────────────────────────────────────────────────

from langchain_core.prompts import ChatPromptTemplate   # builds structured prompt with system + human messages
from config.logger import get_logger                    # structured JSON logger

# ── Module-level logger ───────────────────────────────────────────────
logger = get_logger(__name__)

# ── Prompt version tracking ───────────────────────────────────────────
# WHY version the prompt? When you tune the prompt and measure whether
# retrieval quality improved, you need to know WHICH prompt produced
# which result. Log this in MLflow alongside your eval metrics.
# Increment whenever you make a meaningful change to SYSTEM_PROMPT.
PROMPT_VERSION = "1.1.0"

# ── System Prompt ─────────────────────────────────────────────────────
# Sent as the "system" role — sets LLM behaviour and constraints.
# {context} is filled at runtime with the retrieved document chunks.
# {input}   is filled at runtime with the user's question.
#
# WHY use a multiline string here (not inline in the function)?
# Keeping it as a module-level constant means:
#   - Easy to read and edit without touching function logic
#   - Can be imported and tested independently
#   - Can be versioned and tracked in MLflow experiments
SYSTEM_PROMPT = """You are DS Notes Assistant, an expert tutor \
for Data Science and Python concepts.

Your ONE job is to help students understand topics from their OWN study notes.

STRICT RULES — follow every one of these:
1. ONLY use information from the CONTEXT section below.
   Do not use your general training knowledge.
2. If the answer IS in the context:
   - Explain it clearly and simply in beginner-friendly language
   - Always include code examples if the context contains them
   - Use numbered steps or bullet points where it helps clarity
   - Format any code blocks using triple backticks with language label:
```python
     # code here
```
3. If the answer is NOT in the context, respond with EXACTLY:
   "I could not find this topic in your notes. \
Please check your study material directly."
   Do NOT try to answer from general knowledge.
   Do NOT say "based on my training" or "generally speaking".
4. NEVER make up, assume, or infer information not present in the context.
5. NEVER mention that you are using context or that you have a context window.
   Just answer naturally as a tutor would.

CONTEXT:
{context}
"""


def build_prompt() -> ChatPromptTemplate:
    """
    Build and return the ChatPromptTemplate for the RAG chain.

    The template has TWO message types:
        system  → SYSTEM_PROMPT with {context} placeholder
                  Sets LLM behaviour and fills in retrieved chunks
        human   → "{input}" placeholder
                  Filled with the user's question at inference time

    WHY two separate messages instead of one big string?
    Chat models (like Llama, GPT) are trained to respond differently
    to "system" vs "human" messages. System messages set constraints.
    Human messages represent the actual query.
    Mixing them into one message reduces instruction-following quality.

    Returns:
        ChatPromptTemplate: ready to use as a step in the LCEL chain

    Raises:
        Exception: if template construction fails (rare — no I/O involved)
    """
    logger.info(f"Building prompt template | version={PROMPT_VERSION}")

    try:
        prompt = ChatPromptTemplate.from_messages([
            # ── Message 1: System ─────────────────────────────────────
            # Sets the LLM's role, rules, and injects the retrieved context.
            # {context} is filled by format_docs() in llm_client.py
            # before this prompt is sent to the LLM.
            ("system", SYSTEM_PROMPT),

            # ── Message 2: Human ──────────────────────────────────────
            # The user's actual question.
            # {input} is filled by RunnablePassthrough in llm_client.py
            # with the (possibly rewritten) standalone query.
            ("human", "{input}")
        ])

        logger.info(
            f"Prompt template built successfully | "
            f"version={PROMPT_VERSION}"
        )
        return prompt

    except Exception as e:
        logger.error(
            f"Failed to build prompt template | "
            f"version={PROMPT_VERSION} | "
            f"error={e}",
            exc_info=True
        )
        raise   # re-raise → startup fails with clear error message