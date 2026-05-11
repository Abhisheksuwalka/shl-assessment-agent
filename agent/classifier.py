"""
classifier.py — Step 3.1: Intent Classifier

Classifies the latest user turn into one of:
  CLARIFY, RECOMMEND, REFINE, COMPARE, REFUSE

Uses Groq (Llama-3.3-70b) for fast, cheap classification.
Falls back to CLARIFY on any ambiguous or unexpected output.
"""

import logging
import os
from typing import Literal

from dotenv import load_dotenv
from groq import Groq

from api.models import Message
from agent.prompts import INTENT_CLASSIFIER_PROMPT

load_dotenv()
logger = logging.getLogger(__name__)

Intent = Literal["CLARIFY", "RECOMMEND", "REFINE", "COMPARE", "REFUSE"]
VALID_INTENTS: set[str] = {"CLARIFY", "RECOMMEND", "REFINE", "COMPARE", "REFUSE"}

# Use a small, fast model just for classification — saves tokens
CLASSIFIER_MODEL = "llama-3.3-70b-versatile"


def _format_conversation(messages: list[Message]) -> str:
    """Format the conversation history for the classifier prompt."""
    lines = []
    for msg in messages:
        role = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def classify_intent(messages: list[Message], client: Groq | None = None) -> Intent:
    """
    Classify the latest user turn into an intent.

    Args:
        messages: Full conversation history (stateless — full history per call)
        client: Optional Groq client (if None, creates one from env)

    Returns:
        Intent string: one of CLARIFY, RECOMMEND, REFINE, COMPARE, REFUSE
    """
    if not messages:
        logger.warning("classify_intent called with empty messages — defaulting to CLARIFY")
        return "CLARIFY"

    # Check if the last message is from user
    last_msg = messages[-1]
    if last_msg.role != "user":
        logger.warning("Last message is not from user — defaulting to CLARIFY")
        return "CLARIFY"

    if client is None:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    conversation_text = _format_conversation(messages)

    try:
        response = client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            messages=[
                {"role": "system", "content": INTENT_CLASSIFIER_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Conversation:\n{conversation_text}\n\n"
                        f"Classify the latest user message. Output ONLY the intent label."
                    ),
                },
            ],
            max_tokens=10,
            temperature=0.0,  # Deterministic — classification needs consistency
        )

        raw = response.choices[0].message.content.strip().upper()
        logger.info("Classifier raw output: '%s'", raw)

        # Extract valid intent from output (handles extra punctuation/spaces)
        for intent in VALID_INTENTS:
            if intent in raw:
                logger.info("Classified as: %s", intent)
                return intent  # type: ignore[return-value]

        # Fallback: default to CLARIFY (safest option)
        logger.warning(
            "Classifier returned unrecognised output '%s' — falling back to CLARIFY", raw
        )
        return "CLARIFY"

    except Exception as exc:
        logger.error("Intent classification failed: %s — defaulting to CLARIFY", exc)
        return "CLARIFY"
