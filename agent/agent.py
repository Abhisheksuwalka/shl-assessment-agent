"""
agent.py — Step 3.5: Agent Orchestrator

Wires classifier → extractor → retriever → LLM → response formatter
into a single Agent.chat() method that returns a valid ChatResponse.

Handles all 5 intent paths:
  CLARIFY   → ask one targeted question
  RECOMMEND → retrieve + generate explanation
  REFINE    → merge constraints, re-retrieve + generate explanation
  COMPARE   → retrieve both items, generate comparison
  REFUSE    → polite refusal, no recommendations
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from groq import Groq

from api.models import ChatResponse, Message, Recommendation
from agent.classifier import classify_intent
from agent.extractor import extract_constraints, build_search_query
from agent.retriever import Retriever, SearchConstraints
from agent.prompts import (
    SYSTEM_PROMPT,
    REFUSAL_MESSAGE,
    build_context_block,
    build_recommendation_prompt,
    build_clarification_prompt,
)

load_dotenv()
logger = logging.getLogger(__name__)

# LLM model for full conversational responses
RESPONSE_MODEL = "llama-3.3-70b-versatile"
MAX_TURNS = 8


class Agent:
    """
    SHL Assessment Recommendation Agent.

    Stateless per call — receives full conversation history,
    routes through intent classifier, and returns a ChatResponse.
    """

    def __init__(self, retriever: Retriever, groq_client: Groq | None = None):
        self._retriever = retriever
        self._client = groq_client or Groq(api_key=os.environ.get("GROQ_API_KEY"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, messages: list[Message]) -> ChatResponse:
        """
        Main entry point. Takes full stateless conversation history.
        Returns a ChatResponse with reply, recommendations, end_of_conversation.
        """
        if not messages:
            return ChatResponse(
                reply="Hello! I'm the SHL Assessment Advisor. What role are you hiring for?",
                recommendations=[],
                end_of_conversation=False,
            )

        turn_count = sum(1 for m in messages if m.role == "user")
        logger.info("Agent.chat() — turn %d / %d", turn_count, MAX_TURNS)

        # ── Classify intent ──────────────────────────────────────────
        intent = classify_intent(messages, client=self._client)
        logger.info("Intent: %s", intent)

        # ── REFUSE path ──────────────────────────────────────────────
        if intent == "REFUSE":
            return ChatResponse(
                reply=REFUSAL_MESSAGE,
                recommendations=[],
                end_of_conversation=False,
            )

        # ── Extract constraints from full conversation ────────────────
        # For REFINE: pass previous constraints so they are merged additively
        previous_constraints = self._extract_previous_constraints(messages)
        constraints, constraints_summary = extract_constraints(
            messages,
            previous_constraints=previous_constraints if intent == "REFINE" else None,
            client=self._client,
        )

        # ── CLARIFY path ─────────────────────────────────────────────
        if intent == "CLARIFY":
            # If we're near the turn limit, force a best-effort recommendation
            if turn_count >= MAX_TURNS - 1:
                logger.info("Turn limit approaching — forcing RECOMMEND")
                intent = "RECOMMEND"
            else:
                reply = self._generate_clarification(messages, constraints_summary)
                return ChatResponse(
                    reply=reply,
                    recommendations=[],
                    end_of_conversation=False,
                )

        # ── RECOMMEND / REFINE path ───────────────────────────────────
        if intent in ("RECOMMEND", "REFINE"):
            return self._handle_recommend(
                messages, constraints, constraints_summary, intent
            )

        # ── COMPARE path ──────────────────────────────────────────────
        if intent == "COMPARE":
            return self._handle_compare(messages, constraints, constraints_summary)

        # ── Fallback (should never reach here) ────────────────────────
        logger.warning("Unhandled intent '%s' — falling back to CLARIFY", intent)
        return ChatResponse(
            reply="Could you tell me more about the role you're hiring for?",
            recommendations=[],
            end_of_conversation=False,
        )

    # ------------------------------------------------------------------
    # Intent handlers
    # ------------------------------------------------------------------

    def _handle_recommend(
        self,
        messages: list[Message],
        constraints: SearchConstraints,
        constraints_summary: str,
        intent: str,
    ) -> ChatResponse:
        """Retrieve assessments and generate a recommendation reply."""
        query = build_search_query(messages, constraints_summary)
        logger.info("Search query: '%s'", query[:100])

        # Get raw results (score, item) for rich context block
        raw_results = self._retriever.search_raw(query, constraints=constraints, n=10)

        if not raw_results:
            logger.warning("No retrieval results — returning honest empty response")
            return ChatResponse(
                reply=(
                    "I wasn't able to find assessments matching those specific requirements "
                    "in the SHL catalog. Could you broaden the criteria slightly?"
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        # Build context block for LLM
        context_block = build_context_block(raw_results)
        rec_prompt = build_recommendation_prompt(context_block, constraints_summary, intent)

        # Generate conversational reply
        reply = self._llm_call(
            system=SYSTEM_PROMPT,
            user_prompt=rec_prompt,
            conversation_history=messages,
            max_tokens=400,
        )

        # Build Recommendation objects — URLs verbatim from catalog
        recommendations = [
            self._retriever._to_recommendation(item)
            for _, item in raw_results
        ]

        # end_of_conversation=True on first committed recommendation response
        eoc = intent == "RECOMMEND"

        return ChatResponse(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=eoc,
        )

    def _handle_compare(
        self,
        messages: list[Message],
        constraints: SearchConstraints,
        constraints_summary: str,
    ) -> ChatResponse:
        """Retrieve relevant assessments and generate a comparison reply."""
        # Extract assessment names from the last user message for targeted retrieval
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        query = f"compare assessments: {last_user}"
        raw_results = self._retriever.search_raw(query, constraints=None, n=6)

        context_block = build_context_block(raw_results)
        compare_prompt = build_recommendation_prompt(context_block, constraints_summary, "COMPARE")

        reply = self._llm_call(
            system=SYSTEM_PROMPT,
            user_prompt=compare_prompt,
            conversation_history=messages,
            max_tokens=500,
        )

        return ChatResponse(
            reply=reply,
            recommendations=[],
            end_of_conversation=False,
        )

    # ------------------------------------------------------------------
    # Helper: Clarification generator (Step 3.3)
    # ------------------------------------------------------------------

    def _generate_clarification(
        self,
        messages: list[Message],
        constraints_summary: str,
    ) -> str:
        """
        Generate a single targeted clarifying question.
        Tracks questions already asked to avoid repetition.
        """
        # Extract previous assistant messages (questions already asked)
        previous_questions = [
            m.content for m in messages if m.role == "assistant"
        ]

        prompt = build_clarification_prompt(
            known_constraints_summary=constraints_summary,
            questions_already_asked=previous_questions,
        )

        return self._llm_call(
            system=SYSTEM_PROMPT,
            user_prompt=prompt,
            conversation_history=[],  # No history needed for clarification
            max_tokens=100,
        )

    # ------------------------------------------------------------------
    # Helper: Extract previous constraints for REFINE merging
    # ------------------------------------------------------------------

    def _extract_previous_constraints(
        self, messages: list[Message]
    ) -> SearchConstraints | None:
        """
        Extract constraints from all messages EXCEPT the last user turn.
        Used for REFINE merging — lets us know what was established before.
        """
        if len(messages) <= 1:
            return None
        previous_messages = messages[:-1]
        if not any(m.role == "user" for m in previous_messages):
            return None
        try:
            constraints, _ = extract_constraints(
                previous_messages, client=self._client
            )
            return constraints
        except Exception:
            return None

    # ------------------------------------------------------------------
    # LLM call helper
    # ------------------------------------------------------------------

    def _llm_call(
        self,
        system: str,
        user_prompt: str,
        conversation_history: list[Message],
        max_tokens: int = 400,
    ) -> str:
        """
        Call Groq LLM with system prompt + optional conversation history + user prompt.
        Returns the assistant's text response.
        """
        groq_messages = [{"role": "system", "content": system}]

        # Inject conversation history (for context continuity)
        for msg in conversation_history:
            groq_messages.append({"role": msg.role, "content": msg.content})

        groq_messages.append({"role": "user", "content": user_prompt})

        try:
            response = self._client.chat.completions.create(
                model=RESPONSE_MODEL,
                messages=groq_messages,
                max_tokens=max_tokens,
                temperature=0.3,  # Slight creativity for conversational quality
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return (
                "I'm having trouble generating a response right now. "
                "Please try again in a moment."
            )
