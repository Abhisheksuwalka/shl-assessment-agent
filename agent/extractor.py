"""
extractor.py — Step 3.2: Context Extractor

Extracts structured SearchConstraints from the full conversation history.
Uses Groq LLM to parse out: job_levels, test_types, duration_max, languages, adaptive.
On REFINE intent: merges new constraints with previous ones (additive).
"""

import json
import logging
import os
import re
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from api.models import Message
from agent.prompts import CONSTRAINT_EXTRACTOR_PROMPT
from agent.retriever import SearchConstraints

load_dotenv()
logger = logging.getLogger(__name__)

EXTRACTOR_MODEL = "llama-3.3-70b-versatile"

# Valid job level values (from catalog)
VALID_JOB_LEVELS = {
    "Director", "Entry-Level", "Executive", "General Population",
    "Graduate", "Manager", "Mid-Professional", "Front Line Manager",
    "Supervisor", "Professional Individual Contributor",
}

# Valid type codes
VALID_TYPE_CODES = {"K", "P", "A", "B", "C", "E", "S", "D"}


def _format_conversation(messages: list[Message]) -> str:
    """Format conversation for LLM input."""
    lines = []
    for msg in messages:
        role = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def _extract_json(text: str) -> dict[str, Any]:
    """
    Extract JSON from LLM output — handles markdown fences and surrounding text.
    """
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown fences
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try extracting bare JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning("Could not extract JSON from extractor output: %s", text[:200])
    return {}


def _parse_constraints(data: dict[str, Any]) -> SearchConstraints:
    """
    Parse and validate the extracted JSON into a SearchConstraints object.
    Filters out invalid values to prevent downstream errors.
    """
    # Job levels — filter to valid values only
    raw_levels = data.get("job_levels") or []
    job_levels = [jl for jl in raw_levels if jl in VALID_JOB_LEVELS]

    # Test types — filter to valid codes only
    raw_types = data.get("test_types") or []
    test_types = [tt for tt in raw_types if tt in VALID_TYPE_CODES]

    # Duration — must be positive integer or None
    raw_duration = data.get("duration_max")
    duration_max: int | None = None
    if raw_duration is not None:
        try:
            val = int(raw_duration)
            duration_max = val if val > 0 else None
        except (ValueError, TypeError):
            pass

    # Languages — take as-is (partial match is handled in retriever)
    languages = list(data.get("languages") or [])

    # Adaptive — must be bool or None
    raw_adaptive = data.get("adaptive")
    adaptive: bool | None = None
    if isinstance(raw_adaptive, bool):
        adaptive = raw_adaptive

    return SearchConstraints(
        job_levels=job_levels,
        duration_max=duration_max,
        languages=languages,
        test_types=test_types,
        adaptive=adaptive,
    )


def _merge_constraints(base: SearchConstraints, new: SearchConstraints) -> SearchConstraints:
    """
    Merge new constraints into base (additive — REFINE intent).
    Lists are unioned; scalars prefer new value if set.
    """
    merged_levels = list(set(base.job_levels) | set(new.job_levels))
    merged_types = list(set(base.test_types) | set(new.test_types))
    merged_langs = list(set(base.languages) | set(new.languages))

    return SearchConstraints(
        job_levels=merged_levels,
        duration_max=new.duration_max if new.duration_max is not None else base.duration_max,
        languages=merged_langs,
        test_types=merged_types,
        adaptive=new.adaptive if new.adaptive is not None else base.adaptive,
    )


def extract_constraints(
    messages: list[Message],
    previous_constraints: SearchConstraints | None = None,
    client: Groq | None = None,
) -> tuple[SearchConstraints, str]:
    """
    Extract structured constraints from the conversation history.

    Args:
        messages: Full stateless conversation history
        previous_constraints: Constraints from previous turn (for REFINE merging)
        client: Optional Groq client

    Returns:
        Tuple of (SearchConstraints, constraints_summary_string)
    """
    if client is None:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    conversation_text = _format_conversation(messages)

    try:
        response = client.chat.completions.create(
            model=EXTRACTOR_MODEL,
            messages=[
                {"role": "system", "content": CONSTRAINT_EXTRACTOR_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Conversation:\n{conversation_text}\n\n"
                        "Extract the hiring constraints as JSON. Output ONLY valid JSON."
                    ),
                },
            ],
            max_tokens=300,
            temperature=0.0,
        )

        raw_output = response.choices[0].message.content.strip()
        logger.info("Extractor raw output: %s", raw_output[:300])

        data = _extract_json(raw_output)
        constraints = _parse_constraints(data)

        # Merge with previous constraints if provided (REFINE)
        if previous_constraints is not None:
            constraints = _merge_constraints(previous_constraints, constraints)

        # Build a human-readable summary for logging and prompt context
        summary_parts = []
        if data.get("job_role"):
            summary_parts.append(f"Role: {data['job_role']}")
        if data.get("skills"):
            summary_parts.append(f"Skills: {', '.join(data['skills'])}")
        if constraints.job_levels:
            summary_parts.append(f"Levels: {', '.join(constraints.job_levels)}")
        if constraints.test_types:
            summary_parts.append(f"Types: {', '.join(constraints.test_types)}")
        if constraints.duration_max:
            summary_parts.append(f"Max duration: {constraints.duration_max} min")
        if constraints.languages:
            summary_parts.append(f"Languages: {', '.join(constraints.languages)}")
        if constraints.adaptive is not None:
            summary_parts.append(f"Adaptive: {constraints.adaptive}")

        summary = "; ".join(summary_parts) if summary_parts else "No specific constraints extracted."
        logger.info("Extracted constraints: %s", summary)

        return constraints, summary

    except Exception as exc:
        logger.error("Constraint extraction failed: %s", exc)
        return SearchConstraints(), "Constraint extraction failed."


def build_search_query(messages: list[Message], constraints_summary: str) -> str:
    """
    Build a natural language search query from the conversation context.
    Uses the last user message + constraints summary for richer retrieval.
    """
    last_user_msgs = [m.content for m in messages if m.role == "user"]
    last_query = last_user_msgs[-1] if last_user_msgs else ""

    if constraints_summary and constraints_summary != "No specific constraints extracted.":
        return f"{last_query}. {constraints_summary}"
    return last_query
