"""
Chat endpoint — POST /chat
Stateless conversational endpoint. Accepts full conversation history.
Returns a ChatResponse with reply, recommendations, and end_of_conversation.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from api.models import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Thread pool for running the synchronous Agent.chat() in a non-blocking way
_executor = ThreadPoolExecutor(max_workers=4)

TIMEOUT_SECONDS = 25  # Buffer under the 30s hard limit


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    """
    Stateless conversational endpoint.
    Accepts full conversation history; returns a ChatResponse.

    - Validates request via Pydantic (auto-raises 422 on bad schema)
    - Calls Agent.chat() in a thread pool (non-blocking for async server)
    - Enforces 25s timeout (under the 30s hard limit)
    - Returns safe fallback on timeout or unhandled error
    """
    agent = request.app.state.agent
    loop = asyncio.get_event_loop()

    try:
        response: ChatResponse = await asyncio.wait_for(
            loop.run_in_executor(_executor, agent.chat, body.messages),
            timeout=TIMEOUT_SECONDS,
        )
        return response

    except asyncio.TimeoutError:
        logger.error("Agent timed out after %ds", TIMEOUT_SECONDS)
        return ChatResponse(
            reply=(
                "I'm taking longer than expected to respond. "
                "Please try again — I'll have an answer for you shortly."
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    except Exception as exc:
        logger.exception("Unhandled error in /chat: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal error"},
        )

