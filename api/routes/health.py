"""
Health check endpoint — no auth, no logic.
Returns {"status": "ok"} with HTTP 200.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    """Simple liveness probe. Required by SHL evaluator."""
    return {"status": "ok"}
