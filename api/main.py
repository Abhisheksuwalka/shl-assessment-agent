"""
FastAPI application entrypoint.
Uses lifespan context manager to load shared resources once at startup.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import health, chat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: Load FAISS index + embedding model (once), then create Agent.
    These are stored on app.state so every request reuses them without reloading.
    """
    from agent.retriever import Retriever
    from agent.agent import Agent

    logger.info("SHL Assessment Advisor starting up...")

    logger.info("Loading Retriever (FAISS index + embedding model)...")
    retriever = Retriever.load()
    logger.info("Retriever ready — %d catalog items indexed.", retriever.catalog_size)

    logger.info("Creating Agent...")
    agent = Agent(retriever=retriever)
    logger.info("Agent ready.")

    app.state.retriever = retriever
    app.state.agent = agent

    logger.info("Startup complete. Server ready.")
    yield
    logger.info("SHL Assessment Advisor shutting down.")


app = FastAPI(
    title="SHL Assessment Recommendation API",
    description=(
        "Conversational FastAPI agent that helps hiring managers discover "
        "the right SHL assessments through dialogue."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow CORS for development / evaluation harness
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router, tags=["Health"])
app.include_router(chat.router, tags=["Chat"])
