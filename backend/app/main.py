"""FastAPI application: POST /triage and GET /health.

Concurrency is bounded here so the Task 3 outage scenario (≈50 tickets/second,
each wanting a model call) degrades instead of falling over:

- An asyncio.Semaphore caps in-flight triages. Beyond that, requests wait only
  up to ``request_queue_timeout_seconds`` and then get a clean 429 + Retry-After
  (backpressure) rather than piling up unbounded.
- The blocking classify call runs in a thread so the event loop stays responsive.
- When the model itself is the bottleneck, the provider chain's circuit breaker
  trips and triage degrades to rules — the user still gets a reply.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__
from .config import LLMMode, get_settings
from .llm.chain import ProviderChain
from .llm.gemini import GeminiProvider
from .llm.groq_provider import GroqProvider
from .llm.replay import ReplayProvider
from .logging_config import configure_logging, get_logger
from .schemas import HealthResponse, Ticket, TriageResult
from .storage import Storage
from .triage import TriageService

log = get_logger()


def _build_chain(settings) -> ProviderChain:
    if settings.llm_mode == LLMMode.replay:
        providers = [ReplayProvider(settings.fixtures_path)]
    else:
        providers = [GeminiProvider(settings), GroqProvider(settings)]
    return ProviderChain(
        providers,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        breaker_threshold=settings.circuit_breaker_threshold,
        breaker_reset_seconds=settings.circuit_breaker_reset_seconds,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging()
    app.state.settings = settings
    app.state.storage = Storage(settings.database_url)
    app.state.chain = _build_chain(settings)
    app.state.service = TriageService(app.state.chain, app.state.storage, settings)
    app.state.semaphore = asyncio.Semaphore(settings.max_concurrent_llm_calls)
    log.info("service_started", llm_mode=settings.llm_mode.value, version=__version__)
    yield


app = FastAPI(title="Dhaba Support Triage", version=__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev/demo; lock down to the agent console origin in prod
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = app.state.settings
    return HealthResponse(
        status="ok",
        llm_mode=settings.llm_mode.value,
        circuit=app.state.chain.circuit_state,
        version=__version__,
    )


@app.post("/triage", response_model=TriageResult)
async def triage(ticket: Ticket) -> TriageResult | JSONResponse:
    settings = app.state.settings
    sem: asyncio.Semaphore = app.state.semaphore

    # Backpressure: wait briefly for a slot, else shed load with 429.
    try:
        await asyncio.wait_for(sem.acquire(), timeout=settings.request_queue_timeout_seconds)
    except asyncio.TimeoutError:
        log.warning("load_shed", ticket_id=ticket.ticket_id)
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers={"Retry-After": "5"},
            content={"detail": "Service busy, please retry.", "ticket_id": ticket.ticket_id},
        )
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, app.state.service.triage, ticket)
        return result
    finally:
        sem.release()


@app.get("/")
async def root():
    return {"service": "dhaba-triage", "version": __version__, "docs": "/docs"}
