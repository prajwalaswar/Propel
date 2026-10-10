"""Shared test fixtures: a triage service wired for replay mode + in-memory DB."""

from __future__ import annotations

import json

import pytest

from app.config import LLMMode, Settings
from app.llm.chain import ProviderChain
from app.llm.replay import ReplayProvider
from app.schemas import Ticket
from app.storage import Storage
from app.triage import TriageService


@pytest.fixture
def settings() -> Settings:
    return Settings(llm_mode=LLMMode.replay, database_url="sqlite:///:memory:", log_json=False)


@pytest.fixture
def storage(settings) -> Storage:
    return Storage(settings.database_url)


@pytest.fixture
def service(settings, storage) -> TriageService:
    chain = ProviderChain(
        [ReplayProvider(settings.fixtures_path)],
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        breaker_threshold=settings.circuit_breaker_threshold,
        breaker_reset_seconds=settings.circuit_breaker_reset_seconds,
    )
    return TriageService(chain, storage, settings)


@pytest.fixture
def tickets(settings) -> dict[str, Ticket]:
    raw = json.loads(settings.tickets_path.read_text(encoding="utf-8"))
    return {t["ticket_id"]: Ticket.model_validate(t) for t in raw}
