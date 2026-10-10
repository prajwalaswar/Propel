"""Application configuration.

Everything that changes between a laptop, a CI box and production lives here and
is driven by environment variables (12-factor style). The single most important
switch is LLM_MODE:

    replay  -> no network, no API key. Classifications come from recorded
               fixtures keyed by ticket id. This is the mode the evaluator runs
               so they never burn a token.
    live    -> call a real provider (Gemini, then Groq as fallback).
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent


class LLMMode(str, Enum):
    replay = "replay"
    live = "live"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- runtime mode -------------------------------------------------------
    llm_mode: LLMMode = Field(
        default=LLMMode.replay,
        description="replay = fixtures/no key, live = real provider calls.",
    )

    # --- provider credentials (only needed when llm_mode=live) --------------
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"

    # --- reliability knobs --------------------------------------------------
    llm_timeout_seconds: float = 12.0
    llm_max_retries: int = 1
    # Trip the circuit after this many consecutive provider failures; while the
    # circuit is open we skip the model entirely and degrade to rules.
    circuit_breaker_threshold: int = 5
    circuit_breaker_reset_seconds: float = 30.0
    # Bound the number of in-flight model calls so a 50 req/s spike queues
    # instead of opening 50 sockets at once.
    max_concurrent_llm_calls: int = 8
    request_queue_timeout_seconds: float = 8.0

    # --- refund policy ------------------------------------------------------
    # Any proposed refund at or above this value must be approved by a human
    # before it is executed, regardless of what the model said.
    refund_human_approval_threshold_inr: int = 1000
    # How many days after a charge an automated refund is allowed.
    refund_window_days: int = 7
    # Below this model confidence we never auto-act; the ticket goes to a human.
    low_confidence_threshold: float = 0.55

    # --- storage ------------------------------------------------------------
    database_url: str = f"sqlite:///{(BASE_DIR / 'triage.db').as_posix()}"

    # --- data files ---------------------------------------------------------
    tickets_path: Path = BASE_DIR / "data" / "dhaba_tickets.json"
    fixtures_path: Path = BASE_DIR / "fixtures" / "llm_responses.json"

    # --- logging ------------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
