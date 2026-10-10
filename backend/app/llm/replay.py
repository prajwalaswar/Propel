"""Replay provider — the no-API-key mode.

Classifications are read from ``fixtures/llm_responses.json``, keyed by
ticket_id. This is what makes the evaluator's run deterministic and free: the
whole pipeline (validation, refund gate, idempotency, redaction) executes
exactly as in live mode, only the network call is swapped for a lookup.

If a ticket id is not in the fixtures we raise LLMError so the orchestrator
takes its real degrade path rather than silently inventing an answer.
"""

from __future__ import annotations

import json
from pathlib import Path

from .base import LLMError, LLMProvider


class ReplayProvider(LLMProvider):
    name = "replay"

    def __init__(self, fixtures_path: Path):
        self._fixtures_path = fixtures_path
        self._fixtures: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self._fixtures_path.exists():
            with open(self._fixtures_path, encoding="utf-8") as fh:
                self._fixtures = json.load(fh)

    def classify(self, *, system_prompt: str, user_payload: str) -> str:
        # The orchestrator passes the ticket id in the payload header line so we
        # can look it up without changing the interface shared with live mode.
        ticket_id = _extract_ticket_id(user_payload)
        record = self._fixtures.get(ticket_id)
        if record is None:
            raise LLMError(f"no fixture for ticket_id={ticket_id!r}")
        return json.dumps(record)


_TICKET_ID_MARKER = "#ticket_id="


def with_ticket_id(payload: str, ticket_id: str) -> str:
    """Prefix the payload with a marker the replay provider can read. Harmless
    to live providers — it is just one more line of context."""
    return f"{_TICKET_ID_MARKER}{ticket_id}\n{payload}"


def _extract_ticket_id(payload: str) -> str:
    first_line = payload.splitlines()[0] if payload else ""
    if first_line.startswith(_TICKET_ID_MARKER):
        return first_line[len(_TICKET_ID_MARKER):].strip()
    return ""
