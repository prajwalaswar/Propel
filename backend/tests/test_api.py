"""HTTP-level tests for the service, driven through FastAPI's TestClient.

These exercise the real ASGI app end to end (lifespan, concurrency wrapper,
response model) rather than calling ``TriageService`` directly, so they prove the
endpoint itself behaves — not just the orchestrator underneath it.

Replay mode is the default, so no API key is needed. Each test gets its own
temp-file SQLite DB so idempotency state doesn't leak between tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "replay")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'api_test.db').as_posix()}")
    from app.config import get_settings

    get_settings.cache_clear()
    from app.main import app

    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


def _raw_ticket(ticket_id: str) -> dict:
    from app.config import get_settings

    data = json.loads(Path(get_settings().tickets_path).read_text(encoding="utf-8"))
    return next(t for t in data if t["ticket_id"] == ticket_id)


def test_health_reports_replay_mode(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["llm_mode"] == "replay"
    assert body["circuit"] in {"closed", "half_open", "open"}


def test_triage_returns_schema_valid_result(client):
    r = client.post("/triage", json=_raw_ticket("DH-1001"))
    assert r.status_code == 200
    body = r.json()
    # The documented contract fields are all present.
    for key in (
        "ticket_id", "category", "severity", "refund", "reply_draft",
        "needs_human", "confidence", "confidence_band", "correlation_id",
    ):
        assert key in body
    assert body["ticket_id"] == "DH-1001"
    assert body["category"] == "trial_charge_dispute"
    # Low-value, in-window, confident -> auto refund executed.
    assert body["refund"]["should_refund"] is True
    assert body["refund"]["executed"] is True
    assert body["refund"]["amount"] == 249
    assert body["needs_human"] is False
    assert body["flags"]["replayed_from_cache"] is False


def test_same_ticket_twice_is_served_from_cache(client):
    first = client.post("/triage", json=_raw_ticket("DH-1001")).json()
    second = client.post("/triage", json=_raw_ticket("DH-1001")).json()
    # First time is a fresh computation, second is the idempotent replay.
    assert first["flags"]["replayed_from_cache"] is False
    assert second["flags"]["replayed_from_cache"] is True
    # Same decision returned; the money layer never issues a second refund.
    assert second["refund"]["amount"] == first["refund"]["amount"]
    assert second["category"] == first["category"]


def test_injection_ticket_is_flagged_and_refused(client):
    r = client.post("/triage", json=_raw_ticket("DH-1007"))
    assert r.status_code == 200
    body = r.json()
    assert body["flags"]["prompt_injection_detected"] is True
    assert body["needs_human"] is True
    # It demanded a full refund; it must not have gotten one.
    assert body["refund"]["should_refund"] is False
    assert body["refund"]["amount"] == 0


def test_blank_message_is_rejected_by_validation(client):
    r = client.post("/triage", json={"ticket_id": "DH-X", "message": "   "})
    assert r.status_code == 422  # Pydantic rejects a blank message
