"""Logs are clean: PII never reaches a log line; structural fields survive."""

from __future__ import annotations

from app.redaction import redact, redact_text


def test_email_redacted():
    assert "rohan@example.com" not in redact_text("mail me at rohan@example.com please")
    assert "[email]" in redact_text("mail me at rohan@example.com please")


def test_phone_redacted():
    out = redact_text("call +91 98200 11223 now")
    assert "98200" not in out
    assert "[phone]" in out


def test_sensitive_keys_wholesale_redacted():
    event = {"user_id": "u_123", "email": "a@b.com", "name": "Rohan", "message": "hi"}
    clean = redact(event)
    assert clean["user_id"] == "[redacted]"
    assert clean["email"] == "[redacted]"
    assert clean["name"] == "[redacted]"
    assert clean["message"] == "[redacted]"


def test_structural_keys_preserved():
    event = {"correlation_id": "CORR-d4209b17c0", "ticket_id": "DH-1001", "level": "info"}
    clean = redact(event)
    assert clean["correlation_id"] == "CORR-d4209b17c0"  # not mangled by phone regex
    assert clean["ticket_id"] == "DH-1001"


def test_nested_free_text_scrubbed():
    event = {"detail": {"note": "user said email is x.y@z.com and card 4111 1111 1111 1111"}}
    clean = redact(event)
    assert "x.y@z.com" not in clean["detail"]["note"]
    assert "4111" not in clean["detail"]["note"]
