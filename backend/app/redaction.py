"""PII redaction for logs.

Requirement from the assignment: "nothing in these tickets that identifies a
person or a company should end up in a log line." We never log raw ticket text
or user fields. Everything that goes to a log passes through ``redact`` first,
wired in as a structlog processor so it is applied centrally rather than relying
on every call site to remember.
"""

from __future__ import annotations

import re

# Order matters: emails before the generic word patterns.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[email]"),
    # Indian + international phone shapes: +91 98765 43210, 9876543210, etc.
    (re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){9,13}\d(?!\d)"), "[phone]"),
    # UPI handles like name@okhdfcbank
    (re.compile(r"\b[\w.-]+@(?:ok[a-z]+|paytm|ybl|axl|upi)\b", re.I), "[upi]"),
    # Card-ish 13-16 digit runs
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[card]"),
]

# Field names whose *values* are always PII regardless of content.
_SENSITIVE_KEYS = {
    "email",
    "phone",
    "name",
    "user_id",
    "message",
    "reply_draft",
    "address",
    "upi",
    "card",
}

# Structural / non-PII keys whose values must pass through untouched. Without
# this, the phone/card regexes happily "redact" a correlation id or timestamp
# that merely contains a run of digits, destroying traceability.
_SAFE_KEYS = {
    "correlation_id",
    "ticket_id",
    "event",
    "level",
    "timestamp",
    "logger",
    "category",
    "severity",
    "refund_rule",
    "policy_rule_id",
    "confidence_band",
    "source",
    "detected_language",
}


def redact_text(value: str) -> str:
    out = value
    for pattern, replacement in _PATTERNS:
        out = pattern.sub(replacement, out)
    return out


def redact(obj):
    """Recursively redact a value destined for a log line.

    - sensitive keys get their value replaced wholesale;
    - every string is still scrubbed for emails/phones/cards even under a
      non-sensitive key, because people paste their email into free text.
    """
    if isinstance(obj, dict):
        clean = {}
        for key, val in obj.items():
            if key.lower() in _SAFE_KEYS:
                clean[key] = val  # structural, never PII — leave exactly as-is
            elif key.lower() in _SENSITIVE_KEYS and val is not None:
                clean[key] = "[redacted]"
            else:
                clean[key] = redact(val)
        return clean
    if isinstance(obj, (list, tuple)):
        return [redact(v) for v in obj]
    if isinstance(obj, str):
        return redact_text(obj)
    return obj
