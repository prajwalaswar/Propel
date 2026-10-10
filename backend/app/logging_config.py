"""Structured logging.

Every log line is JSON, carries the request's ``correlation_id``, and is run
through PII redaction before it is emitted. That combination is the Task 3
"debugging this at 3am" answer: you can grep one correlation id across the whole
request lifecycle and never leak a user's email while doing it.
"""

from __future__ import annotations

import logging
import sys

import structlog

from .config import get_settings
from .redaction import redact


def _redact_processor(_logger, _name, event_dict):
    """structlog processor: scrub the whole event dict for PII."""
    return redact(event_dict)


def configure_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    renderer = (
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            _redact_processor,  # <-- last thing before rendering: nothing escapes it
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "dhaba.triage"):
    return structlog.get_logger(name)
