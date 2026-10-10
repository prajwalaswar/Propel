"""Triage orchestrator — the pipeline that boxes the model in on both sides.

Order of operations (see README architecture diagram):

    idempotency hit?  -> return stored result, no model, no second refund
    scan injection    -> flag text aimed at the assistant
    call provider     -> raw text (replay / gemini / groq)
    parse + validate  -> LLMClassification, with repair -> retry -> degrade
    refund gate       -> deterministic policy decides the money
    execute refund    -> once, via the ledger
    persist + audit   -> redacted, correlation-id tagged
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone

from pydantic import ValidationError

from .config import Settings
from .idempotency import refund_key
from .injection import scan as scan_injection
from .llm.base import SYSTEM_PROMPT, build_user_payload
from .llm.base import LLMError
from .llm.chain import ProviderChain
from .llm.replay import with_ticket_id
from .logging_config import get_logger
from .refund_policy import decide_refund
from .schemas import (
    Category,
    ConfidenceBand,
    LLMClassification,
    RefundHint,
    Severity,
    Ticket,
    TriageFlags,
    TriageResult,
)
from .storage import Storage

log = get_logger()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _billing_summary(ticket: Ticket) -> str:
    """A compact, number-bearing summary for the model. No user identity."""
    b = ticket.billing
    parts = [
        f"plan={b.plan}", f"state={b.state.value if b.state else None}",
        f"trial_amount={b.trial_amount}", f"plan_amount={b.plan_amount}",
        f"last_charge_amount={b.last_charge_amount}", f"currency={b.currency}",
        f"last_charge_at={b.last_charge_at}", f"auto_renew={b.auto_renew}",
    ]
    return ", ".join(parts)


_JSON_SPAN = re.compile(r"\{.*\}", re.DOTALL)


def _parse_classification(raw: str) -> tuple[LLMClassification | None, bool]:
    """Return (classification, was_repaired). None means unrecoverable."""
    # First, the happy path.
    try:
        return LLMClassification.model_validate_json(raw), False
    except (ValidationError, ValueError):
        pass

    # Repair pass: strip code fences and pull out the first {...} span. This is
    # the real handler for "the model returned prose / truncated / fenced JSON".
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    match = _JSON_SPAN.search(cleaned)
    if match:
        try:
            return LLMClassification.model_validate_json(match.group(0)), True
        except (ValidationError, ValueError):
            pass
    return None, True


def _degraded_classification(ticket: Ticket) -> LLMClassification:
    """Rules-only fallback when the model is unusable (down, or output could not
    be validated after repair). Deliberately conservative: it never guesses a
    refund direction and always wants a human."""
    msg = ticket.message.lower()
    state = ticket.billing.state

    if any(w in msg for w in ("refund", "paisa wapas", "money back", "wapas")):
        category = Category.refund_request
    elif any(w in msg for w in ("cancel", "band karo", "stop", "unsubscribe")):
        category = Category.cancellation
    elif state and state.value == "failed":
        category = Category.payment_failed
    elif any(w in msg for w in ("renew", "renewal", "again charged", "auto")):
        category = Category.accidental_renewal
    else:
        category = Category.other

    return LLMClassification(
        category=category,
        severity=Severity.medium,
        reply_draft=(
            "Thanks for reaching out. We've received your message and a support "
            "specialist will review your account and get back to you shortly."
        ),
        detected_language=ticket.language_hint or "unknown",
        refund_hint=RefundHint.unsure,
        confidence=0.3,
        reasoning="Degraded mode: classified by rules without a model call.",
    )


def _confidence_band(confidence: float, settings: Settings) -> ConfidenceBand:
    if confidence < settings.low_confidence_threshold:
        return ConfidenceBand.low
    if confidence < 0.8:
        return ConfidenceBand.medium
    return ConfidenceBand.high


class TriageService:
    def __init__(self, chain: ProviderChain, storage: Storage, settings: Settings):
        self._chain = chain
        self._storage = storage
        self._settings = settings

    def triage(self, ticket: Ticket) -> TriageResult:
        correlation_id = "CORR-" + uuid.uuid4().hex[:12]

        # --- 1. Idempotency: have we already answered this ticket? ----------
        cached = self._storage.get_result(ticket.ticket_id)
        if cached is not None:
            cached_result = TriageResult.model_validate(cached)
            cached_result.flags.replayed_from_cache = True
            cached_result.correlation_id = correlation_id
            self._storage.append_audit(
                _now(), correlation_id, ticket.ticket_id, "idempotent_hit", None
            )
            log.info("idempotent_hit", correlation_id=correlation_id,
                     ticket_id=ticket.ticket_id)
            return cached_result

        # --- 2. Injection scan (deterministic, provable in Task 2) ----------
        injection_signals = scan_injection(ticket.message)
        injection_detected = bool(injection_signals)

        # --- 3. Classify (replay / live), with repair -> degrade ------------
        classification, flags, source = self._classify(ticket, correlation_id)

        low_confidence = classification.confidence < self._settings.low_confidence_threshold

        # --- 4. The money gate: deterministic, model never writes a number --
        refund = decide_refund(
            ticket, classification, self._settings,
            injection_detected=injection_detected,
            low_confidence=low_confidence,
        )

        # --- 5. Execute refund at most once (idempotent) --------------------
        if refund.should_refund and refund.executed:
            key = refund_key(ticket.ticket_id, refund.policy_rule_id)
            first_time = self._storage.record_refund(
                key, ticket.ticket_id, refund.amount, refund.currency,
                refund.policy_rule_id, _now(),
            )
            if not first_time:
                refund.executed = False
                refund.reason += " (already executed previously; not paid again)"

        # --- 6. Assemble result --------------------------------------------
        needs_human = (
            low_confidence
            or injection_detected
            or refund.requires_human_approval
            or classification.severity == Severity.urgent
        )
        flags.prompt_injection_detected = injection_detected
        flags.injection_signals = injection_signals

        result = TriageResult(
            ticket_id=ticket.ticket_id,
            category=classification.category,
            severity=classification.severity,
            refund=refund,
            reply_draft=classification.reply_draft,
            needs_human=needs_human,
            confidence=classification.confidence,
            confidence_band=_confidence_band(classification.confidence, self._settings),
            detected_language=classification.detected_language,
            flags=flags,
            correlation_id=correlation_id,
            model_source=source,
        )

        # --- 7. Persist for idempotency + audit (PII-free) ------------------
        self._storage.save_result(ticket.ticket_id, result.model_dump(mode="json"), _now())
        self._storage.append_audit(
            _now(), correlation_id, ticket.ticket_id, "triage_completed",
            {
                "category": result.category.value,
                "severity": result.severity.value,
                "refund_rule": refund.policy_rule_id,
                "refund_amount": refund.amount,
                "needs_human": needs_human,
                "injection": injection_detected,
                "degraded": result.flags.degraded,
                "source": result.model_source,
            },
        )
        log.info(
            "triage_completed", correlation_id=correlation_id,
            ticket_id=ticket.ticket_id, category=result.category.value,
            severity=result.severity.value, refund_rule=refund.policy_rule_id,
            refund_amount=refund.amount, needs_human=needs_human,
            injection=injection_detected, degraded=result.flags.degraded,
            source=result.model_source,
        )
        return result

    def _classify(
        self, ticket: Ticket, correlation_id: str
    ) -> tuple[LLMClassification, TriageFlags, str]:
        flags = TriageFlags()
        source = "degraded-rules"
        payload = build_user_payload(
            message=ticket.message, billing_summary=_billing_summary(ticket)
        )
        payload = with_ticket_id(payload, ticket.ticket_id)
        try:
            raw, provider_name = self._chain.classify(
                system_prompt=SYSTEM_PROMPT, user_payload=payload
            )
            classification, repaired = _parse_classification(raw)
            flags.schema_repaired = repaired
            if classification is None:
                log.warning("schema_unrecoverable", correlation_id=correlation_id,
                            ticket_id=ticket.ticket_id, provider=provider_name)
                classification = _degraded_classification(ticket)
                flags.degraded = True
            else:
                source = provider_name
        except LLMError as exc:
            log.warning("llm_unavailable_degrading", correlation_id=correlation_id,
                        ticket_id=ticket.ticket_id, error=str(exc))
            classification = _degraded_classification(ticket)
            flags.degraded = True
        return classification, flags, source
