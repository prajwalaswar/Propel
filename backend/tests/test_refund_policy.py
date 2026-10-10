"""The money gate: refunds are decided by rules, capped, and gated."""

from __future__ import annotations

from app.config import Settings
from app.refund_policy import decide_refund
from app.schemas import (
    BillingContext,
    BillingState,
    Category,
    LLMClassification,
    RefundHint,
    Severity,
    Ticket,
)


def _classification(category: Category, confidence: float = 0.9) -> LLMClassification:
    return LLMClassification(
        category=category, severity=Severity.medium, reply_draft="x",
        detected_language="english", refund_hint=RefundHint.full, confidence=confidence,
    )


def _ticket(**billing) -> Ticket:
    return Ticket(ticket_id="T", message="m", billing=BillingContext(**billing))


def test_failed_charge_never_refunds():
    s = Settings()
    t = _ticket(state=BillingState.failed, last_charge_amount=None, plan_amount=249)
    d = decide_refund(t, _classification(Category.payment_failed), s,
                      injection_detected=False, low_confidence=False)
    assert d.should_refund is False
    assert d.policy_rule_id == "NO_CHARGE_FAILED_STATE"


def test_initiated_state_never_refunds():
    s = Settings()
    t = _ticket(state=BillingState.initiated, last_charge_amount=None)
    d = decide_refund(t, _classification(Category.billing_question), s,
                      injection_detected=False, low_confidence=False)
    assert d.should_refund is False


def test_low_value_in_window_auto_executes():
    s = Settings()
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    t = _ticket(state=BillingState.successful, last_charge_amount=249, plan_amount=249,
                last_charge_at=now)
    d = decide_refund(t, _classification(Category.trial_charge_dispute), s,
                      injection_detected=False, low_confidence=False)
    assert d.should_refund is True
    assert d.executed is True
    assert d.amount == 249


def test_high_value_requires_human():
    s = Settings()
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    t = _ticket(state=BillingState.renewal, last_charge_amount=1499, plan_amount=1499,
                last_charge_at=now)
    d = decide_refund(t, _classification(Category.accidental_renewal), s,
                      injection_detected=False, low_confidence=False)
    assert d.should_refund is True
    assert d.requires_human_approval is True
    assert d.executed is False


def test_refund_never_exceeds_charge():
    """Even if the model screams 'full refund', the amount is capped at the
    actual charge on record."""
    s = Settings()
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    t = _ticket(state=BillingState.successful, last_charge_amount=249, plan_amount=249,
                last_charge_at=now)
    d = decide_refund(t, _classification(Category.refund_request), s,
                      injection_detected=False, low_confidence=False)
    assert d.amount <= 249


def test_injection_blocks_auto_refund():
    s = Settings()
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    t = _ticket(state=BillingState.successful, last_charge_amount=249, plan_amount=249,
                last_charge_at=now)
    d = decide_refund(t, _classification(Category.refund_request), s,
                      injection_detected=True, low_confidence=False)
    # An injected refund demand must never auto-execute.
    assert d.executed is False
    assert d.requires_human_approval is True


def test_low_confidence_never_auto_executes():
    s = Settings()
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    t = _ticket(state=BillingState.successful, last_charge_amount=249, plan_amount=249,
                last_charge_at=now)
    d = decide_refund(t, _classification(Category.refund_request), s,
                      injection_detected=False, low_confidence=True)
    assert d.executed is False


def test_out_of_window_requires_human():
    s = Settings()
    old = "2026-01-01T00:00:00+00:00"
    t = _ticket(state=BillingState.successful, last_charge_amount=249, plan_amount=249,
                last_charge_at=old)
    d = decide_refund(t, _classification(Category.refund_request), s,
                      injection_detected=False, low_confidence=False)
    assert d.requires_human_approval is True
    assert d.executed is False


def test_cancellation_is_not_a_refund():
    s = Settings()
    t = _ticket(state=BillingState.renewal, last_charge_amount=249, plan_amount=249)
    d = decide_refund(t, _classification(Category.cancellation), s,
                      injection_detected=False, low_confidence=False)
    assert d.should_refund is False
    assert d.policy_rule_id == "CANCEL_NO_REFUND"
