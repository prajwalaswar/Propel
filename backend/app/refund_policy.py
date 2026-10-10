"""The money gate — deterministic refund policy.

This is the part the assignment says is "most of our read on you": a refund
decision must NOT be a field the model can freely write. So the flow is:

    model  -> suggests a category + a refund *hint* (none/partial/full)
    THIS   -> reads the authoritative billing context and decides the actual
              amount and whether to execute, using documented rules only.

The model's hint is advisory. Every number here comes from ``billing``, and no
refund can ever exceed what was actually charged (``last_charge_amount``).
High-value or low-confidence cases are proposed, not executed, and sent to a
human.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .config import Settings
from .schemas import (
    BillingState,
    Category,
    LLMClassification,
    RefundDecision,
    Ticket,
)


def _charged_amount(ticket: Ticket) -> float:
    """The hard cap on any refund: what we actually took from this user."""
    b = ticket.billing
    if b.last_charge_amount is not None:
        return float(b.last_charge_amount)
    # Fall back to the plan/trial amount implied by state, never the model.
    if b.state == BillingState.renewal and b.plan_amount is not None:
        return float(b.plan_amount)
    if b.plan_amount is not None:
        return float(b.plan_amount)
    if b.trial_amount is not None:
        return float(b.trial_amount)
    return 0.0


def _within_window(ticket: Ticket, settings: Settings) -> bool:
    """Is the charge recent enough for an automated refund?"""
    raw = ticket.billing.last_charge_at or ticket.billing.plan_activated_at
    if not raw:
        return True  # unknown date -> don't block on it, but a human still checks
    try:
        charged = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return True
    if charged.tzinfo is None:
        charged = charged.replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - charged).days
    return age_days <= settings.refund_window_days


def decide_refund(
    ticket: Ticket,
    classification: LLMClassification,
    settings: Settings,
    *,
    injection_detected: bool,
    low_confidence: bool,
) -> RefundDecision:
    """Pure function: (ticket, classification) -> RefundDecision. No I/O, no model."""

    cap = _charged_amount(ticket)
    state = ticket.billing.state
    category = classification.category
    currency = ticket.billing.currency or "INR"

    # ---- Rule 0: nothing was ever charged -> nothing to refund. ------------
    if state == BillingState.failed:
        return RefundDecision(
            should_refund=False,
            amount=0.0,
            currency=currency,
            reason="Charge failed, so no money left the user's account; nothing to refund.",
            policy_rule_id="NO_CHARGE_FAILED_STATE",
        )
    if state == BillingState.initiated:
        return RefundDecision(
            should_refund=False,
            amount=0.0,
            currency=currency,
            reason="Payment only initiated, not captured; no refund applies yet.",
            policy_rule_id="NO_CHARGE_INITIATED_STATE",
        )
    if cap <= 0:
        return RefundDecision(
            should_refund=False,
            amount=0.0,
            currency=currency,
            reason="No positive charge on record for this user.",
            policy_rule_id="NO_CHARGE_ON_RECORD",
        )

    # ---- Safety gates: never auto-move money we are unsure about. ----------
    # An injected ticket demanding a refund must NOT get one automatically.
    manipulated = injection_detected
    needs_review = low_confidence or manipulated
    within_window = _within_window(ticket, settings)

    # Decide the eligible amount per category (always <= cap, always from data).
    eligible, rule_id, reason = _eligible_amount(ticket, category, cap, currency)

    if eligible <= 0:
        return RefundDecision(
            should_refund=False, amount=0.0, currency=currency,
            reason=reason, policy_rule_id=rule_id,
        )

    requires_human = (
        needs_review
        or not within_window
        or eligible >= settings.refund_human_approval_threshold_inr
    )

    # If a human must approve, we PROPOSE the refund but do not execute it.
    if requires_human:
        detail = []
        if manipulated:
            detail.append("ticket contains text aimed at the assistant")
        if low_confidence:
            detail.append("model confidence below threshold")
        if not within_window:
            detail.append(f"charge older than {settings.refund_window_days}d window")
        if eligible >= settings.refund_human_approval_threshold_inr:
            detail.append("amount at/above human-approval threshold")
        return RefundDecision(
            should_refund=True,
            amount=round(min(eligible, cap), 2),
            currency=currency,
            reason=f"{reason} Proposed, pending human approval ({'; '.join(detail)}).",
            policy_rule_id=rule_id,
            requires_human_approval=True,
            executed=False,
        )

    # Clear, in-window, low-value, confident -> auto-approve.
    return RefundDecision(
        should_refund=True,
        amount=round(min(eligible, cap), 2),
        currency=currency,
        reason=reason,
        policy_rule_id=rule_id,
        requires_human_approval=False,
        executed=True,
    )


def _eligible_amount(
    ticket: Ticket, category: Category, cap: float, currency: str
) -> tuple[float, str, str]:
    """Map (category, billing) -> (amount, rule_id, reason). Amount is capped."""
    b = ticket.billing

    if category == Category.accidental_renewal and b.state == BillingState.renewal:
        return cap, "RENEWAL_FULL_REFUND", (
            f"Accidental auto-renewal of {currency} {cap:g}; refunding the renewal charge "
            "and turning off auto-renew."
        )

    if category == Category.trial_charge_dispute:
        # If the real plan already auto-activated (successful), refund the plan
        # charge. If only the Rs.1 trial was taken, refund the Rs.1.
        if b.state == BillingState.successful and b.plan_amount:
            amt = min(float(b.plan_amount), cap)
            return amt, "TRIAL_AUTOACTIVATION_REFUND", (
                f"Trial auto-activated into a {currency} {amt:g} plan the user did not intend; "
                "refunding the plan charge."
            )
        trial = float(b.trial_amount) if b.trial_amount else cap
        return min(trial, cap), "TRIAL_FEE_REFUND", (
            f"Refunding the {currency} {min(trial, cap):g} trial fee."
        )

    if category == Category.refund_request:
        # Explicit ask: refund what is on record, subject to the gates above.
        return cap, "EXPLICIT_REQUEST_REFUND", (
            f"User explicitly requested a refund of the {currency} {cap:g} charge on record."
        )

    if category == Category.cancellation:
        # Cancellation stops future billing; it is not itself a refund.
        return 0.0, "CANCEL_NO_REFUND", (
            "Cancellation stops future renewals; the past charge is not auto-refunded."
        )

    # payment_failed / billing_question / product_support / other
    return 0.0, "NO_REFUND_CATEGORY", (
        "Category does not warrant a refund on its own."
    )
