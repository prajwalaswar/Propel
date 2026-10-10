"""Pydantic models — the contract that keeps the output honest.

These models serve three jobs at once:

1. They are the public request/response schema of ``POST /triage``.
2. ``LLMClassification`` is what we force the *model* output to parse into. If
   the model returns prose, truncated JSON or an invented category, parsing
   fails here and the orchestrator takes the repair/degrade path. This is the
   "the output holds its shape" requirement made real.
3. ``RefundDecision`` is produced ONLY by the deterministic policy engine, never
   by the model. The model may *suggest* via ``LLMClassification.refund_hint``
   but it can never write the amount we act on.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# Enums — the small, closed vocabularies the model must stay inside.
# --------------------------------------------------------------------------- #
class Category(str, Enum):
    """What the ticket is about. Deliberately small and documented."""

    trial_charge_dispute = "trial_charge_dispute"      # upset about the Rs.1 / auto-activation
    accidental_renewal = "accidental_renewal"          # charged for a renewal they did not want
    payment_failed = "payment_failed"                  # a charge failed; cannot subscribe
    refund_request = "refund_request"                  # explicit "give my money back"
    cancellation = "cancellation"                      # wants to cancel / stop auto-renew
    billing_question = "billing_question"              # general "what is this charge"
    product_support = "product_support"                # non-billing app help
    other = "other"                                    # fallback when nothing fits


class Severity(str, Enum):
    """Operational urgency, documented in the README.

    low      -> informational, no money at stake.
    medium   -> billing confusion, money involved but not time-critical.
    high     -> money moved against the user's wishes, or clear dissatisfaction.
    urgent   -> legal/chargeback/public threat, or repeated failed charges.
    """

    low = "low"
    medium = "medium"
    high = "high"
    urgent = "urgent"


class ConfidenceBand(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class BillingState(str, Enum):
    """The four states a Dhaba purchase can be in (from the assignment)."""

    initiated = "initiated"
    failed = "failed"
    successful = "successful"
    renewal = "renewal"


# --------------------------------------------------------------------------- #
# Input — a ticket exactly as it appears in dhaba_tickets.json.
# Unknown fields are tolerated so the real file can be richer than this.
# --------------------------------------------------------------------------- #
class BillingContext(BaseModel):
    """What billing knew about this user when the ticket landed.

    The amounts here are authoritative. The refund engine reads
    ``last_charge_amount`` as the hard cap on any refund; the model is never
    consulted for a number.
    """

    model_config = {"extra": "allow"}

    plan: str | None = None                      # "trial" | "monthly" | "yearly"
    state: BillingState | None = None
    currency: str = "INR"
    trial_amount: float | None = None            # typically 1
    plan_amount: float | None = None             # 249 (monthly) or 1499 (yearly)
    last_charge_amount: float | None = None      # what was actually taken
    last_charge_at: str | None = None
    trial_started_at: str | None = None
    plan_activated_at: str | None = None
    auto_renew: bool | None = None
    payment_method: str | None = None


class UserContext(BaseModel):
    """Identity fields. Every attribute here is PII and must be redacted in logs."""

    model_config = {"extra": "allow"}

    user_id: str | None = None
    name: str | None = None
    email: str | None = None
    phone: str | None = None


class Ticket(BaseModel):
    model_config = {"extra": "allow"}

    ticket_id: str
    message: str
    created_at: str | None = None
    channel: str | None = "in_app_help_form"
    language_hint: str | None = None
    user: UserContext = Field(default_factory=UserContext)
    billing: BillingContext = Field(default_factory=BillingContext)

    @field_validator("ticket_id", "message")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must not be blank")
        return v


# --------------------------------------------------------------------------- #
# Model output — what we force the LLM's text to parse into.
# Parsing failure here is the trigger for repair/retry/degrade.
# --------------------------------------------------------------------------- #
class RefundHint(str, Enum):
    """The model may only *suggest* a direction. It can never name an amount."""

    none = "none"
    partial = "partial"
    full = "full"
    unsure = "unsure"


class LLMClassification(BaseModel):
    """Strict schema the model must fill. Note what is absent: no refund amount,
    no 'should_refund' boolean the model controls. Money is decided elsewhere."""

    model_config = {"extra": "forbid"}

    category: Category
    severity: Severity
    reply_draft: str = Field(min_length=1, max_length=2000)
    detected_language: str = Field(default="unknown", max_length=40)
    refund_hint: RefundHint = RefundHint.unsure
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(default="", max_length=1000)


# --------------------------------------------------------------------------- #
# Refund — produced ONLY by the deterministic policy engine.
# --------------------------------------------------------------------------- #
class RefundDecision(BaseModel):
    should_refund: bool
    amount: float = 0.0
    currency: str = "INR"
    reason: str
    policy_rule_id: str                 # which rule fired, for audit
    requires_human_approval: bool = False
    executed: bool = False              # did we actually record the refund now?


# --------------------------------------------------------------------------- #
# Public response of POST /triage.
# --------------------------------------------------------------------------- #
class TriageFlags(BaseModel):
    prompt_injection_detected: bool = False
    injection_signals: list[str] = Field(default_factory=list)
    degraded: bool = False              # produced without the model (fallback path)
    schema_repaired: bool = False       # model output needed a repair pass
    replayed_from_cache: bool = False   # idempotent hit — identical prior result


class TriageResult(BaseModel):
    ticket_id: str
    category: Category
    severity: Severity
    refund: RefundDecision
    reply_draft: str
    needs_human: bool
    confidence: float
    confidence_band: ConfidenceBand
    detected_language: str = "unknown"
    flags: TriageFlags = Field(default_factory=TriageFlags)
    correlation_id: str
    model_source: str = "unknown"       # gemini | groq | replay | degraded-rules


class HealthResponse(BaseModel):
    status: str
    llm_mode: str
    circuit: str
    version: str
