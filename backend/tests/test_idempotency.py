"""It runs twice safely: idempotency at the response and the money layers."""

from __future__ import annotations


def test_same_ticket_twice_is_cached(service, tickets):
    first = service.triage(tickets["DH-1001"])
    assert first.flags.replayed_from_cache is False

    second = service.triage(tickets["DH-1001"])
    assert second.flags.replayed_from_cache is True
    # Same decision returned, not recomputed.
    assert second.category == first.category
    assert second.refund.amount == first.refund.amount


def test_refund_recorded_only_once(service, storage, tickets):
    # DH-1001 auto-executes a 249 refund on first pass.
    service.triage(tickets["DH-1001"])
    from app.idempotency import refund_key
    key = refund_key("DH-1001", "TRIAL_AUTOACTIVATION_REFUND")
    assert storage.refund_exists(key) is True

    # Attempting to record the same refund again must no-op (no double pay).
    recorded_again = storage.record_refund(
        key, "DH-1001", 249, "INR", "TRIAL_AUTOACTIVATION_REFUND", "2026-01-01T00:00:00Z"
    )
    assert recorded_again is False
