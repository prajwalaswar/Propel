"""Idempotency keys.

Two independent guarantees:

1. Response idempotency — same ticket_id in again returns the stored result
   (handled by Storage.get_result / save_result, keyed on ticket_id).
2. Refund idempotency — the money is only moved once. The key is derived from
   the ticket_id and the firing rule, so re-processing the same ticket maps to
   the same key and Storage.record_refund's INSERT OR IGNORE no-ops.
"""

from __future__ import annotations

import hashlib


def refund_key(ticket_id: str, rule_id: str) -> str:
    raw = f"{ticket_id}:{rule_id}".encode()
    return "rf_" + hashlib.sha256(raw).hexdigest()[:24]
