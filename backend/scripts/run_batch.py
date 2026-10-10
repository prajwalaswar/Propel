"""Task 2 batch runner.

Runs every ticket in dhaba_tickets.json through the triage service in REPLAY
mode (no API key, deterministic) and prints:

  1. a Markdown results table ready to paste into the README, and
  2. the single dashboard metric we'd watch to know the service still works.

Usage (from the backend/ directory):
    python -m scripts.run_batch
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running as `python -m scripts.run_batch` or `python scripts/run_batch.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import LLMMode, Settings  # noqa: E402
from app.llm.chain import ProviderChain  # noqa: E402
from app.llm.replay import ReplayProvider  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402
from app.schemas import Ticket  # noqa: E402
from app.storage import Storage  # noqa: E402
from app.triage import TriageService  # noqa: E402


def build_service() -> tuple[TriageService, Settings]:
    settings = Settings(llm_mode=LLMMode.replay, database_url="sqlite:///:memory:", log_json=False)
    chain = ProviderChain(
        [ReplayProvider(settings.fixtures_path)],
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        breaker_threshold=settings.circuit_breaker_threshold,
        breaker_reset_seconds=settings.circuit_breaker_reset_seconds,
    )
    storage = Storage(settings.database_url)
    return TriageService(chain, storage, settings), settings


def refund_cell(r) -> str:
    if not r.refund.should_refund:
        return "no"
    tag = "PROPOSED (human)" if r.refund.requires_human_approval else "AUTO"
    return f"{r.refund.currency} {r.refund.amount:g} -> {tag}"


def main() -> None:
    configure_logging()
    service, settings = build_service()
    tickets = json.loads(settings.tickets_path.read_text(encoding="utf-8"))

    results = []
    for raw in tickets:
        result = service.triage(Ticket.model_validate(raw))
        results.append(result)

    # ---- Markdown table --------------------------------------------------
    print("\n| Ticket | Category | Sev | Refund | Needs human | Injection | Conf |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        inj = "YES" if r.flags.prompt_injection_detected else "-"
        print(
            f"| {r.ticket_id} | {r.category.value} | {r.severity.value} "
            f"| {refund_cell(r)} | {'yes' if r.needs_human else 'no'} "
            f"| {inj} | {r.confidence:.2f} ({r.confidence_band.value}) |"
        )

    # ---- The one dashboard number ---------------------------------------
    total = len(results)
    degraded = sum(1 for r in results if r.flags.degraded)
    repaired = sum(1 for r in results if r.flags.schema_repaired)
    needs_human = sum(1 for r in results if r.needs_human)
    injections = sum(1 for r in results if r.flags.prompt_injection_detected)
    auto = total - needs_human

    print("\n--- metrics ---")
    print(f"tickets processed      : {total}")
    print(f"auto-resolved (no human): {auto} ({auto / total:.0%})")
    print(f"routed to human        : {needs_human} ({needs_human / total:.0%})")
    print(f"prompt injections caught: {injections}")
    print(f"schema repaired/degraded: {repaired}/{degraded}")
    print(
        "\nDASHBOARD METRIC -> model_output_integrity_rate = "
        f"{(total - repaired - degraded) / total:.0%} "
        "(share of tickets whose model output validated first try, no repair/degrade). "
        "A drop here is the earliest sign the prompt or model has drifted."
    )


if __name__ == "__main__":
    main()
