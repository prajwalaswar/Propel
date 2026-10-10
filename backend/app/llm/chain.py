"""Provider chain + circuit breaker.

Tries each provider in order (Gemini -> Groq in live mode; just Replay in replay
mode) with a per-call timeout. A simple circuit breaker tracks consecutive
failures: once it trips, calls fail fast for a cooldown window instead of
hammering a provider that is already down — which is exactly what you want when
50 tickets a second are all trying the same dead endpoint.

``classify`` raises LLMError only when *every* provider fails. The orchestrator
turns that into the degraded, rules-only path.
"""

from __future__ import annotations

import concurrent.futures
import time

from .base import LLMError, LLMProvider


class _CircuitBreaker:
    def __init__(self, threshold: int, reset_seconds: float):
        self._threshold = threshold
        self._reset_seconds = reset_seconds
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if time.monotonic() - self._opened_at >= self._reset_seconds:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        return self.state != "open"

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_at = time.monotonic()


class ProviderChain:
    def __init__(
        self,
        providers: list[LLMProvider],
        *,
        timeout_seconds: float,
        max_retries: int,
        breaker_threshold: int,
        breaker_reset_seconds: float,
    ):
        self._providers = providers
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._breaker = _CircuitBreaker(breaker_threshold, breaker_reset_seconds)
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    @property
    def circuit_state(self) -> str:
        return self._breaker.state

    def classify(self, *, system_prompt: str, user_payload: str) -> tuple[str, str]:
        """Return (raw_text, provider_name) or raise LLMError if all fail."""
        if not self._breaker.allow():
            raise LLMError("circuit open: skipping providers, degrading")

        last_error: Exception | None = None
        for provider in self._providers:
            for attempt in range(self._max_retries + 1):
                try:
                    text = self._call_with_timeout(provider, system_prompt, user_payload)
                    self._breaker.record_success()
                    return text, provider.name
                except Exception as exc:  # noqa: BLE001 - uniform failure handling
                    last_error = exc
                    # Only count hard failures toward the breaker, not the last
                    # retry of the last provider spuriously.
                    continue
        self._breaker.record_failure()
        raise LLMError(f"all providers failed: {last_error}")

    def _call_with_timeout(self, provider, system_prompt, user_payload) -> str:
        future = self._pool.submit(
            provider.classify, system_prompt=system_prompt, user_payload=user_payload
        )
        try:
            return future.result(timeout=self._timeout)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise LLMError(f"{provider.name} timed out after {self._timeout}s") from exc
