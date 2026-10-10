"""Groq provider (fallback, live mode).

Groq is fast and cheap, so it is the natural second leg: if Gemini times out or
is rate-limited during a spike, we fall back here before degrading to rules.
"""

from __future__ import annotations

from ..config import Settings
from .base import LLMError, LLMProvider


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        if not self._settings.groq_api_key:
            raise LLMError("GROQ_API_KEY is not set")
        try:
            from groq import Groq  # lazy import
        except ImportError as exc:  # pragma: no cover - env dependent
            raise LLMError("groq sdk not installed") from exc
        self._client = Groq(api_key=self._settings.groq_api_key)

    def classify(self, *, system_prompt: str, user_payload: str) -> str:
        self._ensure_client()
        try:
            resp = self._client.chat.completions.create(
                model=self._settings.groq_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_payload},
                ],
                temperature=0.1,
                max_tokens=800,
                response_format={"type": "json_object"},
            )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise LLMError("groq returned empty text")
            return text
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"groq call failed: {exc}") from exc
