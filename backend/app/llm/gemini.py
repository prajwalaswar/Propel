"""Google Gemini provider (primary, live mode).

Uses the ``google-genai`` SDK. The import is done lazily so the service runs in
replay mode on a machine that never installed the SDK or set a key.
"""

from __future__ import annotations

from ..config import Settings
from .base import LLMError, LLMProvider


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        if not self._settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        try:
            from google import genai  # lazy import
        except ImportError as exc:  # pragma: no cover - env dependent
            raise LLMError("google-genai not installed") from exc
        self._client = genai.Client(api_key=self._settings.gemini_api_key)

    def classify(self, *, system_prompt: str, user_payload: str) -> str:
        self._ensure_client()
        try:
            from google.genai import types

            resp = self._client.models.generate_content(
                model=self._settings.gemini_model,
                contents=user_payload,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    temperature=0.1,
                    max_output_tokens=800,
                ),
            )
            text = (resp.text or "").strip()
            if not text:
                raise LLMError("gemini returned empty text")
            return text
        except LLMError:
            raise
        except Exception as exc:  # network, quota, SDK errors -> uniform failure
            raise LLMError(f"gemini call failed: {exc}") from exc
