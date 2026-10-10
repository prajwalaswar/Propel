"""Prompt-injection detection.

Task 2 says two of the twelve tickets "contain text addressed to the assistant
rather than to us." A support ticket is *data*, not instructions. This module
flags tickets that try to steer the model, so the orchestrator can (a) refuse to
follow them and (b) route to a human.

Two layers of defence, both needed:

1. Detection here — cheap, deterministic signals we can log and prove in Task 2.
2. Isolation in the prompt (see ``llm/prompts.py``) — the user text is fenced and
   the system prompt tells the model it is untrusted. Detection alone is not a
   guarantee; the fence is what actually blunts a missed one.
"""

from __future__ import annotations

import re

# Phrases that only make sense if the writer is talking to the AI, not to Dhaba
# support. Kept readable so a reviewer can see exactly what we match on.
_SIGNATURES: list[tuple[str, re.Pattern[str]]] = [
    ("ignore_instructions", re.compile(r"ignore\s+(all\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts?)", re.I)),
    ("disregard", re.compile(r"disregard\s+(the\s+)?(system|previous|all)", re.I)),
    ("role_override", re.compile(r"you\s+are\s+(now\s+)?(a|an|chatgpt|gemini|the\s+assistant|dan\b)", re.I)),
    ("system_prompt_probe", re.compile(r"(system|developer)\s+prompt", re.I)),
    ("instruction_to_model", re.compile(r"\b(as an ai|as a language model|respond with|output the following|print the following)\b", re.I)),
    ("forced_approval", re.compile(r"(always|must|you have to)\s+(approve|refund|grant|issue)\b", re.I)),
    ("full_refund_command", re.compile(r"(give|issue|approve|send)\s+(me\s+)?(a\s+)?(full|complete|maximum)\s+refund", re.I)),
    ("fake_authorization", re.compile(r"(i am|this is)\s+(the\s+)?(admin|developer|ceo|founder|from propel|an engineer)", re.I)),
    ("jailbreak_marker", re.compile(r"(jailbreak|bypass\s+(the\s+)?(rules|policy|filter)|developer mode)", re.I)),
    ("delimiter_injection", re.compile(r"(```|</?system>|\[/?INST\]|<\|.*?\|>)", re.I)),
]


def scan(message: str) -> list[str]:
    """Return the list of injection signal names found in the message."""
    hits: list[str] = []
    for name, pattern in _SIGNATURES:
        if pattern.search(message):
            hits.append(name)
    return hits


def is_injection(message: str) -> bool:
    return bool(scan(message))
